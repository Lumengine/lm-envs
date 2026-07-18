"""ANT — the canonical "run forward as fast as possible" locomotion task, on the
MuJoCo ant IMPORTED FROM MJCF. The conventional Ant benchmark (Gym/Brax/IsaacGymEnvs):
joint TORQUE control, reward = forward velocity + alive bonus + uprightness - action
cost. Trains in a few minutes — the simplest "RL works on the engine" demo.

Pipeline: ant.xml (MJCF) -> mujoco-usd-converter -> UsdPhysics -> config-driven prep
(assets/ant.rl.yaml: floating base + an INERT PD drive so torque control is clean) ->
ingest -> RL.

Run via the CLI:
    python train.py --task Ant
    python play.py  --task Ant --checkpoint runs/Ant_.../nn/Ant.pth

obs(34) = base_z + lin_vel_b + ang_vel_b*0.25 + up_proj + heading_b_xy
          + dof_pos_rel + dof_vel*0.05 + prev_actions
action(8) -> joint torque = action * cfg.torque_scale   (eJOINT_FORCE, PD drive inert)
reward = 1.0*v_forward + 0.5*upright + 0.5*alive - 0.005*||action||^2
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # for lumengine_envs
import _bootstrap
_bootstrap.bootstrap()
import lm.rl as rl
from lumengine_envs.config import AntConfig

_ANT = _bootstrap.ASSETS / "ant.xml"               # MuJoCo MJCF (imported via rl.Mjcf)
_CFG = _bootstrap.ASSETS / "ant.rl.yaml"           # floating base + inert drive (prep config)

N_DOF       = 8
GROUND_Z    = 0.0
SUBSTEPS    = 2
ANG_VEL_SCALE = 0.25
DOF_VEL_SCALE = 0.05
CLIP_OBS      = 5.0
MAX_EPISODE_LENGTH = 1000
NUM_OBS = 34

# Reward weights (raw per-step, Brax/Gym ant convention — NOT dt-scaled; reward_shaper=1.0).
W_FORWARD, W_UPRIGHT, W_ALIVE, W_ACTION = 1.0, 0.5, 0.5, 0.005

ANT_PPO_PARAMS = {"params": {
    "network": {"mlp": {"units": [256, 128, 64]}},
    "config": {
        "reward_shaper": {"scale_value": 1.0},
        "critic_coef": 2,
        "bounds_loss_coef": 0.001,
        # The alive+upright bonus (~1.0) makes "stand still" a strong local optimum; some
        # entropy is needed to explore enough to discover a forward gait.
        "entropy_coef": 0.01,
    },
}}


# Declarative observation vector (34) — ObservationManager scales/concats in order.
OBS_TERMS = [
    rl.ObsTerm("height",   rl.observations.base_height),
    rl.ObsTerm("lin_vel",  rl.observations.base_lin_vel),
    rl.ObsTerm("ang_vel",  rl.observations.base_ang_vel,  scale=ANG_VEL_SCALE),
    rl.ObsTerm("up",       rl.observations.up_proj),
    rl.ObsTerm("heading",  rl.observations.heading_xy),
    rl.ObsTerm("dof_pos",  rl.observations.joint_pos_rel),
    rl.ObsTerm("dof_vel",  rl.observations.joint_vel,     scale=DOF_VEL_SCALE),
    rl.ObsTerm("actions",  rl.observations.last_action),
]

# Reset events: teleport home + small joint offset/velocity noise.
EVENT_TERMS = [
    rl.EventTerm("reset_root",   rl.events.reset_root_to_home, mode="reset"),
    rl.EventTerm("reset_joints", rl.events.reset_joints_offset, mode="reset",
                 params={"pos_noise": 0.05, "vel_noise": 0.1}),
]


def _quat_rotate_inv(q, v):
    import torch
    w = q[:, 3:4]; xyz = q[:, 0:3]
    t = 2.0 * torch.cross(xyz, v, dim=1)
    return v - w * t + torch.cross(xyz, t, dim=1)


class AntTask(rl.VecTask):
    """Run forward (+x) as fast as possible on the MJCF-imported ant, torque-controlled."""

    def __init__(self, cfg: AntConfig = None, *, num_envs=None, headless=None):
        # Primary API: an AntConfig. The keyword args are a back-compat path.
        self.cfg = cfg or AntConfig()
        if num_envs is not None:
            self.cfg.num_envs = num_envs
        if headless is not None:
            self.cfg.headless = headless
        self.world = rl.World(num_envs=int(self.cfg.num_envs), env_spacing=self.cfg.env_spacing)
        self.world.add_ground(z=GROUND_Z, friction=1.0)
        self.robot = self.world.add_robot(rl.Mjcf(str(_ANT), config=str(_CFG)), spawn_z=self.cfg.spawn_z)
        sim, runner = self.world.build(
            headless=self.cfg.headless,
            config=rl.SimConfig(substeps=SUBSTEPS, device="auto",
                                # the ant's 8 capsule legs sit near the ground -> many
                                # broadphase pairs; size the GPU buffers generously.
                                gpu_contact_buffer_multiplier=max(2.0, int(self.cfg.num_envs) / 256.0),
                                # Found/lost pair churn outgrows the multiplier-scaled
                                # default — and grows QUADRATICALLY with N here (PhysX
                                # asks 4M at 1024 envs, 64M at 4096; surfaced by the
                                # faulted-scene report — the old silent overflow made the
                                # sim MISS interactions). Suspected warmup transient
                                # (falling ants overlapping neighbor envs' AABBs); the
                                # quadratic sizing matches what PhysX measures until that
                                # is tamed at the source.
                                gpu_found_lost_pairs_capacity=max(
                                    4 << 20,
                                    ((int(self.cfg.num_envs) ** 2) * (4 << 20)) // (1024 * 1024))),
            title="Ant (MJCF, run-forward)")
        sim.play()
        super().__init__(sim, runner, num_obs=NUM_OBS, num_actions=N_DOF, name="Ant",
                         clip_obs=CLIP_OBS, max_episode_length=MAX_EPISODE_LENGTH, seed=int(self.cfg.seed))
        self._nstep = 0

    # -- task hooks ---------------------------------------------------------

    def _capture(self):
        torch = self._torch; dev = self.device
        self._dof = self.sim.acquire_dof_state_tensor()
        self._root = self.sim.acquire_root_state_tensor()
        self.sim.refresh_dof_state_tensor(); self.sim.refresh_root_state_tensor()
        self._default_dof = self.robot.default_dof_positions.unsqueeze(0).repeat(self.num_envs, 1)
        self._world_down = torch.tensor([0.0, 0.0, -1.0], device=dev).repeat(self.num_envs, 1)
        self._world_fwd = torch.tensor([1.0, 0.0, 0.0], device=dev).repeat(self.num_envs, 1)
        self._prev_action = torch.zeros(self.num_envs, N_DOF, device=dev)
        home = torch.zeros(self.num_envs, 13, device=dev)
        home[:, 0:3] = self._root[:, 0:3]
        home[:, 6] = 1.0
        self._home = home
        self._home_z = self._root[:, 2].clone()
        # Fixed command "vx,vy,yaw" from cfg.cmd ("" = none); the ant run-forward reward
        # doesn't use a command, but a fixed forward command is handy as a viewer hint.
        self._cmd = _parse_cmd(self.cfg.cmd)
        # Per-DOF joint armature for numerical stability under direct torque control.
        self.sim.set_dof_armature(torch.full((self.num_envs, N_DOF), self.cfg.armature, device=dev))
        # Managers (declarative). The Ant run-forward reward stays inline (forward + alive +
        # upright - action cost); obs / termination / reset are declarative.
        self.observations = rl.ObservationManager(self, OBS_TERMS)
        self.terminations = rl.TerminationManager(self, [
            rl.TermTerm("fell", rl.terminations.height_below,
                        {"ref_attr": "_home_z", "drop": self.cfg.fall_drop}),
            rl.TermTerm("flipped", rl.terminations.bad_orientation, {"min_up": self.cfg.upright_min}),
        ])
        self.events = rl.EventManager(self, EVENT_TERMS)

    def _pre_physics_step(self, actions):
        # `actions` is already clipped to [-1, 1] by the base; map to joint torque.
        self._prev_action = actions
        self.sim.set_dof_actuation_force_tensor(actions * self.cfg.torque_scale)

    def _compute_observations(self):
        self.sim.refresh_root_state_tensor()
        root, dof = self._root, self._dof
        quat = root[:, 3:7]
        self._lin_world = root[:, 7:10]              # world-frame linear velocity (reward: forward)
        self.base_height = root[:, 2]
        self.lin_vel_b = _quat_rotate_inv(quat, root[:, 7:10])
        self.ang_vel_b = _quat_rotate_inv(quat, root[:, 10:13])
        proj_grav = _quat_rotate_inv(quat, self._world_down)
        self._up_proj = self.up_proj = -proj_grav[:, 2]
        self.heading_b = _quat_rotate_inv(quat, self._world_fwd)
        self.dof_pos_rel = dof[:, :N_DOF, 0] - self._default_dof
        self.dof_vel = dof[:, :N_DOF, 1]
        self.actions = self._prev_action
        self.obs_buf[:] = self.observations.compute()

    def _compute_reward(self):
        forward = self._lin_world[:, 0]                      # +x world velocity
        action_cost = (self._prev_action ** 2).sum(dim=1)
        self.rew_buf = (W_FORWARD * forward
                        + W_UPRIGHT * self._up_proj
                        + W_ALIVE
                        - W_ACTION * action_cost)
        # Declarative termination (relative fall + flipped); the manager reads the state above.
        self.reset_buf = self.terminations.compute()
        height = self._root[:, 2]
        self._nstep += 1
        if self._nstep % 500 == 0:
            print(f"[ant-dbg] step {self._nstep} | ep_len(mean)={float(self.progress_buf.mean()):.1f} "
                  f"| rew={float(self.rew_buf.mean()):.3f} | vx={float(forward.mean()):+.2f} m/s "
                  f"| upright={float(self._up_proj.mean()):.2f} | height={float(height.mean()):.2f} "
                  f"| fails={int(self.reset_buf.sum())}", flush=True)

    def _reset_idx(self, ids):
        self.events.reset(ids)             # teleport home + joint offset/velocity noise
        self._prev_action[ids] = 0.0
        self.progress_buf[ids] = 0.0


def _parse_cmd(s):
    if not s:
        return None
    return [float(x) for x in str(s).split(",")]


def _frame_camera(task):
    try:
        task.runner.frame(eye=(3.0, -3.0, 2.0), target=(0.0, 0.0, 0.4))
    except Exception:
        pass
