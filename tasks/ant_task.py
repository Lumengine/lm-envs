"""ANT — the canonical "run forward as fast as possible" locomotion task, on the
MuJoCo ant IMPORTED FROM MJCF. This is the conventional Ant benchmark (Gym/Brax/
IsaacGymEnvs style): the policy outputs joint TORQUES, and the reward is forward
velocity + an alive bonus + an uprightness term − an action cost. It trains in a
few minutes and is the simplest "RL just works on the engine" demo.

Pipeline: ant.xml (MJCF) -> mujoco-usd-converter -> UsdPhysics -> config-driven prep
(assets/ant.rl.yaml: floating base + an INERT PD drive so torque control is clean) ->
ingest -> RL. Same rl.VecTask scaffolding as anymal_task, sized for the 8-DOF ant.

    LM_RL_VIEW=1 python ant_task.py                 # windowed: watch it train live
    python ant_task.py                              # headless train (rl_games)
    LM_RL_PLAY=runs/.../nn/Ant.pth python ant_task.py   # watch a trained policy

obs(34) = base_z + lin_vel_b + ang_vel_b*0.25 + up_proj + heading_b_xy
          + dof_pos_rel + dof_vel*0.05 + prev_actions
action(8) -> joint torque = action * TORQUE_SCALE   (eJOINT_FORCE, PD drive inert)
reward = 1.0*v_forward + 0.5*upright + 0.5*alive − 0.005*||action||^2
"""
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap
_bootstrap.bootstrap()
import lm.rl as rl

_ANT = _bootstrap.ASSETS / "ant.xml"               # MuJoCo MJCF (imported via rl.Mjcf)
_CFG = _bootstrap.ASSETS / "ant.rl.yaml"           # floating base + inert drive (prep config)

NUM_ENVS    = int(os.environ.get("LM_RL_NUM_ENVS", "4096"))   # PPO wants a big batch
ENV_SPACING = float(os.environ.get("LM_RL_SPACING", "2.5"))
N_DOF       = 8
GROUND_Z    = 0.0
SPAWN_Z     = float(os.environ.get("LM_RL_ANT_SPAWN_Z", "0.62"))   # clear the feet off the ground
SUBSTEPS    = 2

# Torque control: action in [-1,1] -> joint torque = action * TORQUE_SCALE (N*m). The
# ant is light (torso density 5); MuJoCo's gear=150 is for a heavier scale, so a modest
# torque is right here. Tunable via env so a validation run can dial it without an edit.
TORQUE_SCALE = float(os.environ.get("LM_RL_ANT_TORQUE", "15.0"))
ARMATURE     = float(os.environ.get("LM_RL_ANT_ARMATURE", "0.5"))   # joint inertia (stability)

ANG_VEL_SCALE = 0.25
DOF_VEL_SCALE = 0.05
CLIP_OBS      = 5.0
MAX_EPISODE_LENGTH = 1000
NUM_OBS = 34

# Reward weights (raw per-step, Brax/Gym ant convention — NOT dt-scaled; reward_shaper=1.0).
W_FORWARD, W_UPRIGHT, W_ALIVE, W_ACTION = 1.0, 0.5, 0.5, 0.005
# Termination is RELATIVE to the captured spawn height (the ant's natural upright rest is
# ~1.2 m, so an absolute upper bound would spuriously kill it as it settles): fail only if
# it dropped FALL_DROP below spawn (collapsed) or tipped past UPRIGHT_MIN (flipped).
FALL_DROP   = float(os.environ.get("LM_RL_ANT_FALL_DROP", "0.5"))
UPRIGHT_MIN = float(os.environ.get("LM_RL_ANT_UPRIGHT_MIN", "0.2"))

ANT_PPO_PARAMS = {"params": {
    "network": {"mlp": {"units": [256, 128, 64]}},
    "config": {
        "reward_shaper": {"scale_value": 1.0},
        "critic_coef": 2,
        "bounds_loss_coef": 0.001,
        # The alive+upright bonus (~1.0) makes "stand still" a strong local optimum; some
        # entropy is needed for the policy to explore enough to discover a forward gait.
        "entropy_coef": 0.01,
    },
}}


def _quat_rotate_inv(q, v):
    import torch
    w = q[:, 3:4]; xyz = q[:, 0:3]
    t = 2.0 * torch.cross(xyz, v, dim=1)
    return v - w * t + torch.cross(xyz, t, dim=1)


class AntTask(rl.VecTask):
    """Run forward (+x) as fast as possible on the MJCF-imported ant, torque-controlled."""

    def __init__(self, num_envs=NUM_ENVS, headless=True):
        self.world = rl.World(num_envs=int(num_envs), env_spacing=ENV_SPACING)
        self.world.add_ground(z=GROUND_Z, friction=1.0)
        self.robot = self.world.add_robot(rl.Mjcf(str(_ANT), config=str(_CFG)), spawn_z=SPAWN_Z)
        sim, runner = self.world.build(
            headless=headless,
            config=rl.SimConfig(substeps=SUBSTEPS, device="auto",
                                # the ant's 8 capsule legs sit near the ground -> many
                                # broadphase pairs; size the GPU buffers generously.
                                gpu_contact_buffer_multiplier=max(2.0, int(num_envs) / 256.0)),
            title="Ant (MJCF, run-forward)")
        sim.play()
        super().__init__(sim, runner, num_obs=NUM_OBS, num_actions=N_DOF, name="Ant",
                         clip_obs=CLIP_OBS, max_episode_length=MAX_EPISODE_LENGTH,
                         seed=int(os.environ.get("LM_RL_SEED", "0")))
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
        self._home_z = self._root[:, 2].clone()      # per-env spawn height (fall baseline)
        # Per-DOF joint armature for numerical stability under direct torque control
        # (mimics MuJoCo's armature=1). One-time CPU write across roots.
        self.sim.set_dof_armature(torch.full((self.num_envs, N_DOF), ARMATURE, device=dev))

    def _pre_physics_step(self, actions):
        # `actions` is already clipped to [-1, 1] by the base; map to joint torque.
        self._prev_action = actions
        self.sim.set_dof_actuation_force_tensor(actions * TORQUE_SCALE)

    def _compute_observations(self):
        self.sim.refresh_root_state_tensor()
        root, dof = self._root, self._dof
        quat = root[:, 3:7]
        self._lin_world = root[:, 7:10]                      # world-frame linear velocity
        lin_b = _quat_rotate_inv(quat, root[:, 7:10])
        ang_b = _quat_rotate_inv(quat, root[:, 10:13])
        proj_grav = _quat_rotate_inv(quat, self._world_down)
        self._up_proj = -proj_grav[:, 2]                     # 1 = upright, <0 = flipped
        heading_b = _quat_rotate_inv(quat, self._world_fwd)  # world +x in body frame
        o = self.obs_buf
        o[:, 0:1] = root[:, 2:3]                             # torso height
        o[:, 1:4] = lin_b
        o[:, 4:7] = ang_b * ANG_VEL_SCALE
        o[:, 7:8] = self._up_proj.unsqueeze(1)
        o[:, 8:10] = heading_b[:, :2]
        o[:, 10:18] = dof[:, :N_DOF, 0] - self._default_dof
        o[:, 18:26] = dof[:, :N_DOF, 1] * DOF_VEL_SCALE
        o[:, 26:34] = self._prev_action

    def _compute_reward(self):
        torch = self._torch
        forward = self._lin_world[:, 0]                      # +x world velocity
        action_cost = (self._prev_action ** 2).sum(dim=1)
        self.rew_buf = (W_FORWARD * forward
                        + W_UPRIGHT * self._up_proj
                        + W_ALIVE
                        - W_ACTION * action_cost)

        height = self._root[:, 2]
        # RELATIVE fall (dropped below spawn) or flipped — no absolute upper bound (the ant
        # naturally rests ~1.2 m and would trip an absolute cap as it settles).
        fail = (height < self._home_z - FALL_DROP) | (self._up_proj < UPRIGHT_MIN)
        self.reset_buf = fail.float()

        self._nstep += 1
        if self._nstep % 500 == 0:
            print(f"[ant-dbg] step {self._nstep} | ep_len(mean)={float(self.progress_buf.mean()):.1f} "
                  f"| rew={float(self.rew_buf.mean()):.3f} | vx={float(forward.mean()):+.2f} m/s "
                  f"| upright={float(self._up_proj.mean()):.2f} | height={float(height.mean()):.2f} "
                  f"| fails={int(fail.sum())}", flush=True)

    def _reset_idx(self, ids):
        torch = self._torch; n = ids.numel(); dev = self.device
        self._root[ids] = self._home[ids]
        self._dof[ids, :N_DOF, 0] = self._default_dof[ids] + (torch.rand(n, N_DOF, device=dev) - 0.5) * 0.1
        self._dof[ids, :N_DOF, 1] = (torch.rand(n, N_DOF, device=dev) - 0.5) * 0.2
        self.sim.set_root_state_tensor_indexed(self._root, ids)
        self.sim.set_dof_state_tensor_indexed(self._dof, ids)
        self._prev_action[ids] = 0.0
        self.progress_buf[ids] = 0.0


def _frame_camera(task):
    try:
        task.runner.frame(eye=(3.0, -3.0, 2.0), target=(0.0, 0.0, 0.4))
    except Exception:
        pass


if __name__ == "__main__":
    play_ckpt = os.environ.get("LM_RL_PLAY")
    view = os.environ.get("LM_RL_VIEW") == "1"
    headless = os.environ.get("LM_RL_HEADLESS") == "1" or not (play_ckpt or view)
    task = AntTask(num_envs=NUM_ENVS, headless=headless)
    try:
        if not headless:
            _frame_camera(task)
        if play_ckpt:
            import copy
            pp = copy.deepcopy(ANT_PPO_PARAMS)
            pp["params"]["config"]["player"] = {
                "games_num": int(os.environ.get("LM_RL_GAMES", "100000")),
                "deterministic": True, "render": False}
            rl.play_rl_games(task, play_ckpt, params=pp)
        else:
            rl.train_rl_games(task, max_epochs=int(os.environ.get("LM_RL_EPOCHS", "500")), seed=0,
                              horizon_length=24, mini_epochs=5, params=ANT_PPO_PARAMS)
    except BaseException:
        import traceback; print("[ant-dbg] run raised:"); traceback.print_exc()
    finally:
        rl.destroy_world(task.sim, task.runner)
