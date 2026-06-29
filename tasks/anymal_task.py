"""ANYmal velocity-command locomotion on the USD world (the real anymal_c).

The robot is the real anymal_c (anymal.urdf -> USD), authored via the rl.World facade;
floating base + PD drives come from assets/anymal_c.rl.yaml. Task = velocity-command
walking (track a commanded base velocity vx/vy/yaw under PD position control), with the
IsaacLab velocity-flat reward set. Z-up.

Run via the CLI:
    python train.py --task Anymal
    python train.py --task Anymal --set terrain=noise --set terrain_amp=0.15
    python play.py  --task Anymal --checkpoint runs/Anymal_.../nn/Anymal.pth --cmd 1,0,0

obs(48) = base_lin_vel*2 + base_ang_vel*0.25 + projected_gravity + commands*[2,2,0.25]
          + dof_pos_rel + dof_vel*0.05 + prev_actions
action(12) -> target = 0.5*action + default_dof ; PD Kp=85 Kd=2
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # for lumengine_envs
import _bootstrap
_bootstrap.bootstrap()
import lm.rl as rl
from lumengine_envs.config import AnymalConfig

_ROBOT = _bootstrap.ASSETS / "anymal_converted" / "anymal.usda"
_CFG   = _bootstrap.ASSETS / "anymal_c.rl.yaml"   # floating base + PD drives (config-driven prep)

N_DOF       = 12
GROUND_Z    = -0.65
SUBSTEPS    = 2
ENV_SPACING = 4.0          # default env grid spacing (also the AnymalConfig default; kept
                           # as a module constant for tests that author worlds directly)

LIN_VEL_SCALE, ANG_VEL_SCALE = 2.0, 0.25
DOF_VEL_SCALE = 0.05
ACTION_SCALE  = 0.5
CLIP_OBS      = 5.0
TRACK_SIGMA   = 0.25
H_MIN_DROP, UPRIGHT_MIN = 0.30, 0.6
MAX_EPISODE_LENGTH = 1000
# Forward-first command distribution: the bare velocity-tracking reward leaves the policy
# stuck at vx=0 (crouch + step in place) under backward/large-yaw commands. Bias forward
# so forward walking is the dominant skill; widen later once it walks.
CMD_X, CMD_Y, CMD_YAW = (0.0, 1.5), (-0.5, 0.5), (-0.5, 0.5)

# IsaacGymEnvs AnymalPPO network + the entropy that escapes the "step in place" optimum.
ANYMAL_PPO_PARAMS = {"params": {
    "network": {"mlp": {"units": [256, 128, 64]}},
    "config": {
        "reward_shaper": {"scale_value": 1.0},
        "critic_coef": 2,
        "bounds_loss_coef": 0.001,
        "entropy_coef": 0.01,
    },
}}

# IsaacLab velocity-FLAT reward set (produces a clean gait, not tracking-by-sliding).
# lm.rl's RewardManager multiplies each by dt exactly like IsaacLab, so weights port directly.
REWARD_TERMS = [
    rl.RewardTerm("track_lin_vel_xy",   rl.rewards.track_lin_vel_xy_exp,   1.0,     {"std": 0.5}),
    rl.RewardTerm("track_ang_vel_z",    rl.rewards.track_ang_vel_z_exp,    0.5,     {"std": 0.5}),
    rl.RewardTerm("lin_vel_z",          rl.rewards.lin_vel_z_l2,          -2.0),
    rl.RewardTerm("ang_vel_xy",         rl.rewards.ang_vel_xy_l2,         -0.05),
    rl.RewardTerm("dof_torques",        rl.rewards.dof_torques_l2,        -2.5e-5),
    rl.RewardTerm("dof_acc",            rl.rewards.dof_acc_l2,            -2.5e-7),
    rl.RewardTerm("action_rate",        rl.rewards.action_rate_l2,        -0.01),
    rl.RewardTerm("feet_air_time",      rl.rewards.feet_air_time,          0.5,     {"threshold": 0.5}),
    rl.RewardTerm("undesired_contacts", rl.rewards.undesired_contacts,    -1.0,     {"threshold": 1.0}),
    rl.RewardTerm("flat_orientation",   rl.rewards.flat_orientation_l2,   -5.0),
    # Anti-crouch: pull the base back to its captured standing height.
    rl.RewardTerm("base_height",        rl.rewards.base_height_l2,       -50.0),
]

# Declarative observation vector (48) — ObservationManager scales/concats these in order.
# velocity_commands already applies the per-component command scale, so it carries no scale.
OBS_TERMS = [
    rl.ObsTerm("lin_vel",  rl.observations.base_lin_vel,      scale=LIN_VEL_SCALE),
    rl.ObsTerm("ang_vel",  rl.observations.base_ang_vel,      scale=ANG_VEL_SCALE),
    rl.ObsTerm("gravity",  rl.observations.projected_gravity),
    rl.ObsTerm("commands", rl.observations.velocity_commands),
    rl.ObsTerm("dof_pos",  rl.observations.joint_pos_rel),
    rl.ObsTerm("dof_vel",  rl.observations.joint_vel,         scale=DOF_VEL_SCALE),
    rl.ObsTerm("actions",  rl.observations.last_action),
]

# Reset events (declarative): teleport to home + randomize the joint stance (IsaacGymEnvs reset).
EVENT_TERMS = [
    rl.EventTerm("reset_root",   rl.events.reset_root_to_home, mode="reset"),
    rl.EventTerm("reset_joints", rl.events.reset_joints_scaled, mode="reset",
                 params={"lo": 0.5, "hi": 1.5, "vel_noise": 0.1}),
]


def _quat_rotate_inv(q, v):
    import torch
    qv, qw = -q[:, 0:3], q[:, 3:4]
    t = 2.0 * torch.cross(qv, v, dim=1)
    return v + qw * t + torch.cross(qv, t, dim=1)


def _parse_cmd(s):
    if not s:
        return None
    return [float(x) for x in str(s).split(",")]


class AnymalTask(rl.VecTask):
    """ANYmal velocity-command locomotion on rl.VecTask."""

    def __init__(self, cfg: AnymalConfig = None, *, num_envs=None, headless=None):
        # Primary API: an AnymalConfig. The keyword args are a back-compat path (tests +
        # direct construction).
        self.cfg = cfg or AnymalConfig()
        if num_envs is not None:
            self.cfg.num_envs = num_envs
        if headless is not None:
            self.cfg.headless = headless
        headless = self.cfg.headless
        # instanceable shares ONE composed robot prototype across all envs (memory win). Only
        # when headless: the windowed render delegate doesn't draw USD instance proxies.
        inst = self.cfg.instance
        instanceable = bool(headless) if inst == "auto" else (inst == "on")
        self.world = rl.World(num_envs=int(self.cfg.num_envs), env_spacing=self.cfg.env_spacing)
        self.curr = None
        if self.cfg.terrain == "curriculum":
            self.world.add_terrain_curriculum(
                types=[
                    lambda d: rl.terrain.Slope(slope=0.05 + 0.45 * d, axis="x"),
                    lambda d: rl.terrain.Stairs(step_height=0.02 + 0.10 * d, step_width=0.4, platform=1.5),
                    lambda d: rl.terrain.Noise(amplitude=0.04 + 0.30 * d, base_cells=4, seed=0),
                ],
                num_levels=self.cfg.curriculum_levels, max_init_level=self.cfg.curriculum_init,
                size_m=self.cfg.curriculum_size, friction=1.0, base_z=GROUND_Z)
        elif self.cfg.terrain == "variants":
            self.world.add_terrain_variants([
                ("flat",   rl.terrain.Flat()),
                ("noise",  rl.terrain.Noise(amplitude=0.18, base_cells=4, seed=0)),
                ("slope",  rl.terrain.Slope(slope=0.15, axis="x")),
                ("stairs", rl.terrain.Stairs(step_width=0.4, step_height=0.06)),
            ], size_m=self.cfg.env_spacing, friction=1.0, base_z=GROUND_Z)
            self.world.assign_terrain(strategy=self.cfg.terrain_strategy, seed=0)
        elif self.cfg.terrain == "noise":
            self.world.add_terrain(rl.terrain.Noise(amplitude=self.cfg.terrain_amp,
                                                    base_cells=self.cfg.terrain_cells, seed=0),
                                   friction=1.0, base_z=GROUND_Z)
        else:  # "flat"
            self.world.add_ground(z=GROUND_Z, friction=1.0)
        if self.cfg.scatter > 0:
            self.world.scatter(rl.Cylinder(radius=0.2, height=0.8, color=(0.7, 0.4, 0.2)),
                               count=self.cfg.scatter, area=(2.5, 2.5), z=GROUND_Z + 0.4,
                               per_env=True, seed=0, clearance=1.2)
        self.robot = self.world.add_robot(rl.Usd(str(_ROBOT), prep=True, config=str(_CFG)), spawn_z=0.0)
        # Scale the GPU contact buffers with env count; the terrain curriculum needs MORE
        # (triangle-mesh tiles + clustering). NOTE: scale UP with envs, never down.
        _gpu_mult = max(1.0, int(self.cfg.num_envs) / 512.0)
        if self.cfg.terrain == "curriculum":
            _gpu_mult = max(_gpu_mult, int(self.cfg.num_envs) / 256.0, 2.0)
        sim, runner = self.world.build(
            headless=headless, instanceable=instanceable,
            # 50 Hz control (dt=0.02) to match IsaacLab's dt-scaled penalty weights.
            config=rl.SimConfig(dt=1.0 / 50.0, substeps=SUBSTEPS, device="auto",
                                gpu_contact_buffer_multiplier=_gpu_mult),
            title="ANYmal (rl_games, vel-cmd)")
        sim.play()
        super().__init__(sim, runner, num_obs=48, num_actions=N_DOF, name="Anymal",
                         clip_obs=CLIP_OBS, max_episode_length=MAX_EPISODE_LENGTH, seed=int(self.cfg.seed))
        self._nstep = 0
        self.curr = self.world.make_terrain_curriculum(self.device)

    # -- task hooks ---------------------------------------------------------

    def _capture(self):
        torch = self._torch; dev = self.device
        self._dof = self.sim.acquire_dof_state_tensor()
        self._root = self.sim.acquire_root_state_tensor()
        self._contact = self.sim.acquire_link_net_contact_force_tensor()   # (envs, links, 3)
        self.sim.refresh_dof_state_tensor(); self.sim.refresh_root_state_tensor()
        self._default_dof = self.robot.default_dof_positions.unsqueeze(0).repeat(self.num_envs, 1)
        self._world_down = torch.tensor([0.0, 0.0, -1.0], device=dev).repeat(self.num_envs, 1)
        self._cmd_scale = torch.tensor([LIN_VEL_SCALE, LIN_VEL_SCALE, ANG_VEL_SCALE], device=dev)
        self._prev_action = torch.zeros(self.num_envs, N_DOF, device=dev)
        self._last_action = torch.zeros(self.num_envs, N_DOF, device=dev)
        self._cmd = _parse_cmd(self.cfg.cmd)        # fixed command, or None for per-env random
        # Settle onto the feet before capturing the home/reset pose (the batch is "ready"
        # before the robot settles from its spawn height; capturing home now would reset to
        # a too-high pose that drops into a crouch). Hold the default PD stance briefly.
        for _ in range(80):
            self.sim.set_dof_position_target_tensor(self._default_dof)
            self.sim.simulate(); self.sim.fetch_results()
            if self.runner is not None:
                self.runner.run()
        self.sim.refresh_dof_state_tensor(); self.sim.refresh_root_state_tensor()
        self._h_min = self._root[:, 2] - H_MIN_DROP
        home = torch.zeros(self.num_envs, 13, device=dev)
        home[:, 0:3] = self._root[:, 0:3]
        home[:, 6] = 1.0
        self._home = home
        self.base_height_target = self._root[:, 2].clone()   # target for the base_height reward

        # control step (for dof_acc + IsaacLab dt-scaled reward weights)
        self.dt = float(self.sim._cfg.dt) * int(self.control_freq_inv)
        self._dof_force = self.sim.acquire_applied_dof_force_tensor()
        self._prev_dof_vel = torch.zeros(self.num_envs, N_DOF, device=dev)
        lm = self.robot.view.link_map
        # Feet = the FOOT links (correct after the engine fix that stops giving collider-less
        # inertia-authored links a spurious fallback sphere). undesired = THIGH only.
        self.feet_indices = torch.tensor(
            [i for n, i in lm.items() if n.upper().endswith("FOOT")], device=dev, dtype=torch.long)
        self.undesired_contact_indices = torch.tensor(
            [i for n, i in lm.items() if n.upper().endswith("THIGH")], device=dev, dtype=torch.long)
        self.knee_indices = torch.tensor(
            [i for n, i in lm.items() if n.upper().endswith("THIGH")], device=dev, dtype=torch.long)
        nf = int(self.feet_indices.numel())
        self._air_time = torch.zeros(self.num_envs, nf, device=dev)
        self._prev_in_contact = torch.zeros(self.num_envs, nf, dtype=torch.bool, device=dev)

        # Managers (declarative task definition, IsaacLab-style):
        self.rewards = rl.RewardManager(self, REWARD_TERMS)
        self.observations = rl.ObservationManager(self, OBS_TERMS)
        self.terminations = rl.TerminationManager(self, [
            rl.TermTerm("base_contact", rl.terminations.base_contact,
                        {"threshold": self.cfg.base_contact_fail_n, "link": 0}),
            rl.TermTerm("knee_contact", rl.terminations.contact_on,
                        {"indices": self.knee_indices, "threshold": self.cfg.base_contact_fail_n}),
        ])
        self.commands_mgr = rl.CommandManager(self, {"base_velocity": rl.UniformVelocityCommand(
            self, vx=CMD_X, vy=CMD_Y, yaw=CMD_YAW, fixed=self._cmd)})
        self.events = rl.EventManager(self, EVENT_TERMS)

    def _pre_physics_step(self, actions):
        self._prev_action = self._last_action    # action_rate reads both
        self._last_action = actions
        targets = self._default_dof + ACTION_SCALE * actions
        self.sim.set_dof_position_target_tensor(targets)

    def _update_state(self):
        """Refresh the standardized per-step robot state the reward terms read off `self`."""
        torch = self._torch
        self.sim.refresh_root_state_tensor()
        self.sim.refresh_applied_dof_force_tensor()
        self.sim.refresh_link_net_contact_force_tensor()
        root, dof = self._root, self._dof
        quat = root[:, 3:7]
        self.base_height = root[:, 2]                    # for the base_height reward
        self.lin_vel_b = _quat_rotate_inv(quat, root[:, 7:10])
        self.ang_vel_b = _quat_rotate_inv(quat, root[:, 10:13])
        self.proj_gravity = _quat_rotate_inv(quat, self._world_down)
        self._up_proj = self.up_proj = -self.proj_gravity[:, 2]
        self.dof_pos_rel = dof[:, :N_DOF, 0] - self._default_dof
        self.dof_vel = dof[:, :N_DOF, 1]
        self.dof_acc = (self.dof_vel - self._prev_dof_vel) / self.dt
        self._prev_dof_vel = self.dof_vel.clone()
        self.dof_torque = self._dof_force[:, :N_DOF]
        self.contact_forces = self._contact
        self.actions, self.prev_actions = self._last_action, self._prev_action
        self.commands = self.commands_mgr.get("base_velocity")
        ff = self.contact_forces[:, self.feet_indices, :].norm(dim=2)     # (envs, n_feet)
        in_contact = ff > 1.0
        self.feet_first_contact = in_contact & (~self._prev_in_contact)
        self.feet_air_time = self._air_time.clone()
        self._air_time = torch.where(in_contact, torch.zeros_like(self._air_time),
                                     self._air_time + self.dt)
        self._prev_in_contact = in_contact

    def _compute_observations(self):
        self._update_state()
        self.obs_buf[:] = self.observations.compute()

    def _compute_reward(self):
        torch = self._torch
        # IsaacLab convention: sum the weighted terms WITHOUT clamping at 0 — the penalty
        # terms must keep their negative gradient.
        self.rew_buf = self.rewards.compute()
        self.reset_buf = self.terminations.compute()
        height = self._root[:, 2]

        self._nstep += 1
        # Instanced fleet (windowed): once instancing + articulations are up, bind the shared
        # render copies to the per-env articulation poses so the geometry animates per-env.
        if self._nstep == 10 and not getattr(self, "_driven", False):
            self._driven = True
            try:
                import lm.physx as _physx
                _physx.drive_instanced_articulations(self.sim.scene)
            except Exception:
                pass
        if self._nstep % 500 == 0:
            cmd = self.commands.mean(0)
            curr_s = f" | curr_level(mean)={self.curr.mean_level():.2f}/{self.curr.max_level - 1}" \
                     if self.curr is not None else ""
            print(f"[anymal-dbg] step {self._nstep} | ep_len(mean)={float(self.progress_buf.mean()):.1f} "
                  f"| rew={float(self.rew_buf.mean()):.3f} | vx={float(self.lin_vel_b[:,0].mean()):+.2f} "
                  f"(cmd_x~{float(cmd[0]):+.2f}) | upright={float(self._up_proj.mean()):.2f} "
                  f"| height={float(height.mean()):.2f} | air={float(self._air_time.mean()):.2f}"
                  f" | fails={int(self.reset_buf.sum())}{curr_s}", flush=True)
            ep = self.rewards.episode_means()
            print("[anymal-dbg]   rew terms: " + " ".join(f"{k}={v:+.3f}" for k, v in ep.items()), flush=True)

    def _reset_idx(self, ids):
        if self.curr is not None:
            dist = (self._root[ids, 0:2] - self._home[ids, 0:2]).norm(dim=1)
            move_up = dist > self.curr.size_m * 0.5
            move_down = (dist < 0.5) & (~move_up)
            self.curr.update(ids, move_up, move_down)
            self._home[ids, 0:3] = self.curr.env_origins[ids]
            self._h_min[ids] = self.curr.env_origins[ids, 2] - H_MIN_DROP
        self.events.reset(ids)                 # teleport home + randomize the joint stance
        self.commands_mgr.resample(ids)        # new per-episode velocity command
        self._prev_action[ids] = 0.0
        self._last_action[ids] = 0.0
        self._prev_dof_vel[ids] = 0.0
        self._air_time[ids] = 0.0
        self._prev_in_contact[ids] = False
        self.rewards.reset(ids)
        self.progress_buf[ids] = 0.0


def _frame_camera(task):
    v = task.runner
    if v is None or not hasattr(v, "set_camera"):
        return
    side = int(math.ceil(math.sqrt(task.num_envs)))
    ext = (side - 1) * task.cfg.env_spacing
    c = ext / 2.0
    dist = ext * 0.9 + 8.0
    v.set_camera(pos=(c - dist * 0.5, c - dist * 0.7, dist * 0.6), target=(c, c, 0.3))
