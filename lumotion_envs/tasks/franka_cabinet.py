"""Franka Panda opens a cabinet drawer — IsaacGym FrankaCabinet port (Tier C #3).

The FIRST multi-articulation task: every env holds TWO fixed-base articulations in
one direct-GPU batch — the 9-DOF arm at the cell origin and a passive 4-DOF sektion
cabinet 1 m in front of it (yawed 180 deg so the top drawer slides out toward the
arm). Batch rows are mapped to (type, env) at capture time from the root poses
(cabinet roots sit at z=0.4, and each type's XY matches its env cell) — row order
is EnTT view order, never assume row == env.

Control is IsaacGym's delta-target scheme: targets += speed_scale*dt*action*7.5,
clamped to the real Franka joint limits. Reward = reach the drawer handle, align
the gripper, straddle the handle with the fingers, and pull the drawer open;
success (drawer > 0.39 m) terminates the episode.

Run via the CLI:
    python train.py --task FrankaCabinet
"""
import math
from pathlib import Path

from lumotion_envs._engine import ensure_engine
ensure_engine()
import lm.rl as rl
from lumotion_envs.config import FrankaCabinetConfig

from lumotion_envs.assets import ASSETS
CLIP_OBS = 5.0
MAX_EPISODE_LENGTH = 500
SUBSTEPS = 2
# Real Franka joint limits (rad; fingers m) in joint1..7 + finger order.
FRANKA_LOWER = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, 0.0, 0.0]
FRANKA_UPPER = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 0.04, 0.04]

FRANKA_CABINET_PPO_PARAMS = {"params": {
    "network": {"mlp": {"units": [256, 128, 64]}},
    "config": {
        "reward_shaper": {"scale_value": 0.01},
        "critic_coef": 4,
        "entropy_coef": 0.0,
        "bounds_loss_coef": 0.0001,
        "horizon_length": 16,
    },
}}


class FrankaCabinetTask(rl.VecTask):
    """Vectorized drawer opening (fixed-base Franka + passive cabinet per env)."""

    def __init__(self, cfg: FrankaCabinetConfig = None, *, num_envs=None, headless=None):
        self.cfg = cfg or FrankaCabinetConfig()
        if num_envs is not None:
            self.cfg.num_envs = num_envs
        if headless is not None:
            self.cfg.headless = headless
        c = self.cfg
        self.n_dof = int(c.num_dof)              # the ARM: 7 + 2 fingers
        self.world = rl.World(num_envs=int(c.num_envs), env_spacing=c.env_spacing)
        self.world.add_ground(z=0.0, friction=1.0)
        self.robot = self.world.add_robot(
            rl.Mjcf(str(ASSETS / c.robot), prep=True, config=str(ASSETS / c.rl_yaml)))
        # The cabinet: URDF drawer axis is its local +x; yawed pi it slides out
        # toward the arm. Same env partition -> collides with its arm only.
        self.cabinet = self.world.add_robot(
            rl.Urdf(str(ASSETS / c.cabinet), prep=True, config=str(ASSETS / c.cabinet_yaml)),
            name="cabinet", at=(c.cabinet_x, 0.0), rpy=(0.0, 0.0, math.pi), spawn_z=c.cabinet_z)
        sim, runner = self.world.build(
            headless=c.headless,
            config=rl.SimConfig(dt=1.0 / 60.0, substeps=SUBSTEPS, device="auto",
                                gpu_found_lost_pairs_capacity=2_000_000,
                                # Sustained gripper contact on the drawer handle across
                                # 4096 envs overflows the 81920-patch narrowphase default
                                # (silent GPU exhaustion -> exit 1). 3x = 245760 stays
                                # under the internal 2^18=262144 cap (4x crashes init).
                                gpu_max_rigid_patch_count=245_760),
            title=f"{c.name} (drawer)")
        sim.play()
        # actions: 7 arm + 2 fingers (IsaacGym drives both fingers). obs 23 =
        # dof_pos_scaled(9) + dof_vel(9) + to_target(3) + drawer pos(1) + vel(1)
        super().__init__(sim, runner, num_obs=23, num_actions=9,
                         name=c.name, clip_obs=CLIP_OBS,
                         max_episode_length=MAX_EPISODE_LENGTH, seed=int(c.seed))

    # -- task hooks ---------------------------------------------------------

    def _capture(self):
        torch = self._torch; dev = self.device; c = self.cfg
        self._dof = self.sim.acquire_dof_state_tensor()          # (2N, max_dofs, 2)
        self._root = self.sim.acquire_root_state_tensor()        # (2N, 13)
        self._rigid_state = self.sim.acquire_rigid_body_state_tensor()
        self.sim.refresh_dof_state_tensor(); self.sim.refresh_root_state_tensor()

        # Row -> (type, env): cabinets sit at z=cabinet_z, arms at z~0; within a
        # type, match the root XY to its env cell (arm at the cell, cabinet at
        # cell + (cabinet_x, 0)). Row order is EnTT view order — NEVER row == env.
        side = int(math.ceil(math.sqrt(self.num_envs)))
        cells = torch.stack([
            torch.arange(self.num_envs, device=dev) % side * c.env_spacing,
            torch.arange(self.num_envs, device=dev) // side * c.env_spacing], dim=1).float()
        rz = self._root[:, 2]
        cab_mask = rz > c.cabinet_z * 0.5
        cab_rows = cab_mask.nonzero(as_tuple=False).flatten()
        arm_rows = (~cab_mask).nonzero(as_tuple=False).flatten()
        assert cab_rows.numel() == self.num_envs and arm_rows.numel() == self.num_envs, \
            f"row split failed: {cab_rows.numel()} cabinets / {arm_rows.numel()} arms"

        def rows_by_env(rows, offset_x):
            d = (self._root[rows, None, 0:2]
                 - (cells[None, :, :] + torch.tensor([offset_x, 0.0], device=dev))).norm(dim=2)
            return rows[d.argmin(dim=0)]                     # env -> batch row

        self._arm_rows = rows_by_env(arm_rows, 0.0)
        self._cab_rows = rows_by_env(cab_rows, c.cabinet_x)

        # DOF columns per type (name map from each type's own root layout).
        self._arm_idx = self.robot.view.dof_indices(
            [f"joint{i}" for i in range(1, 8)] + ["finger_joint1", "finger_joint2"])
        self._drawer_col = self.cabinet.view.get_joint("drawer_top_joint").dof_index
        # Link rows (per-link tensor columns) per type.
        lm_ = self.robot.view.link_map
        self._hand_idx = lm_["hand"]
        fingertips = [i for n, i in lm_.items() if n.lower().endswith("finger")]
        assert len(fingertips) == 2, f"finger links not found in {sorted(lm_)}"
        self._finger_a, self._finger_b = fingertips
        self._handle_idx = self.cabinet.view.link_map["drawer_handle_top"]

        # Default arm pose (name map -> DOF order), limits, live PD targets.
        self._default_dof = self.robot.view.default_dof_positions.unsqueeze(0).repeat(
            self.num_envs, 1)                                # (N, 9) in arm DOF order
        low = torch.zeros_like(self._default_dof); up = torch.zeros_like(self._default_dof)
        low[:, self._arm_idx] = torch.tensor(FRANKA_LOWER, device=dev)
        up[:, self._arm_idx] = torch.tensor(FRANKA_UPPER, device=dev)
        self._dof_lower, self._dof_upper = low, up
        self._speed_scale = torch.ones(self.num_envs, self._dof.shape[1], device=dev)
        self._speed_scale[:, self._arm_idx[7:]] = 0.1        # fingers move slowly
        self._targets = torch.zeros(self.num_envs, self._dof.shape[1], device=dev)
        self._targets[:, :] = self._default_dof

        # Settle: hold the default pose a few steps so both articulations rest.
        full = torch.zeros_like(self._dof[:, :, 0])
        for _ in range(20):
            full[self._arm_rows] = self._targets
            self.sim.set_dof_position_target_tensor(full)
            self.sim.simulate(); self.sim.fetch_results()
            if self.runner is not None:
                self.runner.run()
        self.sim.refresh_dof_state_tensor(); self.sim.refresh_root_state_tensor()
        self.sim.refresh_rigid_body_state_tensor()
        self._full_targets = full
        self._last_action = torch.zeros(self.num_envs, 9, device=dev)

    # -- per-step frames -----------------------------------------------------

    def hand_pos(self):
        return self._rigid_state[self._arm_rows, self._hand_idx, 0:3]

    def grasp_pos(self):
        """TCP: midpoint of the two fingertip links."""
        fa = self._rigid_state[self._arm_rows, self._finger_a, 0:3]
        fb = self._rigid_state[self._arm_rows, self._finger_b, 0:3]
        return 0.5 * (fa + fb)

    def handle_pos(self):
        return self._rigid_state[self._cab_rows, self._handle_idx, 0:3]

    def drawer_state(self):
        """(pos, vel) of the top drawer joint, (N,) each."""
        d = self._dof[self._cab_rows]
        return d[:, self._drawer_col, 0], d[:, self._drawer_col, 1]

    def _pre_physics_step(self, actions):
        self._last_action = actions.clone()
        # IsaacGym delta-target control: the target walks at speed_scale*dt*scale.
        step = self._speed_scale * self.sim.dt * self.cfg.action_scale
        t = self._targets
        t[:, self._arm_idx] += step[:, self._arm_idx] * actions
        self._targets = t.clamp(self._dof_lower, self._dof_upper)
        self._full_targets[self._arm_rows] = self._targets
        self.sim.set_dof_position_target_tensor(self._full_targets)

    def _compute_observations(self):
        torch = self._torch; c = self.cfg
        self.sim.refresh_dof_state_tensor()
        self.sim.refresh_rigid_body_state_tensor()
        arm = self._dof[self._arm_rows]                       # (N, max_dofs, 2)
        pos, vel = arm[..., 0], arm[..., 1]
        pos_scaled = (2.0 * (pos - self._dof_lower)
                      / (self._dof_upper - self._dof_lower + 1e-8) - 1.0)
        dpos, dvel = self.drawer_state()
        o = self.obs_buf
        o[:, 0:9] = pos_scaled[:, self._arm_idx]
        o[:, 9:18] = vel[:, self._arm_idx] * c.dof_vel_scale
        o[:, 18:21] = self.handle_pos() - self.grasp_pos()
        o[:, 21] = dpos
        o[:, 22] = dvel
        torch.nan_to_num_(self.obs_buf, nan=0.0, posinf=CLIP_OBS, neginf=-CLIP_OBS)

    def _compute_reward(self):
        torch = self._torch; c = self.cfg
        tcp = self.grasp_pos()
        hand = self.hand_pos()
        handle = self.handle_pos()
        dpos, _ = self.drawer_state()
        fa_z = self._rigid_state[self._arm_rows, self._finger_a, 2]
        fb_z = self._rigid_state[self._arm_rows, self._finger_b, 2]
        fa_x = self._rigid_state[self._arm_rows, self._finger_a, 0]
        fb_x = self._rigid_state[self._arm_rows, self._finger_b, 0]

        d = (tcp - handle).norm(dim=1)
        dist_reward = (1.0 / (1.0 + d * d)) ** 2
        dist_reward = torch.where(d <= 0.02, dist_reward * 2.0, dist_reward)

        # Alignment (world-frame port of IsaacGym's axis dots): the approach axis
        # (hand -> TCP) should point at the cabinet (+x); the finger-separation
        # axis should be vertical (fingers straddle the horizontal handle bar).
        fwd = torch.nn.functional.normalize(tcp - hand, dim=1)
        dot1 = fwd[:, 0]
        fin_axis = torch.nn.functional.normalize(
            self._rigid_state[self._arm_rows, self._finger_a, 0:3]
            - self._rigid_state[self._arm_rows, self._finger_b, 0:3], dim=1)
        dot2 = fin_axis[:, 2]
        rot_reward = 0.5 * (torch.sign(dot1) * dot1 * dot1 + dot2 * dot2)

        hz = handle[:, 2]
        straddle = (torch.maximum(fa_z, fb_z) > hz) & (torch.minimum(fa_z, fb_z) < hz)
        around = torch.where(straddle, 0.5, 0.0)
        finger_dist = torch.where(
            straddle, (0.04 - (fa_z - hz).abs()) + (0.04 - (fb_z - hz).abs()),
            torch.zeros_like(hz))

        action_penalty = self._last_action.square().sum(dim=1)
        open_reward = dpos * around + dpos                    # drawer travel

        rew = (2.0 * dist_reward + 0.5 * rot_reward + 0.25 * around
               + 7.5 * open_reward + 5.0 * finger_dist - 0.01 * action_penalty)
        rew = torch.where(dpos > 0.01, rew + 0.5, rew)
        rew = torch.where(dpos > 0.2, rew + around, rew)
        rew = torch.where(dpos > 0.39, rew + 2.0 * around, rew)
        # Bad style: a finger reached PAST the handle into the cabinet face.
        hx = handle[:, 0]
        bad = (fa_x > hx + 0.04) | (fb_x > hx + 0.04)
        rew = torch.where(bad, torch.full_like(rew, -1.0), rew)

        self.rew_buf = torch.nan_to_num(rew, nan=0.0, posinf=0.0, neginf=0.0)
        # Success termination: the drawer is open. (Time limit is added by the base.)
        self.reset_buf = (dpos > c.open_target).float()

    def _reset_idx(self, env_ids):
        torch = self._torch
        n = env_ids.numel(); c = self.cfg
        arm_rows = self._arm_rows[env_ids]
        cab_rows = self._cab_rows[env_ids]
        noise = torch.zeros(n, self._dof.shape[1], device=self.device)
        noise[:, self._arm_idx] = (torch.rand(n, 9, device=self.device) - 0.5) \
            * 2.0 * c.reset_joint_noise
        pose = (self._default_dof[env_ids] + noise).clamp(
            self._dof_lower[env_ids], self._dof_upper[env_ids])
        self._dof[arm_rows, :, 0] = pose
        self._dof[arm_rows, :, 1] = 0.0
        self._dof[cab_rows] = 0.0                             # drawer/doors closed, still
        rows = torch.cat([arm_rows, cab_rows])
        self.sim.set_dof_state_tensor_indexed(self._dof, rows)
        self._targets[env_ids] = pose
        self._full_targets[arm_rows] = pose
        self.sim.set_dof_position_target_tensor_indexed(self._full_targets, rows)
        self._last_action[env_ids] = 0.0
        self.progress_buf[env_ids] = 0.0


def _frame_camera(task):
    """Close-up on env 0's workspace for small demos; grid overview otherwise."""
    v = task.runner
    if v is None or not hasattr(v, "set_camera"):
        return
    if task.num_envs <= 4:
        v.set_camera(pos=(0.6, -1.6, 1.0), target=(0.7, 0.0, 0.5))
        return
    side = int(math.ceil(math.sqrt(task.num_envs)))
    ext = (side - 1) * task.cfg.env_spacing
    cx = cy = ext / 2.0
    dist = ext * 0.7 + 4.0
    v.set_camera(pos=(cx - dist * 0.5, cy - dist * 0.7, dist * 0.6), target=(cx, cy, 0.4))
