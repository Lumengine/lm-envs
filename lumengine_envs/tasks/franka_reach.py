"""Franka Panda end-effector reach — the first manipulation task (Tier C).

A 7-DOF fixed-base arm (panda_nohand.xml, no gripper) tracks a random 3D target with
its end-effector under PD position control. IsaacLab reach-style MDP: position-error
reward (coarse L2 + fine tanh kernel), action-rate + joint-velocity penalties, no
failure termination (timeout only), target resampled each episode.

Run via the CLI:
    python train.py --task FrankaReach
    python play.py  --task FrankaReach --checkpoint runs/FrankaReach_.../nn/FrankaReach.pth
"""
import math
from pathlib import Path

from lumengine_envs._engine import ensure_engine
ensure_engine()
import lm.rl as rl
from lumengine_envs.config import FrankaReachConfig

from lumengine_envs.assets import ASSETS
CLIP_OBS = 5.0
MAX_EPISODE_LENGTH = 250            # 5 s at 50 Hz — plenty for one reach
SUBSTEPS = 2
DOF_VEL_SCALE = 0.1

FRANKA_PPO_PARAMS = {"params": {
    "network": {"mlp": {"units": [256, 128, 64]}},
    "config": {
        "reward_shaper": {"scale_value": 1.0},
        "critic_coef": 2,
        "bounds_loss_coef": 0.001,
        "entropy_coef": 0.005,
        "horizon_length": 16,
    },
}}


class FrankaReachTask(rl.VecTask):
    """Vectorized end-effector reach on the USD world (fixed-base Franka)."""

    def __init__(self, cfg: FrankaReachConfig = None, *, num_envs=None, headless=None):
        self.cfg = cfg or FrankaReachConfig()
        if num_envs is not None:
            self.cfg.num_envs = num_envs
        if headless is not None:
            self.cfg.headless = headless
        c = self.cfg
        self.n_dof = int(c.num_dof)
        self.world = rl.World(num_envs=int(c.num_envs), env_spacing=c.env_spacing)
        self.world.add_ground(z=0.0, friction=1.0)
        self.robot = self.world.add_robot(
            rl.Mjcf(str(ASSETS / c.robot), prep=True, config=str(ASSETS / c.rl_yaml)))
        sim, runner = self.world.build(
            headless=c.headless,
            config=rl.SimConfig(dt=1.0 / 50.0, substeps=SUBSTEPS, device="auto"),
            title=f"{c.name} (reach)")
        sim.play()
        # obs: dof_pos_rel(7) + dof_vel(7) + target_rel(3) + ee_rel(3) + last_action(7)
        super().__init__(sim, runner, num_obs=3 * self.n_dof + 6, num_actions=self.n_dof,
                         name=c.name, clip_obs=CLIP_OBS,
                         max_episode_length=MAX_EPISODE_LENGTH, seed=int(c.seed))

    # -- task hooks ---------------------------------------------------------

    def _capture(self):
        torch = self._torch; dev = self.device; nd = self.n_dof; c = self.cfg
        self._dof = self.sim.acquire_dof_state_tensor()
        self._root = self.sim.acquire_root_state_tensor()
        self._rigid_state = self.sim.acquire_rigid_body_state_tensor()
        self.sim.refresh_dof_state_tensor(); self.sim.refresh_root_state_tensor()
        self._default_dof = self.robot.default_dof_positions.unsqueeze(0).repeat(self.num_envs, 1)
        # Short settle under the PD-held default pose (a bolted arm cannot tip; this just
        # lets the drives pull the joints onto the stance before the home capture).
        for _ in range(20):
            self.sim.set_dof_position_target_tensor(self._default_dof)
            self.sim.simulate(); self.sim.fetch_results()
            if self.runner is not None:
                self.runner.run()
        self.sim.refresh_dof_state_tensor(); self.sim.refresh_root_state_tensor()
        self.sim.refresh_rigid_body_state_tensor()
        self._base = self._root[:, 0:3].clone()          # fixed-base world origin per env
        # End-effector link row: prefer the named EE body; the converter may merge a
        # massless leaf body, so fall back to link7, then to the last link.
        lm = self.robot.view.link_map
        ee = c.ee_link.lower()
        idx = next((i for n, i in lm.items() if n.lower().endswith(ee)), None)
        if idx is None:
            idx = next((i for n, i in lm.items() if n.lower().endswith("link7")),
                       max(lm.values()))
        self.ee_index = int(idx)
        self._target = torch.zeros(self.num_envs, 3, device=dev)   # world coords
        self._last_action = torch.zeros(self.num_envs, nd, device=dev)
        self._prev_action = torch.zeros(self.num_envs, nd, device=dev)
        self._sample_targets(torch.arange(self.num_envs, device=dev))

    def _sample_targets(self, env_ids):
        torch = self._torch; c = self.cfg
        n = env_ids.numel()
        r = torch.rand(n, 3, device=self.device)
        lo = torch.tensor([c.target_x_min, c.target_y_min, c.target_z_min], device=self.device)
        hi = torch.tensor([c.target_x_max, c.target_y_max, c.target_z_max], device=self.device)
        self._target[env_ids] = self._base[env_ids] + lo + r * (hi - lo)

    def _pre_physics_step(self, actions):
        self._prev_action = self._last_action.clone()
        self._last_action = actions.clone()
        targets = self._default_dof + self.cfg.action_scale * actions
        self.sim.set_dof_position_target_tensor(targets)

    def _compute_observations(self):
        torch = self._torch; nd = self.n_dof
        self.sim.refresh_dof_state_tensor()
        self.sim.refresh_rigid_body_state_tensor()
        ee = self._rigid_state[:, self.ee_index, 0:3]
        o = self.obs_buf
        o[:, 0:nd] = self._dof[:, :nd, 0] - self._default_dof
        o[:, nd:2 * nd] = self._dof[:, :nd, 1] * DOF_VEL_SCALE
        o[:, 2 * nd:2 * nd + 3] = self._target - self._base
        o[:, 2 * nd + 3:2 * nd + 6] = ee - self._base
        o[:, 2 * nd + 6:] = self._last_action

    def _compute_reward(self):
        torch = self._torch
        ee = self._rigid_state[:, self.ee_index, 0:3]
        dist = (ee - self._target).norm(dim=1)
        # IsaacLab reach shape: coarse L2 pull + fine bounded kernel near the target.
        track = -0.2 * dist + 0.5 * (1.0 - torch.tanh(dist / 0.1))
        action_rate = (self._last_action - self._prev_action).square().sum(dim=1)
        joint_vel = self._dof[:, :self.n_dof, 1].square().sum(dim=1)
        self.rew_buf = track - 0.01 * action_rate - 0.001 * joint_vel
        # No failure termination — a bolted arm cannot fall; episodes end on timeout.

    def _reset_idx(self, env_ids):
        torch = self._torch
        n = env_ids.numel(); nd = self.n_dof; c = self.cfg
        noise = (torch.rand(n, nd, device=self.device) - 0.5) * 2.0 * c.reset_joint_noise
        self._dof[env_ids, :nd, 0] = self._default_dof[env_ids] + noise
        self._dof[env_ids, :nd, 1] = 0.0
        self.sim.set_dof_state_tensor_indexed(self._dof, env_ids)
        self.sim.set_dof_position_target_tensor_indexed(self._default_dof + 0.0, env_ids)
        self._sample_targets(env_ids)
        self._last_action[env_ids] = 0.0
        self._prev_action[env_ids] = 0.0
        self.progress_buf[env_ids] = 0.0


def _frame_camera(task):
    """Point the windowed viewer at the env grid (Z-up; envs spread on X-Y)."""
    v = task.runner
    if v is None or not hasattr(v, "set_camera"):
        return
    side = int(math.ceil(math.sqrt(task.num_envs)))
    ext = (side - 1) * task.cfg.env_spacing
    cx = cy = ext / 2.0
    dist = ext * 0.7 + 4.0
    v.set_camera(pos=(cx - dist * 0.5, cy - dist * 0.7, dist * 0.6), target=(cx, cy, 0.4))
