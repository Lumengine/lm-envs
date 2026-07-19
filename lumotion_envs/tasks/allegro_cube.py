"""Wonik Allegro Hand in-hand cube reorientation — IsaacGym AllegroHand port (Tier C dexterous).

A fixed-base 16-DOF hand is mounted palm-UP (rpy pitch -pi/2) and cradles a per-env dynamic
cube. Each episode picks a random GOAL orientation; the policy must rotate the cube in-hand to
match it. IsaacGym-style reward: rot_reward = 1/(|rot_dist| + eps) every step (so holding AND
aligning both pay), a big bonus when the goal is reached (then the goal is resampled in place),
minus an action penalty; dropping the cube costs a penalty and resets the episode.

    python train.py --task AllegroCube
    python train.py --task AllegroCube --num-envs 16 --view      # watch it learn
"""
import math
from pathlib import Path

from lumotion_envs._engine import ensure_engine
ensure_engine()
import lumotion as rl
from lumotion_envs.config import AllegroCubeConfig

from lumotion_envs.assets import ASSETS
CLIP_OBS = 5.0
SUBSTEPS = 2
MAX_EPISODE_LENGTH = 400            # ~6.7 s at 60 Hz

ALLEGRO_PPO_PARAMS = {"params": {
    "network": {"mlp": {"units": [512, 256, 128]}},
    "config": {
        "reward_shaper": {"scale_value": 0.01},
        "critic_coef": 4,
        "bounds_loss_coef": 0.0001,
        "entropy_coef": 0.003,          # exploration — reorientation is a hard skill to find
        "horizon_length": 16,
        "score_to_win": 100000000,      # never auto-stop ("Network won" fired at ep100 before)
        # STABILITY (IsaacGym AllegroHand): an ADAPTIVE learning rate that targets a fixed
        # policy KL is the key fix for the spike-then-collapse instability — it shrinks the
        # LR whenever the policy tries to change too fast. Plus value/obs normalization and
        # gradient clipping to keep the huge-variance reward (chained successes) trainable.
        "learning_rate": 5e-4,
        "lr_schedule": "adaptive",
        "kl_threshold": 0.016,
        "normalize_input": True,
        "normalize_value": True,
        "normalize_advantage": True,
        "truncate_grads": True,
        "grad_norm": 1.0,
        "clip_value": True,
        "gamma": 0.99,
        "tau": 0.95,
        "e_clip": 0.2,
    },
}}


# ── quaternion helpers (xyzw, matching the free-body tensor layout) ────────────
def _quat_mul(a, b):
    ax, ay, az, aw = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx, by, bz, bw = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    import torch
    return torch.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz], dim=1)


def _quat_conj(q):
    import torch
    return torch.cat([-q[:, 0:3], q[:, 3:4]], dim=1)


def _rot_dist(qa, qb):
    """Geodesic angle (rad) between two unit quaternions."""
    import torch
    d = _quat_mul(qa, _quat_conj(qb))
    return 2.0 * torch.asin(torch.clamp(d[:, 0:3].norm(dim=1), max=1.0))


def _rand_quat(n, device):
    """Uniform random unit quaternion (xyzw), Shoemake's method."""
    import torch
    u1, u2, u3 = torch.rand(n, device=device), torch.rand(n, device=device), torch.rand(n, device=device)
    return torch.stack([
        torch.sqrt(1 - u1) * torch.sin(2 * math.pi * u2),
        torch.sqrt(1 - u1) * torch.cos(2 * math.pi * u2),
        torch.sqrt(u1) * torch.sin(2 * math.pi * u3),
        torch.sqrt(u1) * torch.cos(2 * math.pi * u3)], dim=1)


# ── reward terms (read per-step state set in _compute_reward) — IsaacGym AllegroHand ──
def _r_dist(task):
    return task._goal_dist                          # position drift (scale < 0)


def _r_rot(task):
    return 1.0 / (task._rot_dist + task.cfg.rot_eps)


def _r_reach_bonus(task):
    return task._success.float()                    # +bonus on reaching the goal orientation


def _r_action(task):
    return (task._last_action ** 2).sum(dim=1)


class AllegroCubeTask(rl.VecTask):
    """Vectorized in-hand cube reorientation (fixed-base Allegro hand + per-env cube)."""

    def __init__(self, cfg: AllegroCubeConfig = None, *, num_envs=None, headless=None):
        self.cfg = cfg or AllegroCubeConfig()
        if num_envs is not None:
            self.cfg.num_envs = num_envs
        if headless is not None:
            self.cfg.headless = headless
        c = self.cfg
        self.world = rl.World(num_envs=int(c.num_envs), env_spacing=c.env_spacing)
        self.world.add_ground(z=0.0, friction=1.0)
        self.robot = self.world.add_robot(
            rl.Mjcf(str(ASSETS / c.robot), prep=True, config=str(ASSETS / c.rl_yaml)),
            spawn_z=c.hand_z, rpy=(0.0, c.hand_pitch, 0.0))
        s = float(c.cube_size)
        # Cube spawns just above the palm, in the cup of the up-pointing fingers.
        self.world.add_static(rl.Box(size=(s, s, s), dynamic=True, color=(0.85, 0.25, 0.15),
                                     solver_position_iterations=16, solver_velocity_iterations=1,
                                     max_depenetration_velocity=5.0, max_linear_velocity=10.0),
                              at=(0.0, 0.0, c.hand_z + c.cube_z_offset), per_env=True)
        sim, runner = self.world.build(
            headless=c.headless,
            config=rl.SimConfig(dt=1.0 / 60.0, substeps=SUBSTEPS, device="auto",
                                gpu_found_lost_pairs_capacity=4_000_000,
                                gpu_max_rigid_patch_count=245_760),
            title=f"{c.name} (in-hand)")
        sim.play()
        # obs: dof_pos(16) + dof_vel(16) + cube_pos_rel(3) + cube_quat(4)
        #      + goal_quat(4) + quat_diff(4) + last_action(16) = 63
        super().__init__(sim, runner, num_obs=63, num_actions=16,
                         name=c.name, clip_obs=CLIP_OBS,
                         max_episode_length=MAX_EPISODE_LENGTH, seed=int(c.seed))
        self.rewards = rl.RewardManager(self, [
            rl.RewardTerm("dist",        _r_dist,        c.dist_reward_scale),   # < 0
            rl.RewardTerm("rot",         _r_rot,         c.rot_reward_scale),
            rl.RewardTerm("reach_bonus", _r_reach_bonus, c.reach_goal_bonus),
            rl.RewardTerm("action",      _r_action,      c.action_penalty_scale),  # < 0
        ], dt_scale=False)

    # -- task hooks ---------------------------------------------------------
    def _capture(self):
        torch = self._torch; dev = self.device; c = self.cfg
        self._dof = self.sim.acquire_dof_state_tensor()
        self._root = self.sim.acquire_root_state_tensor()
        self._rigid = self.sim.acquire_rigid_body_state_tensor()
        self._cube = self.sim.acquire_free_body_state_tensor()
        self.sim.refresh_dof_state_tensor(); self.sim.refresh_root_state_tensor()
        self._default_dof = self.robot.default_dof_positions.unsqueeze(0).repeat(self.num_envs, 1)
        # Settle the hand into its cupped default pose so the cube seats in the fingers.
        for _ in range(30):
            self.sim.set_dof_position_target_tensor(self._default_dof)
            self.sim.simulate(); self.sim.fetch_results()
            if self.runner is not None:
                self.runner.run()
        self.sim.refresh_dof_state_tensor(); self.sim.refresh_root_state_tensor()
        self.sim.refresh_rigid_body_state_tensor(); self.sim.refresh_free_body_state_tensor()
        self._base = self._root[:, 0:3].clone()                 # palm world pos (fixed)
        # Cube home = the CUP CENTER (mean of the 4 fingertip links) — the palm origin sits
        # at the wrist, so spawning the cube there drops it off the back of the hand.
        lm_ = self.robot.view.link_map
        self._tip_idx = torch.tensor([lm_[n] for n in ["ff_tip", "mf_tip", "rf_tip", "th_tip"]],
                                     device=dev, dtype=torch.long)
        self._cube_home = self._rigid[:, self._tip_idx, 0:3].mean(dim=1).clone()
        # Free-body row -> env: match each cube to the nearest cup XY (rows follow EnTT order).
        d = (self._cube[:, None, 0:2] - self._cube_home[None, :, 0:2]).norm(dim=2)
        self._cube_env_row = d.argmin(dim=0)
        self._goal_quat = _rand_quat(self.num_envs, dev)
        self._last_action = torch.zeros(self.num_envs, 16, device=dev)

    def cube_state(self):
        return self._cube[self._cube_env_row]                   # (num_envs, 13)

    def _pre_physics_step(self, actions):
        self._last_action = actions.clone()
        targets = self._default_dof + self.cfg.action_scale * actions
        self.sim.set_dof_position_target_tensor(targets)

    def _compute_observations(self):
        self.sim.refresh_dof_state_tensor()
        self.sim.refresh_free_body_state_tensor()
        cube = self.cube_state()
        cube_pos, cube_quat = cube[:, 0:3], cube[:, 3:7]
        o = self.obs_buf
        o[:, 0:16] = self._dof[:, :16, 0] - self._default_dof
        o[:, 16:32] = self._dof[:, :16, 1] * self.cfg.dof_vel_scale
        o[:, 32:35] = cube_pos - self._base
        o[:, 35:39] = cube_quat
        o[:, 39:43] = self._goal_quat
        o[:, 43:47] = _quat_mul(cube_quat, _quat_conj(self._goal_quat))
        o[:, 47:63] = self._last_action

    def _compute_reward(self):
        torch = self._torch; c = self.cfg
        cube = self.cube_state()
        cube_pos, cube_quat = cube[:, 0:3], cube[:, 3:7]
        self._rot_dist = _rot_dist(cube_quat, self._goal_quat)
        # goal_dist = how far the cube drifted from its home position (the -10 dist term
        # keeps it in the hand; crossing fall_dist counts as dropped). IsaacGym AllegroHand.
        self._goal_dist = (cube_pos - self._cube_home).norm(dim=1)
        self._success = self._rot_dist < c.success_tolerance
        self.rew_buf = self.rewards.compute()
        # Reaching the goal orientation -> resample the goal AND restart the episode timer
        # (progress_buf=0) so the hand chains reorientations in one episode without a reset —
        # this is what makes the AllegroHand task a continuous reorientation benchmark.
        if self._success.any():
            ids = self._success.nonzero(as_tuple=False).flatten()
            self._goal_quat[ids] = _rand_quat(ids.numel(), self.device)
            self.progress_buf[ids] = 0.0
        # Drop -> reset (fall_penalty is 0 in AllegroHand). Non-finite cube counts as dropped.
        finite = torch.isfinite(cube).all(dim=1)
        dropped = (self._goal_dist > c.fall_dist) | ~finite
        self.rew_buf = self.rew_buf - dropped.float() * c.fall_penalty
        self.reset_buf = dropped.float()
        self.rew_buf = torch.nan_to_num(self.rew_buf, nan=0.0, posinf=0.0, neginf=0.0)
        torch.nan_to_num_(self.obs_buf, nan=0.0, posinf=CLIP_OBS, neginf=-CLIP_OBS)

    def _reset_idx(self, env_ids):
        torch = self._torch; n = env_ids.numel(); c = self.cfg
        noise = (torch.rand(n, 16, device=self.device) - 0.5) * 2.0 * c.reset_joint_noise
        self._dof[env_ids, :16, 0] = self._default_dof[env_ids] + noise
        self._dof[env_ids, :16, 1] = 0.0
        self.sim.set_dof_state_tensor_indexed(self._dof, env_ids)
        self.sim.set_dof_position_target_tensor_indexed(self._default_dof + 0.0, env_ids)
        # Cube back above the palm at a small jitter (indexed free-body write).
        self.sim.refresh_free_body_state_tensor()
        rows = self._cube_env_row[env_ids]
        jit = (torch.rand(n, 2, device=self.device) - 0.5) * 2.0 * c.cube_xy_noise
        self._cube[rows, 0] = self._cube_home[env_ids, 0] + jit[:, 0]
        self._cube[rows, 1] = self._cube_home[env_ids, 1] + jit[:, 1]
        self._cube[rows, 2] = self._cube_home[env_ids, 2] + c.cube_z_offset
        # RANDOM start orientation (not identity): the hand must reorient from an arbitrary
        # pose to the random goal every episode — the real AllegroHand dexterity challenge,
        # not a fixed-face hold.
        self._cube[rows, 3:7] = _rand_quat(n, self.device)
        self._cube[rows, 7:13] = 0.0
        self.sim.set_free_body_state_tensor_indexed(self._cube, rows)
        self._goal_quat[env_ids] = _rand_quat(n, self.device)
        self._last_action[env_ids] = 0.0
        self.progress_buf[env_ids] = 0.0
        self.rewards.reset(env_ids)
