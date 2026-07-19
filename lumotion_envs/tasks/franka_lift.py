"""Franka Panda lift — grasp a per-env cube and hold it at a goal height (Tier C #2).

IsaacLab lift-style MDP on the free-rigid-body tensor API: the cube is a dynamic
per-env Box observed via acquire_free_body_state_tensor and reset by writing the
batch state back. The MJCF gripper tendon is emulated by mirroring ONE gripper
action onto both finger PD targets — CONTINUOUS: action in [-1, 1] maps linearly
to the finger opening [0, 0.04 m] (IsaacLab's binary open/close is the special
case of saturating this).

Rewards run through rl.RewardManager (dt_scale=False keeps the validated scale)
so the IsaacLab lift curriculum applies natively: modify_reward_weight hardens
action_rate/joint_vel from -1e-4 to -1e-1 after 10k global steps — smoothness is
only enforced once the grasp behavior exists.

Run via the CLI:
    python train.py --task FrankaLift
"""
import math
from pathlib import Path

from lumotion_envs._engine import ensure_engine
ensure_engine()
import lm.rl as rl
from lumotion_envs.config import FrankaLiftConfig

from lumotion_envs.assets import ASSETS
CLIP_OBS = 5.0
MAX_EPISODE_LENGTH = 300
SUBSTEPS = 2
DOF_VEL_SCALE = 0.1
GRIP_RATE = 0.005      # m/step: full 4 cm travel in ~8 steps (~0.16 s at 50 Hz)

# -- reward terms (RewardManager funcs) — the SAME math as the validated v11
# inline reward, split per term so modify_reward_weight can retune weights live.
# They read per-step values cached by _compute_reward (_d_reach/_lifted/_d_goal).

def _r_reach(task):
    # Coarse L2 pull + fine tanh kernel: the tanh(d/0.1) alone is FLAT beyond
    # ~25 cm (saturated), giving zero approach signal from the 42 cm spawn pose.
    # CLAMPED: an ejected cube would otherwise inject a -10^4-scale reward that
    # destroys the value function (v5 collapsed to reward -4.5M).
    d = task._d_reach
    return 1.0 - task._torch.tanh(d / 0.1) - 0.2 * d.clamp(max=1.0)


def _r_lifted(task):
    return task._lifted


def _r_goal(task, std):
    return (1.0 - task._torch.tanh(task._d_goal / std)) * task._lifted


def _r_action_rate(task):
    return (task._last_action - task._prev_action).square().sum(dim=1)


def _r_joint_vel(task):
    return task._dof[:, :9, 1].square().sum(dim=1)


FRANKA_LIFT_PPO_PARAMS = {"params": {
    "network": {"mlp": {"units": [256, 128, 64]}},
    "config": {
        "reward_shaper": {"scale_value": 0.1},
        "critic_coef": 2,
        "bounds_loss_coef": 0.001,
        "entropy_coef": 0.005,
        "horizon_length": 24,
    },
}}


class FrankaLiftTask(rl.VecTask):
    """Vectorized cube lift (fixed-base Franka + gripper + per-env dynamic cube)."""

    def __init__(self, cfg: FrankaLiftConfig = None, *, num_envs=None, headless=None):
        self.cfg = cfg or FrankaLiftConfig()
        if num_envs is not None:
            self.cfg.num_envs = num_envs
        if headless is not None:
            self.cfg.headless = headless
        c = self.cfg
        self.n_dof = int(c.num_dof)              # 9 = 7 arm + 2 fingers
        self.world = rl.World(num_envs=int(c.num_envs), env_spacing=c.env_spacing)
        self.world.add_ground(z=0.0, friction=1.0)
        self.robot = self.world.add_robot(
            rl.Mjcf(str(ASSETS / c.robot), prep=True, config=str(ASSETS / c.rl_yaml)))
        s = float(c.cube_size)
        ph, ps = float(c.pedestal_height), float(c.pedestal_size)
        # IsaacLab layout: the cube sits on a small static pedestal ("table"); knocking
        # it off ends the episode (see _compute_reward) — slapping punishes itself.
        self.world.add_static(rl.Box(size=(ps, ps, ph), color=(0.35, 0.35, 0.4)),
                              at=(c.cube_x, c.cube_y, ph / 2.0), per_env=True)
        # IsaacLab DexCube-style solver props: 16/1 iterations, depenetration capped
        # at 5 m/s, velocity capped — a pinched cube can no longer be ejected at
        # silly speeds (the broadphase-explosion --> OOM chain of the early runs).
        self.world.add_static(rl.Box(size=(s, s, s), dynamic=True, color=(0.85, 0.25, 0.15),
                                     solver_position_iterations=16,
                                     solver_velocity_iterations=1,
                                     max_depenetration_velocity=5.0,
                                     max_linear_velocity=10.0),
                              at=(c.cube_x, c.cube_y, ph + s / 2.0), per_env=True)
        sim, runner = self.world.build(
            headless=c.headless,
            # found/lost pairs headroom: early training resets ~300 envs/step, each
            # re-inserting its arm+cube colliders into the GPU broadphase — bounded
            # churn peaking ~1.3M pairs (the unbounded explosion is gone: cube solver
            # props + indexed resets). 2M gives margin => zero missed interactions.
            config=rl.SimConfig(dt=1.0 / 50.0, substeps=SUBSTEPS, device="auto",
                                gpu_found_lost_pairs_capacity=2_000_000),
            title=f"{c.name} (lift)")
        sim.play()
        # actions: 7 arm + 1 gripper. obs: dof_pos_rel(9) + dof_vel(9) + ee_rel(3)
        # + cube_rel(3) + cube-goal(3) + last_action(8) = 35
        super().__init__(sim, runner, num_obs=35, num_actions=8,
                         name=c.name, clip_obs=CLIP_OBS,
                         max_episode_length=MAX_EPISODE_LENGTH, seed=int(c.seed))
        # IsaacLab lift reward set (weights = the validated v11 inline reward;
        # dt_scale=False keeps the raw per-step scale those weights were tuned at).
        self.rewards = rl.RewardManager(self, [
            rl.RewardTerm("reach",       _r_reach,        1.0),
            rl.RewardTerm("lifted",      _r_lifted,      15.0),
            rl.RewardTerm("goal",        _r_goal,        16.0, {"std": 0.3}),
            rl.RewardTerm("goal_fine",   _r_goal,         5.0, {"std": 0.05}),
            rl.RewardTerm("action_rate", _r_action_rate, -1e-4),
            rl.RewardTerm("joint_vel",   _r_joint_vel,   -1e-4),
        ], dt_scale=False)
        # IsaacLab lift curriculum: harden the smoothness penalties x1000 once the
        # grasp behavior exists. `curriculum_start_steps` is the global-step threshold
        # (~epoch 417 at horizon 24 for a from-scratch run; 0 to harden immediately
        # when REFINING an already-grasping policy via --resume-from, which smooths
        # the end-hold tremble without waiting).
        n = int(getattr(c, "curriculum_start_steps", 10000))
        self.curriculum = rl.CurriculumManager(self, [
            rl.CurrTerm("action_rate", rl.modify_reward_weight,
                        {"term_name": "action_rate", "weight": -1e-1, "num_steps": n}),
            rl.CurrTerm("joint_vel", rl.modify_reward_weight,
                        {"term_name": "joint_vel", "weight": -1e-1, "num_steps": n}),
        ])

    # -- task hooks ---------------------------------------------------------

    def _capture(self):
        torch = self._torch; dev = self.device; c = self.cfg
        self._dof = self.sim.acquire_dof_state_tensor()
        self._root = self.sim.acquire_root_state_tensor()
        self._rigid_state = self.sim.acquire_rigid_body_state_tensor()
        self._cube = self.sim.acquire_free_body_state_tensor()
        self.sim.refresh_dof_state_tensor(); self.sim.refresh_root_state_tensor()
        self._default_dof = self.robot.default_dof_positions.unsqueeze(0).repeat(self.num_envs, 1)
        # Arm vs finger DOF indices come from the name map — batch DOF order is opaque.
        self._finger_idx = torch.tensor(
            self.robot.view.dof_indices(["finger_joint1", "finger_joint2"]),
            device=dev, dtype=torch.long)
        self._arm_idx = torch.tensor(
            self.robot.view.dof_indices([f"joint{i}" for i in range(1, 8)]),
            device=dev, dtype=torch.long)
        for _ in range(20):
            self.sim.set_dof_position_target_tensor(self._default_dof)
            self.sim.simulate(); self.sim.fetch_results()
            if self.runner is not None:
                self.runner.run()
        self.sim.refresh_dof_state_tensor(); self.sim.refresh_root_state_tensor()
        self.sim.refresh_rigid_body_state_tensor(); self.sim.refresh_free_body_state_tensor()
        self._base = self._root[:, 0:3].clone()
        # Grasp point = the midpoint of the two FINGERTIP links (the TCP). The `hand`
        # body origin sits ~10 cm above the fingertips: pulling IT onto the cube drives
        # the fingers into the floor (contact blow-up + no possible grasp).
        lm_ = self.robot.view.link_map
        self._fingertip_idx = self._torch.tensor(
            [i for n, i in lm_.items() if n.lower().endswith("finger")],
            device=self.device, dtype=self._torch.long)
        assert self._fingertip_idx.numel() == 2, f"finger links not found in {sorted(lm_)}"
        # Free-body row -> env: match each settled cube to the nearest base XY (rows
        # follow EnTT order, not the env index; positions are unique per env cell).
        d = (self._cube[:, None, 0:2] - self._base[None, :, 0:2]).norm(dim=2)
        self._cube_env_row = d.argmin(dim=0)          # env -> batch row
        self._goal = self._base.clone()
        self._goal[:, 0] += c.cube_x
        self._goal[:, 2] = c.goal_z
        self._last_action = torch.zeros(self.num_envs, 8, device=dev)
        self._prev_action = torch.zeros(self.num_envs, 8, device=dev)
        # Persistent gripper target (delta-controlled), starts OPEN like the pose.
        self._grip_target = torch.full((self.num_envs, 1), 0.04, device=dev)

    def cube_pos(self):
        return self._cube[self._cube_env_row, 0:3]

    def grasp_pos(self):
        """TCP: midpoint of the two fingertip link positions."""
        return self._rigid_state[:, self._fingertip_idx, 0:3].mean(dim=1)

    def _pre_physics_step(self, actions):
        torch = self._torch
        self._prev_action = self._last_action.clone()
        self._last_action = actions.clone()
        targets = self._default_dof.clone()
        arm_t = self._default_dof[:, self._arm_idx] + self.cfg.action_scale * actions[:, 0:7]
        targets[:, self._arm_idx] = arm_t
        # One gripper action drives both (tendon-coupled on the real robot) —
        # CONTINUOUS as a DELTA-target (IsaacGym-cabinet style): the action is a
        # closing/opening RATE integrated into a persistent target saturating at
        # the rails [0, 0.04]. v12 proved a direct linear mapping kills
        # exploration (policy noise around 0 puts the fingers exactly at the
        # 4 cm cube surface -> zero grip force -> `lifted` never discovered);
        # the integrator random-walks INTO the rails, where full squeeze lives,
        # while the policy can still hold any intermediate aperture.
        self._grip_target = (self._grip_target
                             + GRIP_RATE * actions[:, 7:8]).clamp(0.0, 0.04)
        targets[:, self._finger_idx] = self._grip_target
        self.sim.set_dof_position_target_tensor(targets)

    def _compute_observations(self):
        self.sim.refresh_dof_state_tensor()
        self.sim.refresh_rigid_body_state_tensor()
        self.sim.refresh_free_body_state_tensor()
        ee = self.grasp_pos()
        cube = self.cube_pos()
        o = self.obs_buf
        o[:, 0:9] = self._dof[:, :9, 0] - self._default_dof
        o[:, 9:18] = self._dof[:, :9, 1] * DOF_VEL_SCALE
        o[:, 18:21] = ee - self._base
        o[:, 21:24] = cube - self._base
        o[:, 24:27] = cube - self._goal
        o[:, 27:35] = self._last_action

    def _compute_reward(self):
        torch = self._torch; c = self.cfg
        ee = self.grasp_pos()
        cube = self.cube_pos()
        # Per-step values the RewardManager terms read (computed once, shared).
        self._d_reach = (ee - cube).norm(dim=1)
        # IsaacLab lift MDP: lifted is NOT hold-gated — the anti-batting mechanism is
        # structural instead (knocking the cube off the pedestal terminates below).
        self._lifted = (cube[:, 2] > c.pedestal_height + c.lift_min_height).float()
        self._d_goal = (cube - self._goal).norm(dim=1)
        self.rew_buf = self.rewards.compute()
        # object_dropping: the cube left the pedestal (below its top) -> episode ends.
        # A NON-FINITE cube state counts as dropped too: a hard finger pinch can eject
        # the cube violently or NaN it in deep interpenetration, and clamp(NaN)=NaN
        # would poison the policy (train4 died on 'normal expects std >= 0'). The
        # reset teleport writes a clean state back into PhysX.
        finite = torch.isfinite(cube).all(dim=1)
        ejected = (cube - self._base).norm(dim=1) > 1.5   # pinched-launch cleanup
        self.reset_buf = ((cube[:, 2] < c.pedestal_height - 0.01) | ~finite | ejected).float()
        self.rew_buf = torch.nan_to_num(self.rew_buf, nan=0.0, posinf=0.0, neginf=0.0)
        torch.nan_to_num_(self.obs_buf, nan=0.0, posinf=CLIP_OBS, neginf=-CLIP_OBS)

    def _reset_idx(self, env_ids):
        torch = self._torch
        n = env_ids.numel(); c = self.cfg
        noise = (torch.rand(n, 9, device=self.device) - 0.5) * 2.0 * c.reset_joint_noise
        self._dof[env_ids, :9, 0] = self._default_dof[env_ids] + noise
        self._dof[env_ids, :9, 1] = 0.0
        self.sim.set_dof_state_tensor_indexed(self._dof, env_ids)
        self.sim.set_dof_position_target_tensor_indexed(self._default_dof + 0.0, env_ids)
        # Cube back on the ground at a jittered XY (full-batch free-body write).
        self.sim.refresh_free_body_state_tensor()
        rows = self._cube_env_row[env_ids]
        jit = (torch.rand(n, 2, device=self.device) - 0.5) * 2.0 * c.cube_xy_noise
        self._cube[rows, 0] = self._base[env_ids, 0] + c.cube_x + jit[:, 0]
        self._cube[rows, 1] = self._base[env_ids, 1] + c.cube_y + jit[:, 1]
        self._cube[rows, 2] = c.pedestal_height + c.cube_size / 2.0
        self._cube[rows, 3:6] = 0.0
        self._cube[rows, 6] = 1.0                     # identity quat (xyzw)
        self._cube[rows, 7:13] = 0.0
        # INDEXED write — a full-batch write re-inserts all 4096 cubes into the GPU
        # broadphase every reset step and the found/lost churn explodes (v8: 174
        # foundLostPairs warnings by epoch 91 with zero actual ejections).
        self.sim.set_free_body_state_tensor_indexed(self._cube, rows)
        self._last_action[env_ids] = 0.0
        self._prev_action[env_ids] = 0.0
        self._grip_target[env_ids] = 0.04            # fingers back to open
        self.progress_buf[env_ids] = 0.0
        self.rewards.reset(env_ids)
        self.curriculum.compute(env_ids)


def _frame_camera(task):
    """Close-up on env 0's workspace for small demos; grid overview otherwise."""
    v = task.runner
    if v is None or not hasattr(v, "set_camera"):
        return
    if task.num_envs <= 4:
        # Face the arm from the side of its reach box, looking at the grasp zone.
        v.set_camera(pos=(1.35, -0.95, 0.75), target=(0.45, 0.0, 0.15))
        return
    side = int(math.ceil(math.sqrt(task.num_envs)))
    ext = (side - 1) * task.cfg.env_spacing
    cx = cy = ext / 2.0
    dist = ext * 0.7 + 4.0
    v.set_camera(pos=(cx - dist * 0.5, cy - dist * 0.7, dist * 0.6), target=(cx, cy, 0.3))
