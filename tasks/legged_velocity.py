"""Generic velocity-command quadruped locomotion (LeggedVelocityTask).

ONE task class for any quadruped — ANYmal, Go2, A1, … — selected by a LeggedConfig
(robot asset + prep yaml + dof count + foot/thigh link suffix; PD gains live in the yaml).
The task is DECLARATIVE: observation / termination / command / event / reward are term
lists on the lm.rl managers, so a new robot is a config, not a new task body.

    python train.py --task Anymal       # AnymalConfig
    python train.py --task Go2          # Go2Config

obs(12 + 3*n_dof) = base_lin_vel*2 + base_ang_vel*0.25 + projected_gravity
                    + commands*[2,2,0.25] + dof_pos_rel + dof_vel*0.05 + prev_actions
action(n_dof) -> target = action_scale*action + default stance ; PD from the yaml
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # for lumengine_envs
import _bootstrap
_bootstrap.bootstrap()
import lm.rl as rl
from lumengine_envs.config import LeggedConfig

ASSETS = _bootstrap.ASSETS
SUBSTEPS = 2
LIN_VEL_SCALE, ANG_VEL_SCALE = 2.0, 0.25
DOF_VEL_SCALE = 0.05
CLIP_OBS = 5.0
H_MIN_DROP = 0.30
MAX_EPISODE_LENGTH = 1000
# Forward-first command distribution (escapes the vx=0 crouch local optimum); widen later.
CMD_X, CMD_Y, CMD_YAW = (0.0, 1.5), (-0.5, 0.5), (-0.5, 0.5)

LEGGED_PPO_PARAMS = {"params": {
    "network": {"mlp": {"units": [256, 128, 64]}},
    "config": {
        "reward_shaper": {"scale_value": 1.0},
        "critic_coef": 2,
        "bounds_loss_coef": 0.001,
        "entropy_coef": 0.01,
    },
}}

# Two reward recipes, picked by cfg.reward (see LeggedConfig.reward):
#  - "isaaclab": the 11-term velocity-FLAT set — clean gait for a high-authority robot (anymal).
#  - "genesis":  the minimal Unitree-go2 set (tracking + a few light penalties +
#                similar_to_default). A low-authority robot like Go2 needs this: the heavy
#                IsaacLab penalties (torques/dof_acc/orientation/undesired_contacts/feet_air_time)
#                smother its motion into a crouch, while the minimal set lets it walk.
# Built per-task so the feet_air_time threshold follows the robot's gait speed.
def _reward_terms(cfg):
    if getattr(cfg, "reward", "isaaclab") == "biped":
        # IsaacLab H1 velocity recipe (proven biped). Key biped differences vs the quad recipes:
        # NO lin_vel_z penalty (a biped bobs vertically), track_ang weight 1.0, feet_air_time +
        # feet_slide on the ankle "feet", MILD flat_orientation (-1.0, not -5/-10), and
        # similar_to_default to keep the arms/torso from flailing. No base_height term.
        return [
            rl.RewardTerm("track_lin_vel_xy",   rl.rewards.track_lin_vel_xy_exp,   1.0,    {"std": 0.5}),
            rl.RewardTerm("track_ang_vel_z",    rl.rewards.track_ang_vel_z_exp,    1.0,    {"std": 0.5}),
            rl.RewardTerm("feet_air_time",      rl.rewards.feet_air_time,          0.25,
                          {"threshold": cfg.feet_air_time_threshold}),
            rl.RewardTerm("feet_slide",         rl.rewards.feet_slide,            -0.25),
            rl.RewardTerm("flat_orientation",   rl.rewards.flat_orientation_l2,   -1.0),
            rl.RewardTerm("action_rate",        rl.rewards.action_rate_l2,        -0.005),
            rl.RewardTerm("dof_acc",            rl.rewards.dof_acc_l2,            -1.25e-7),
            rl.RewardTerm("similar_to_default", rl.rewards.joint_deviation_l1,    -0.1),
        ]
    if getattr(cfg, "reward", "isaaclab") == "genesis":
        return [
            # std=0.5 so the exp kernel is exp(-err²/0.25) — matches Genesis tracking_sigma=0.25
            # (our kernel squares std). std=0.25 would be 4× too sharp → ~no signal until already
            # fast → the policy never leaves the stand-still optimum.
            rl.RewardTerm("track_lin_vel_xy",   rl.rewards.track_lin_vel_xy_exp,   1.0,   {"std": 0.5}),
            # ang_vel_z tracking ×2.5 so a yaw=0 command holds a straight heading (the robot
            # was curving while advancing).
            rl.RewardTerm("track_ang_vel_z",    rl.rewards.track_ang_vel_z_exp,    0.5,   {"std": 0.5}),
            rl.RewardTerm("lin_vel_z",          rl.rewards.lin_vel_z_l2,          -1.0),
            # Gait shaping (on the now-correct 50 Hz base): feet_clearance rewards swing feet for
            # LIFTING (incremental, no threshold valley) → a real foot-lifting gait. Tuned DOWN
            # (target 0.05, weight 1.0) — 0.10/2.0 gave an exaggerated prancing lift.
            rl.RewardTerm("feet_clearance",     rl.rewards.feet_clearance,         1.0,   {"target": 0.05}),
            rl.RewardTerm("feet_air_time",      rl.rewards.feet_air_time,          0.5,
                          {"threshold": cfg.feet_air_time_threshold}),
            rl.RewardTerm("action_rate",        rl.rewards.action_rate_l2,        -0.01),
            rl.RewardTerm("similar_to_default", rl.rewards.joint_deviation_l1,    -0.1),
            rl.RewardTerm("base_height",        rl.rewards.base_height_l2,       -50.0),
        ]
    return [
        rl.RewardTerm("track_lin_vel_xy",   rl.rewards.track_lin_vel_xy_exp,   1.0,     {"std": 0.5}),
        rl.RewardTerm("track_ang_vel_z",    rl.rewards.track_ang_vel_z_exp,    0.5,     {"std": 0.5}),
        rl.RewardTerm("lin_vel_z",          rl.rewards.lin_vel_z_l2,          -2.0),
        rl.RewardTerm("ang_vel_xy",         rl.rewards.ang_vel_xy_l2,         -0.05),
        rl.RewardTerm("dof_torques",        rl.rewards.dof_torques_l2,        -2.5e-5),
        rl.RewardTerm("dof_acc",            rl.rewards.dof_acc_l2,            -2.5e-7),
        rl.RewardTerm("action_rate",        rl.rewards.action_rate_l2,        -0.01),
        rl.RewardTerm("feet_air_time",      rl.rewards.feet_air_time,          0.5,
                      {"threshold": cfg.feet_air_time_threshold}),
        rl.RewardTerm("undesired_contacts", rl.rewards.undesired_contacts,    -1.0,     {"threshold": 1.0}),
        rl.RewardTerm("flat_orientation",   rl.rewards.flat_orientation_l2,   -5.0),
        rl.RewardTerm("base_height",        rl.rewards.base_height_l2,       -50.0),
    ]

# Declarative observation vector — velocity_commands already applies the command scale.
OBS_TERMS = [
    rl.ObsTerm("lin_vel",  rl.observations.base_lin_vel,      scale=LIN_VEL_SCALE),
    rl.ObsTerm("ang_vel",  rl.observations.base_ang_vel,      scale=ANG_VEL_SCALE),
    rl.ObsTerm("gravity",  rl.observations.projected_gravity),
    rl.ObsTerm("commands", rl.observations.velocity_commands),
    rl.ObsTerm("dof_pos",  rl.observations.joint_pos_rel),
    rl.ObsTerm("dof_vel",  rl.observations.joint_vel,         scale=DOF_VEL_SCALE),
    rl.ObsTerm("actions",  rl.observations.last_action),
]

def _event_terms(cfg):
    """Reset events. 'scaled' = default*U(0.5,1.5) (IsaacGymEnvs anymal); 'offset' = default +
    small noise (a gentle spawn near the stance, so a stiff PD snap doesn't tip a light robot)."""
    if cfg.reset_mode == "offset":
        reset_joints = rl.EventTerm("reset_joints", rl.events.reset_joints_offset, mode="reset",
                                    params={"pos_noise": 0.1, "vel_noise": 0.1})
    else:
        reset_joints = rl.EventTerm("reset_joints", rl.events.reset_joints_scaled, mode="reset",
                                    params={"lo": 0.5, "hi": 1.5, "vel_noise": 0.1})
    return [rl.EventTerm("reset_root", rl.events.reset_root_to_home, mode="reset"), reset_joints]


def _quat_rotate_inv(q, v):
    import torch
    qv, qw = -q[:, 0:3], q[:, 3:4]
    t = 2.0 * torch.cross(qv, v, dim=1)
    return v + qw * t + torch.cross(qv, t, dim=1)


def _parse_cmd(s):
    if not s:
        return None
    return [float(x) for x in str(s).split(",")]


def _make_morph(path, rl_yaml):
    """Pick the morph by file extension: .usd* -> Usd, .urdf -> Urdf, .xml -> Mjcf."""
    ext = Path(path).suffix.lower()
    if ext in (".usd", ".usda", ".usdc"):
        return rl.Usd(str(path), prep=True, config=str(rl_yaml))
    if ext == ".urdf":
        return rl.Urdf(str(path), prep=True, config=str(rl_yaml))
    if ext == ".xml":
        return rl.Mjcf(str(path), prep=True, config=str(rl_yaml))
    raise ValueError(f"unsupported robot asset extension: {ext}")


class LeggedVelocityTask(rl.VecTask):
    """Velocity-command locomotion for any quadruped, configured by a LeggedConfig."""

    def __init__(self, cfg: LeggedConfig = None, *, num_envs=None, headless=None):
        self.cfg = cfg or LeggedConfig()
        if num_envs is not None:
            self.cfg.num_envs = num_envs
        if headless is not None:
            self.cfg.headless = headless
        c = self.cfg
        self.n_dof = int(c.num_dof)
        headless = c.headless
        gz = c.ground_z
        instanceable = bool(headless) if c.instance == "auto" else (c.instance == "on")
        self.world = rl.World(num_envs=int(c.num_envs), env_spacing=c.env_spacing)
        self.curr = None
        if c.terrain == "curriculum":
            self.world.add_terrain_curriculum(
                types=[
                    lambda d: rl.terrain.Slope(slope=0.05 + 0.45 * d, axis="x"),
                    lambda d: rl.terrain.Stairs(step_height=0.02 + 0.10 * d, step_width=0.4, platform=1.5),
                    lambda d: rl.terrain.Noise(amplitude=0.04 + 0.30 * d, base_cells=4, seed=0),
                ],
                num_levels=c.curriculum_levels, max_init_level=c.curriculum_init,
                size_m=c.curriculum_size, friction=1.0, base_z=gz)
        elif c.terrain == "variants":
            self.world.add_terrain_variants([
                ("flat",   rl.terrain.Flat()),
                ("noise",  rl.terrain.Noise(amplitude=0.18, base_cells=4, seed=0)),
                ("slope",  rl.terrain.Slope(slope=0.15, axis="x")),
                ("stairs", rl.terrain.Stairs(step_width=0.4, step_height=0.06)),
            ], size_m=c.env_spacing, friction=1.0, base_z=gz)
            self.world.assign_terrain(strategy=c.terrain_strategy, seed=0)
        elif c.terrain == "noise":
            self.world.add_terrain(rl.terrain.Noise(amplitude=c.terrain_amp, base_cells=c.terrain_cells, seed=0),
                                   friction=1.0, base_z=gz)
        else:  # "flat"
            self.world.add_ground(z=gz, friction=1.0)
        if c.scatter > 0:
            self.world.scatter(rl.Cylinder(radius=0.2, height=0.8, color=(0.7, 0.4, 0.2)),
                               count=c.scatter, area=(2.5, 2.5), z=gz + 0.4,
                               per_env=True, seed=0, clearance=1.2)
        self.robot = self.world.add_robot(
            _make_morph(ASSETS / c.robot, ASSETS / c.rl_yaml), spawn_z=c.spawn_z)
        _gpu_mult = max(1.0, int(c.num_envs) / 512.0)
        if c.terrain == "curriculum":
            _gpu_mult = max(_gpu_mult, int(c.num_envs) / 256.0, 2.0)
        sim, runner = self.world.build(
            headless=headless, instanceable=instanceable,
            config=rl.SimConfig(dt=1.0 / 50.0, substeps=SUBSTEPS, device="auto",
                                gpu_contact_buffer_multiplier=_gpu_mult),
            title=f"{getattr(c, 'name', type(self).__name__)} (vel-cmd)")
        sim.play()
        super().__init__(sim, runner, num_obs=12 + 3 * self.n_dof, num_actions=self.n_dof,
                         name=getattr(c, "name", "Legged"),
                         clip_obs=CLIP_OBS, max_episode_length=MAX_EPISODE_LENGTH, seed=int(c.seed))
        self._nstep = 0
        self.curr = self.world.make_terrain_curriculum(self.device)

    # -- task hooks ---------------------------------------------------------

    def _capture(self):
        torch = self._torch; dev = self.device; nd = self.n_dof
        self._dof = self.sim.acquire_dof_state_tensor()
        self._root = self.sim.acquire_root_state_tensor()
        self._contact = self.sim.acquire_link_net_contact_force_tensor()
        self._rigid_state = self.sim.acquire_rigid_body_state_tensor()  # per-link [pos,quat,lin,ang]
        self.sim.refresh_dof_state_tensor(); self.sim.refresh_root_state_tensor()
        self._default_dof = self.robot.default_dof_positions.unsqueeze(0).repeat(self.num_envs, 1)
        self._world_down = torch.tensor([0.0, 0.0, -1.0], device=dev).repeat(self.num_envs, 1)
        self._cmd_scale = torch.tensor([LIN_VEL_SCALE, LIN_VEL_SCALE, ANG_VEL_SCALE], device=dev)
        self._prev_action = torch.zeros(self.num_envs, nd, device=dev)
        self._last_action = torch.zeros(self.num_envs, nd, device=dev)
        self._cmd = _parse_cmd(self.cfg.cmd)
        # Settle onto the feet before capturing the home/reset pose (the batch is "ready"
        # before the robot settles, else every reset drops into a crouch). A biped must
        # capture EARLY (short settle) — it tips open-loop before a quadruped-length settle.
        for _ in range(int(getattr(self.cfg, "settle_steps", 80))):
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
        self.base_height_target = self._root[:, 2].clone()

        self.dt = float(self.sim._cfg.dt) * int(self.control_freq_inv)
        self._dof_force = self.sim.acquire_applied_dof_force_tensor()
        self._prev_dof_vel = torch.zeros(self.num_envs, nd, device=dev)
        lm = self.robot.view.link_map
        foot = self.cfg.foot_suffix.upper()
        thigh = self.cfg.thigh_suffix.upper()
        self.feet_indices = torch.tensor(
            [i for n, i in lm.items() if n.upper().endswith(foot)], device=dev, dtype=torch.long)
        self.undesired_contact_indices = torch.tensor(
            [i for n, i in lm.items() if n.upper().endswith(thigh)], device=dev, dtype=torch.long)
        self.knee_indices = self.undesired_contact_indices
        nf = int(self.feet_indices.numel())
        self._air_time = torch.zeros(self.num_envs, nf, device=dev)
        self._prev_in_contact = torch.zeros(self.num_envs, nf, dtype=torch.bool, device=dev)
        # Foot kinematics for feet_clearance/feet_slide: stance height = the settled foot-link z
        # (baseline), so clearance = current foot z − stance z (height of the lifted foot).
        self.sim.refresh_rigid_body_state_tensor()
        self._foot_stance_z = self._rigid_state[:, self.feet_indices, 2].clone()
        self.feet_pos = self._rigid_state[:, self.feet_indices, 0:3]
        self.feet_vel = self._rigid_state[:, self.feet_indices, 7:10]
        self.feet_clearance = self.feet_pos[:, :, 2] - self._foot_stance_z

        # Managers (declarative task definition):
        self.rewards = rl.RewardManager(self, _reward_terms(self.cfg))
        self.observations = rl.ObservationManager(self, OBS_TERMS)
        self.terminations = rl.TerminationManager(self, [
            rl.TermTerm("base_contact", rl.terminations.base_contact,
                        {"threshold": self.cfg.base_contact_fail_n, "link": 0}),
            rl.TermTerm("knee_contact", rl.terminations.contact_on,
                        {"indices": self.knee_indices, "threshold": self.cfg.base_contact_fail_n}),
            # Tipped over: some robots (e.g. Go2) splay their legs and never register a base/
            # knee contact when they fall — terminate on orientation so collapsing costs the
            # episode (else the policy learns a sprawled, non-walking posture).
            rl.TermTerm("tipped", rl.terminations.bad_orientation, {"min_up": self.cfg.upright_min}),
        ])
        self.commands_mgr = rl.CommandManager(self, {"base_velocity": rl.UniformVelocityCommand(
            self, vx=CMD_X, vy=CMD_Y, yaw=CMD_YAW, fixed=self._cmd)})
        self.events = rl.EventManager(self, _event_terms(self.cfg))

    def _pre_physics_step(self, actions):
        self._prev_action = self._last_action
        self._last_action = actions
        self.sim.set_dof_position_target_tensor(self._default_dof + self.cfg.action_scale * actions)

    def _update_state(self):
        torch = self._torch; nd = self.n_dof
        self.sim.refresh_root_state_tensor()
        self.sim.refresh_applied_dof_force_tensor()
        self.sim.refresh_link_net_contact_force_tensor()
        self.sim.refresh_rigid_body_state_tensor()
        self.feet_pos = self._rigid_state[:, self.feet_indices, 0:3]
        self.feet_vel = self._rigid_state[:, self.feet_indices, 7:10]
        self.feet_clearance = self.feet_pos[:, :, 2] - self._foot_stance_z   # foot height above stance
        root, dof = self._root, self._dof
        quat = root[:, 3:7]
        self.base_height = root[:, 2]
        self.lin_vel_b = _quat_rotate_inv(quat, root[:, 7:10])
        self.ang_vel_b = _quat_rotate_inv(quat, root[:, 10:13])
        self.proj_gravity = _quat_rotate_inv(quat, self._world_down)
        self._up_proj = self.up_proj = -self.proj_gravity[:, 2]
        self.dof_pos_rel = dof[:, :nd, 0] - self._default_dof
        self.dof_vel = dof[:, :nd, 1]
        self.dof_acc = (self.dof_vel - self._prev_dof_vel) / self.dt
        self._prev_dof_vel = self.dof_vel.clone()
        self.dof_torque = self._dof_force[:, :nd]
        self.contact_forces = self._contact
        self.actions, self.prev_actions = self._last_action, self._prev_action
        self.commands = self.commands_mgr.get("base_velocity")
        ff = self.contact_forces[:, self.feet_indices, :].norm(dim=2)
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
        self.rew_buf = self.rewards.compute()
        self.reset_buf = self.terminations.compute()
        grace = int(getattr(self.cfg, "contact_grace_steps", 0))
        if grace > 0:
            # A just-reset env can register a 1-frame teleport contact spike (the biped
            # momentarily penetrates the ground before settling onto its feet). Don't let a
            # contact-based termination fire during the short post-reset grace window.
            self.reset_buf = self.reset_buf * (self.progress_buf >= float(grace))
        self._nstep += 1
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
            print(f"[legged-dbg] {self.name} step {self._nstep} | ep_len(mean)={float(self.progress_buf.mean()):.1f} "
                  f"| rew={float(self.rew_buf.mean()):.3f} | vx={float(self.lin_vel_b[:,0].mean()):+.2f} "
                  f"(cmd_x~{float(cmd[0]):+.2f}) | upright={float(self._up_proj.mean()):.2f} "
                  f"| height={float(self.base_height.mean()):.2f} | air={float(self._air_time.mean()):.2f}"
                  f" | fails={int(self.reset_buf.sum())}{curr_s}", flush=True)

    def _reset_idx(self, ids):
        if self.curr is not None:
            dist = (self._root[ids, 0:2] - self._home[ids, 0:2]).norm(dim=1)
            move_up = dist > self.curr.size_m * 0.5
            move_down = (dist < 0.5) & (~move_up)
            self.curr.update(ids, move_up, move_down)
            self._home[ids, 0:3] = self.curr.env_origins[ids]
            self._h_min[ids] = self.curr.env_origins[ids, 2] - H_MIN_DROP
        self.events.reset(ids)
        self.commands_mgr.resample(ids)
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
