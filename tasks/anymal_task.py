"""ANYmal velocity-command locomotion on the USD world, trained by rl_games.

Production path: a VecTask-shaped task the rl_games adapter (lm.rl/_rlgames.py) drives.
The robot is the REAL anymal_c (anymal.urdf -> USD via urdf_usd_converter), authored via
the rl.World facade; floating base + PD drives are applied by lm.rl's config-driven prep
from assets/anymal_c.rl.yaml. Task = IsaacGymEnvs Anymal: track a commanded base velocity
(lin x/y + yaw) sampled per episode, under PD position control. Z-up.

    set LM_PHYSX_SHARE_CUDA_CONTEXT=1
    python anymal_task.py                       # headless train (-> runs/<...>/nn/*.pth)
    LM_RL_VIEW=1 python anymal_task.py           # windowed: watch it train live
    LM_RL_PLAY=runs/.../nn/Anymal.pth python anymal_task.py   # watch a trained policy

obs(48) = base_lin_vel*2 + base_ang_vel*0.25 + projected_gravity + commands*[2,2,0.25]
          + dof_pos_rel + dof_vel*0.05 + prev_actions
action(12) -> target = 0.5*action + default_dof ; PD Kp=85 Kd=2
reward = exp(-||cmd_xy - v_xy||^2/0.25)*1 + exp(-(cmd_yaw - w_z)^2/0.25)*0.5, clipped>=0
"""
import faulthandler
import math
import os
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))   # _bootstrap
import _bootstrap
_bootstrap.bootstrap()
import lm.rl as rl

faulthandler.enable()

_ROBOT = _bootstrap.ASSETS / "anymal_converted" / "anymal.usda"
_CFG   = _bootstrap.ASSETS / "anymal_c.rl.yaml"   # floating base + PD drives (config-driven prep)

NUM_ENVS    = int(os.environ.get("LM_RL_NUM_ENVS", "4096"))   # PPO wants a big batch (Isaac uses 4096)
ENV_SPACING = float(os.environ.get("LM_RL_SPACING", "4.0"))
N_DOF       = 12
GROUND_Z    = -0.65
SUBSTEPS    = 2

LIN_VEL_SCALE, ANG_VEL_SCALE = 2.0, 0.25
DOF_VEL_SCALE = 0.05
ACTION_SCALE  = 0.5
CLIP_OBS      = 5.0
TRACK_SIGMA   = 0.25
REW_LIN_XY, REW_ANG_Z = 1.0, 0.5
# Forward-first command distribution: the bare velocity-tracking reward leaves the policy
# stuck at vx=0 (crouch + step in place) because backward/lateral/large-yaw commands make the
# task too hard to bootstrap. Bias the commands forward (no backward, modest lateral/yaw) so
# forward walking is the dominant skill to learn; widen later once it walks. (Genesis's simple
# demo trains forward-only; IsaacLab gets the full range only with 4096 envs + 300 iters.)
CMD_X, CMD_Y, CMD_YAW = (0.0, 1.5), (-0.5, 0.5), (-0.5, 0.5)
# Height-fall threshold as an ABSOLUTE drop below the spawn height (a fraction of the
# spawn z is too tight here: the anymal base sits at ~0.12 m, so 0.55x = a 4 cm margin
# that a normal gait bob trips). Fall = base dropped > H_MIN_DROP below where it spawned.
H_MIN_DROP, UPRIGHT_MIN = 0.30, 0.6
MAX_EPISODE_LENGTH = 1000
# Net contact force on the base link (articulation link 0) above this => the trunk is on
# the ground = a fall (complements the height/upright test with a real contact signal).
BASE_CONTACT_FAIL_N = float(os.environ.get("LM_RL_BASE_CONTACT_FAIL_N", "1.0"))

# IsaacGymEnvs AnymalPPO.yaml training config, deep-merged into rl.train_rl_games's base. The
# default rl_games config is cartpole-sized ([32,32] MLP) — too small to represent a
# coordinated gait, so a faithful reward still plateaus at "stand and survive". This is
# the locomotion-sized network + the matching reward scale that lets Isaac learn from
# the SAME reward/data: [256,128,64] net, reward_shaper 1.0, horizon 24, mini_epochs 5.
ANYMAL_PPO_PARAMS = {"params": {
    "network": {"mlp": {"units": [256, 128, 64]}},
    "config": {
        "reward_shaper": {"scale_value": 1.0},
        "critic_coef": 2,
        "bounds_loss_coef": 0.001,
        # A little entropy to escape the "step in place at vx=0" local optimum: the tracking
        # reward gradient is weak far from the target, so the policy needs to explore enough
        # to discover that striding forward pays off.
        "entropy_coef": 0.01,
    },
}}

# IsaacLab velocity-FLAT reward set (the one that produces a clean walking gait, not just
# velocity-tracking-by-sliding). Weights are the IsaacLab values applied as-is: lm.rl's
# RewardManager multiplies each by dt exactly like IsaacLab's reward_manager (func*weight*dt),
# so the weights port directly. The gait-shaping terms (feet_air_time lifts the feet,
# flat_orientation + lin_vel_z keep the trunk level, action_rate + dof_acc smooth the motion)
# are what was missing from the bare 3-term IsaacGymEnvs reward. All terms already live in
# lm/rl/_rewards.py and read state set up in _update_state.
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
    # Anti-crouch: pull the base back to its captured standing height (target set per-env in
    # _capture). Not an IsaacLab term, but it cleans up the low/crouched gait a small budget
    # converges to. -10 was too weak (the gait stayed crouched at -0.34 vs -0.17 stand); -50
    # is Genesis's value, strong enough to actually keep the trunk up.
    rl.RewardTerm("base_height",        rl.rewards.base_height_l2,       -50.0),
]


def _quat_rotate_inv(q, v):
    import torch
    qv, qw = -q[:, 0:3], q[:, 3:4]
    t = 2.0 * torch.cross(qv, v, dim=1)
    return v + qw * t + torch.cross(qv, t, dim=1)


class AnymalTask(rl.VecTask):
    """ANYmal velocity-command locomotion on rl.VecTask: the base owns the step/reset
    loop; the hooks below define the obs/reward/reset and the PD-target action."""

    def __init__(self, num_envs=NUM_ENVS, headless=True):
        # instanceable shares ONE composed robot prototype across all envs instead of expanding
        # the 60-link subtree per env (at 2048: ~530k-prim stage / ~20 GB RAM -> ~2k prims).
        # Anymal uses the default physics material so per-env friction DR still works (build-time
        # per-articulation un-sharing); the per-env collision id + root pose live on the instance
        # prim. ONLY when headless: the windowed render delegate does not draw USD instance
        # proxies, so the viewer would show an empty scene -> use real prims when rendering.
        # LM_RL_INSTANCE forces instancing on/off (test hook for the windowed render path);
        # default: only headless (training), real prims when rendering.
        _inst_env = os.environ.get("LM_RL_INSTANCE")
        instanceable = bool(headless) if _inst_env is None else (_inst_env == "1")
        # USD-native World: shared ground/terrain + the anymal morph. The morph's
        # config-driven prep (assets/anymal_c.rl.yaml) frees the base and authors the 12
        # PD DriveAPI joints. LM_RL_TERRAIN_VARIANTS=1 gives each env one of K terrain types
        # (USD variantSet) with per-env spawn-from-terrain; LM_RL_TERRAIN=1 is a single
        # shared noise terrain (amplitude LM_RL_TERRAIN_AMP, default 0.10 m, gentle so the
        # fixed spawn clears it); else flat ground.
        self.world = rl.World(num_envs=int(num_envs), env_spacing=ENV_SPACING)
        self.curr = None   # terrain difficulty curriculum (set after super().__init__)
        if os.environ.get("LM_RL_CURRICULUM"):
            # Terrain difficulty curriculum: a static grid of (level x type) tiles; each env
            # starts easy and is moved up/down levels at reset by how far it walked. Level l
            # uses difficulty d = l/(L-1), harder = steeper slope / taller steps / more noise.
            nlev = int(os.environ.get("LM_RL_CURRICULUM_LEVELS", "8"))
            csize = float(os.environ.get("LM_RL_CURRICULUM_SIZE", "8.0"))   # tile side (m) — room
            self.world.add_terrain_curriculum(                             # to walk + spread robots
                types=[
                    lambda d: rl.terrain.Slope(slope=0.05 + 0.45 * d, axis="x"),
                    lambda d: rl.terrain.Stairs(step_height=0.02 + 0.10 * d,
                                                step_width=0.4, platform=1.5),
                    lambda d: rl.terrain.Noise(amplitude=0.04 + 0.30 * d, base_cells=4, seed=0),
                ],
                num_levels=nlev, max_init_level=int(os.environ.get("LM_RL_CURRICULUM_INIT", "1")),
                size_m=csize, friction=1.0, base_z=GROUND_Z)
        elif os.environ.get("LM_RL_TERRAIN_VARIANTS"):
            # Per-env terrain via a USD variantSet: each env SELECTS one of K terrain types
            # (round_robin by default). Concrete per-env tiles (issue #95).
            self.world.add_terrain_variants([
                ("flat",   rl.terrain.Flat()),
                ("noise",  rl.terrain.Noise(amplitude=0.18, base_cells=4, seed=0)),
                ("slope",  rl.terrain.Slope(slope=0.15, axis="x")),
                ("stairs", rl.terrain.Stairs(step_width=0.4, step_height=0.06)),
            ], size_m=ENV_SPACING, friction=1.0, base_z=GROUND_Z)
            self.world.assign_terrain(
                strategy=os.environ.get("LM_RL_TERRAIN_STRATEGY", "round_robin"), seed=0)
        elif os.environ.get("LM_RL_TERRAIN"):
            amp = float(os.environ.get("LM_RL_TERRAIN_AMP", "0.10"))
            cells = int(os.environ.get("LM_RL_TERRAIN_CELLS", "4"))   # more = tighter, steeper hills
            self.world.add_terrain(rl.terrain.Noise(amplitude=amp, base_cells=cells, seed=0),
                                   friction=1.0, base_z=GROUND_Z)
        else:
            self.world.add_ground(z=GROUND_Z, friction=1.0)
        # LM_RL_SCATTER=N scatters N cylinder obstacles per env (concrete colliders),
        # kept clear of the robot spawn. Demonstrates the World scatter; the flat-ground
        # policy isn't obstacle-aware, so it bumps into them (they're solid).
        n_scatter = int(os.environ.get("LM_RL_SCATTER", "0"))
        if n_scatter > 0:
            self.world.scatter(rl.Cylinder(radius=0.2, height=0.8, color=(0.7, 0.4, 0.2)),
                               count=n_scatter, area=(2.5, 2.5), z=GROUND_Z + 0.4,
                               per_env=True, seed=0, clearance=1.2)
        self.robot = self.world.add_robot(
            rl.Usd(str(_ROBOT), prep=True, config=str(_CFG)), spawn_z=0.0)
        # Scale the GPU contact-pair / collision-stack buffers with env count so large-N runs
        # (1k-4k) don't overflow PhysX defaults. The terrain curriculum needs MORE: its tiles
        # are triangle-mesh colliders and many robots cluster on the same low-difficulty tile
        # (lots of broadphase pairs + contacts), so bump the multiplier when it is active.
        _gpu_mult = max(1.0, int(num_envs) / 512.0)
        if os.environ.get("LM_RL_CURRICULUM"):
            _gpu_mult = max(_gpu_mult, int(num_envs) / 256.0, 2.0)
        sim, runner = self.world.build(
            headless=headless, instanceable=instanceable,
            # 50 Hz control (dt=0.02) to match IsaacLab: the dt-scaled penalty weights
            # (dof_acc ∝ 1/dt², etc.) are tuned for 50 Hz, so 60 Hz over-weighted them.
            config=rl.SimConfig(dt=1.0 / 50.0, substeps=SUBSTEPS, device="auto",
                                gpu_contact_buffer_multiplier=_gpu_mult),
            title="ANYmal (rl_games, vel-cmd)")
        sim.play()
        super().__init__(sim, runner, num_obs=48, num_actions=N_DOF, name="Anymal",
                         clip_obs=CLIP_OBS, max_episode_length=MAX_EPISODE_LENGTH,
                         seed=int(os.environ.get("LM_RL_SEED", "0")))
        self._nstep = 0
        self._drive = None   # set to the _DRIVE list (windowed play) -> live UI command
        # Build the runtime terrain curriculum now that the device is resolved (None if
        # LM_RL_CURRICULUM was not requested). _reset_idx drives the level promotion.
        self.curr = self.world.make_terrain_curriculum(self.device)

    # -- task hooks ---------------------------------------------------------

    def _capture(self):
        torch = self._torch; dev = self.device
        self._dof = self.sim.acquire_dof_state_tensor()
        self._root = self.sim.acquire_root_state_tensor()
        self._contact = self.sim.acquire_link_net_contact_force_tensor()   # (envs, links, 3)
        self.sim.refresh_dof_state_tensor(); self.sim.refresh_root_state_tensor()
        # Authored PD stance from anymal_c.rl.yaml, in engine DOF order (via the RobotView
        # name->index map) — no longer inferred from the settled state. Per-env shape
        # (num_envs, N_DOF) so the reset (which indexes [ids]) and obs/action broadcasting
        # all work unchanged.
        self._default_dof = self.robot.default_dof_positions.unsqueeze(0).repeat(self.num_envs, 1)
        self._world_down = torch.tensor([0.0, 0.0, -1.0], device=dev).repeat(self.num_envs, 1)
        self._cmd_scale = torch.tensor([LIN_VEL_SCALE, LIN_VEL_SCALE, ANG_VEL_SCALE], device=dev)
        self._prev_action = torch.zeros(self.num_envs, N_DOF, device=dev)
        self._last_action = torch.zeros(self.num_envs, N_DOF, device=dev)
        self._commands = torch.zeros(self.num_envs, 3, device=dev)
        # Settle onto the feet before capturing the home/reset pose. The direct-GPU batch
        # reports "ready" a few steps after build — well before the robot has settled from
        # its spawn height onto its feet — so capturing home now would make every episode
        # reset to a too-high pose that drops into a crouch. Hold the default PD stance for a
        # moment so `home` is a clean stand at the natural height.
        for _ in range(80):
            self.sim.set_dof_position_target_tensor(self._default_dof)
            self.sim.simulate(); self.sim.fetch_results()
            if self.runner is not None:
                self.runner.run()
        self.sim.refresh_dof_state_tensor(); self.sim.refresh_root_state_tensor()
        # Per-env spawn pose: capture the SETTLED root x/y/z per env. With terrain
        # variants each env's robot is lifted to its own terrain height, so reset AND the
        # fall threshold are per-env — a uniform mean would re-explode robots on raised
        # terrain when they reset.
        self._h_min = self._root[:, 2] - H_MIN_DROP
        home = torch.zeros(self.num_envs, 13, device=dev)
        home[:, 0:3] = self._root[:, 0:3]
        home[:, 6] = 1.0
        self._home = home
        # Target for the base_height reward: the settled standing height (per-env).
        self.base_height_target = self._root[:, 2].clone()

        # --- reward-manager state (B1) ---------------------------------------
        # control step (for dof_acc + IsaacLab dt-scaled reward weights).
        self.dt = float(self.sim._cfg.dt) * int(self.control_freq_inv)
        # applied joint torques (for the torque penalty) + acceleration tracking.
        self._dof_force = self.sim.acquire_applied_dof_force_tensor()
        self._prev_dof_vel = torch.zeros(self.num_envs, N_DOF, device=dev)
        # Feet + undesired-contact link rows, picked by name pattern from the engine link
        # map (anymal: *_FOOT, *_THIGH/*_SHANK). Drive the feet_air_time / undesired_contacts
        # terms off the per-link contact-force tensor.
        lm = self.robot.view.link_map
        # Feet = the FOOT links (where ground contact now correctly lands after the engine fix
        # that stops giving collider-less-but-inertia-authored links a spurious fallback sphere;
        # before it, the anymal rested on fallback spheres on the SHANK links and the FOOT links
        # read 0 N). Verified: standing stance -> each FOOT ~63 N (~weight/4), SHANK ~0.
        self.feet_indices = torch.tensor(
            [i for n, i in lm.items() if n.upper().endswith("FOOT")], device=dev, dtype=torch.long)
        # Undesired contact = THIGH only (IsaacLab): a distinct link that should never touch.
        self.undesired_contact_indices = torch.tensor(
            [i for n, i in lm.items() if n.upper().endswith("THIGH")], device=dev, dtype=torch.long)
        # Termination bodies: base + THIGH (knee) only (IsaacGymEnvs' anymal terminates on these).
        self.knee_indices = torch.tensor(
            [i for n, i in lm.items() if n.upper().endswith("THIGH")], device=dev, dtype=torch.long)
        nf = int(self.feet_indices.numel())
        self._air_time = torch.zeros(self.num_envs, nf, device=dev)
        self._prev_in_contact = torch.zeros(self.num_envs, nf, dtype=torch.bool, device=dev)
        self.rewards = rl.RewardManager(self, REWARD_TERMS)

    def _sample_commands(self, ids):
        torch = self._torch; dev = self.device; n = ids.numel()
        if self._drive is not None:   # live UI driving: keep the current command on reset
            self._commands[ids, 0] = self._drive[0]
            self._commands[ids, 1] = self._drive[1]
            self._commands[ids, 2] = self._drive[2]
            return
        # LM_RL_CMD="vx,vy,yaw" forces ALL envs to the same fixed command (demo/play:
        # e.g. "1,0,0" = everyone walks forward, so following is obvious). Default =
        # per-env random (the IsaacGym omnidirectional training distribution).
        fixed = os.environ.get("LM_RL_CMD")
        if fixed:
            c = [float(x) for x in fixed.split(",")]
            self._commands[ids, 0] = c[0]; self._commands[ids, 1] = c[1]; self._commands[ids, 2] = c[2]
            return
        self._commands[ids, 0] = torch.empty(n, device=dev).uniform_(*CMD_X)
        self._commands[ids, 1] = torch.empty(n, device=dev).uniform_(*CMD_Y)
        self._commands[ids, 2] = torch.empty(n, device=dev).uniform_(*CMD_YAW)

    def _pre_physics_step(self, actions):
        # `actions` is already clipped to [-1, 1] by the base.
        if self._drive is not None:   # live UI command -> every env, every step
            self._commands[:, 0] = self._drive[0]
            self._commands[:, 1] = self._drive[1]
            self._commands[:, 2] = self._drive[2]
        self._prev_action = self._last_action    # action_rate penalty reads both
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
        self._up_proj = -self.proj_gravity[:, 2]
        self.dof_vel = dof[:, :N_DOF, 1]
        self.dof_acc = (self.dof_vel - self._prev_dof_vel) / self.dt
        self._prev_dof_vel = self.dof_vel.clone()
        self.dof_torque = self._dof_force[:, :N_DOF]
        self.contact_forces = self._contact
        self.actions, self.prev_actions = self._last_action, self._prev_action
        self.commands = self._commands
        # feet air time: accumulate while airborne; the reward reads it at the first contact.
        ff = self.contact_forces[:, self.feet_indices, :].norm(dim=2)     # (envs, n_feet)
        in_contact = ff > 1.0
        self.feet_first_contact = in_contact & (~self._prev_in_contact)
        self.feet_air_time = self._air_time.clone()
        self._air_time = torch.where(in_contact, torch.zeros_like(self._air_time),
                                     self._air_time + self.dt)
        self._prev_in_contact = in_contact

    def _compute_observations(self):
        self._update_state()
        o = self.obs_buf
        o[:, 0:3] = self.lin_vel_b * LIN_VEL_SCALE
        o[:, 3:6] = self.ang_vel_b * ANG_VEL_SCALE
        o[:, 6:9] = self.proj_gravity
        o[:, 9:12] = self.commands * self._cmd_scale
        o[:, 12:24] = self._dof[:, :N_DOF, 0] - self._default_dof
        o[:, 24:36] = self.dof_vel * DOF_VEL_SCALE
        o[:, 36:48] = self._last_action

    def _compute_reward(self):
        torch = self._torch
        # IsaacLab convention: sum the weighted terms WITHOUT clamping at 0 — the penalty
        # terms (flat_orientation, lin_vel_z, ...) must keep their negative gradient so the
        # policy is pushed to fix a bad posture, not just collapse to a clamped-zero reward.
        self.rew_buf = self.rewards.compute()

        height = self._root[:, 2]
        # Fall = base OR thigh (knee) on the ground (IsaacGymEnvs' anymal termination). NO
        # height/upright test (too tight here) and NO shank (it touches in a normal gait).
        base_contact = self._contact[:, 0, :].norm(dim=1)
        knee_contact = self._contact[:, self.knee_indices, :].norm(dim=2)              # (envs, 4)
        f_c = (base_contact > BASE_CONTACT_FAIL_N) | (knee_contact > BASE_CONTACT_FAIL_N).any(dim=1)
        fail = f_c
        self.reset_buf = fail.float()
        if self._nstep % 500 == 0:
            print(f"[anymal-dbg]   FAIL: contact(base+knee)={int(f_c.sum())}", flush=True)

        self._nstep += 1
        # Instanced fleet (LM_RL_INSTANCE=1): once instancing has resolved (a few rendered
        # frames) AND the articulations are built (a few steps), bind the shared render
        # copies to the per-env articulation poses so the geometry animates per-env.
        if self._nstep == 10 and not getattr(self, "_driven", False):
            self._driven = True
            try:
                import lm.physx as _physx
                n = _physx.drive_instanced_articulations(self.sim.scene)
                print(f"[anymal-dbg] drive_instanced_articulations -> {n} prototype(s) driven", flush=True)
            except Exception as _e:
                print(f"[anymal-dbg] drive_instanced_articulations failed: {_e}", flush=True)
        if self._nstep % 500 == 0:
            cmd = self._commands.mean(0)
            curr_s = f" | curr_level(mean)={self.curr.mean_level():.2f}/{self.curr.max_level-1}" \
                     if self.curr is not None else ""
            print(f"[anymal-dbg] step {self._nstep} | ep_len(mean)={float(self.progress_buf.mean()):.1f} "
                  f"| rew={float(self.rew_buf.mean()):.3f} | vx={float(self.lin_vel_b[:,0].mean()):+.2f} "
                  f"(cmd_x~{float(cmd[0]):+.2f}) | upright={float(self._up_proj.mean()):.2f} "
                  f"| height={float(height.mean()):.2f} | air={float(self._air_time.mean()):.2f}"
                  f" | fails={int(fail.sum())}{curr_s}", flush=True)
            ep = self.rewards.episode_means()
            print("[anymal-dbg]   rew terms: " + " ".join(f"{k}={v:+.3f}" for k, v in ep.items()), flush=True)

    def _reset_idx(self, ids):
        torch = self._torch; n = ids.numel(); dev = self.device
        if self.curr is not None:
            # Terrain curriculum (mirrors IsaacLab's terrain_levels_vel): promote the env if
            # it walked across its tile, demote if it barely moved, then respawn it on the
            # (new) tile. _home + the per-env fall threshold follow the new tile height.
            dist = (self._root[ids, 0:2] - self._home[ids, 0:2]).norm(dim=1)
            move_up = dist > self.curr.size_m * 0.5
            move_down = (dist < 0.5) & (~move_up)
            self.curr.update(ids, move_up, move_down)
            self._home[ids, 0:3] = self.curr.env_origins[ids]
            self._h_min[ids] = self.curr.env_origins[ids, 2] - H_MIN_DROP
        self._root[ids] = self._home[ids]
        self._dof[ids, :N_DOF, 0] = self._default_dof[ids] * (0.5 + torch.rand(n, N_DOF, device=dev))
        self._dof[ids, :N_DOF, 1] = (torch.rand(n, N_DOF, device=dev) - 0.5) * 0.2
        self.sim.set_root_state_tensor_indexed(self._root, ids)
        self.sim.set_dof_state_tensor_indexed(self._dof, ids)
        self.sim.set_dof_position_target_tensor_indexed(self._default_dof, ids)
        self._prev_action[ids] = 0.0
        self._last_action[ids] = 0.0
        # Reward-state trackers must restart with the episode (else stale air time / dof_acc
        # leak across the reset), and the per-term episode sums are logged-then-zeroed.
        self._prev_dof_vel[ids] = 0.0
        self._air_time[ids] = 0.0
        self._prev_in_contact[ids] = False
        self.rewards.reset(ids)
        self._sample_commands(ids)
        self.progress_buf[ids] = 0.0


# Live drive command (vx, vy, yaw), shared between the UI sliders and the task. When
# `task._drive` points at this, every env is commanded with it each step (windowed play).
_DRIVE = [1.0, 0.0, 0.0]


def _draw_drive(task):
    import lm.gui as gui
    gui.set_next_window_pos(gui.ImVec2(8, 8))
    gui.set_next_window_bg_alpha(0.7)
    gui.begin("Drive ANYmal (velocity command -> all robots)")
    ch, v = gui.slider_float("vx  fwd/back", _DRIVE[0], -2.0, 2.0)
    if ch: _DRIVE[0] = v
    ch, v = gui.slider_float("vy  strafe", _DRIVE[1], -1.0, 1.0)
    if ch: _DRIVE[1] = v
    ch, v = gui.slider_float("yaw  turn", _DRIVE[2], -1.0, 1.0)
    if ch: _DRIVE[2] = v
    lb = getattr(task, "_lin_body", None)
    if lb is not None:
        gui.text(f"achieved (mean): vx={float(lb[:, 0].mean()):+.2f}  vy={float(lb[:, 1].mean()):+.2f} m/s")
    gui.end()


def _frame_camera(task):
    v = task.runner
    if v is None or not hasattr(v, "set_camera"):
        return
    side = int(math.ceil(math.sqrt(task.num_envs)))
    ext = (side - 1) * ENV_SPACING
    c = ext / 2.0
    dist = ext * 0.9 + 8.0
    v.set_camera(pos=(c - dist * 0.5, c - dist * 0.7, dist * 0.6), target=(c, c, 0.3))


if __name__ == "__main__":
    play_ckpt = os.environ.get("LM_RL_PLAY")
    view = os.environ.get("LM_RL_VIEW") == "1"
    headless = os.environ.get("LM_RL_HEADLESS") == "1" or not (play_ckpt or view)
    trainer = os.environ.get("LM_RL_TRAINER", "rl_games")  # rl_games | rsl_rl | skrl
    task = AnymalTask(num_envs=NUM_ENVS, headless=headless)
    try:
        if not headless:
            _frame_camera(task)
            task._drive = _DRIVE   # live UI driving: the slider panel commands all robots
            if hasattr(task.runner, "set_ui_callback"):
                task.runner.set_ui_callback(lambda: _draw_drive(task))
        if play_ckpt:
            if trainer == "rsl_rl":
                rl.play_rsl_rl(task, play_ckpt)
            elif trainer == "skrl":
                rl.play_skrl(task, play_ckpt, headless=headless)
            else:
                games = int(os.environ.get("LM_RL_GAMES", "100000"))
                # Must rebuild the SAME network the checkpoint was trained with
                # ([256,128,64]) or the state_dict load mismatches — so play reuses
                # ANYMAL_PPO_PARAMS and just adds the player config.
                import copy
                pp = copy.deepcopy(ANYMAL_PPO_PARAMS)
                pp["params"]["config"]["player"] = {
                    "games_num": games, "deterministic": True, "render": False}
                rl.play_rl_games(task, play_ckpt, params=pp)
        elif trainer == "rsl_rl":
            # ANYMAL_PPO_PARAMS is rl_games-specific; rsl_rl/skrl use their own defaults.
            rl.train_rsl_rl(task, max_iterations=int(os.environ.get("LM_RL_EPOCHS", "1500")), seed=0)
        elif trainer == "skrl":
            rl.train_skrl(task, timesteps=int(os.environ.get("LM_RL_TIMESTEPS", "1000000")),
                          seed=0, headless=headless)
        else:
            rl.train_rl_games(task, max_epochs=int(os.environ.get("LM_RL_EPOCHS", "1500")), seed=0,
                              horizon_length=24, mini_epochs=5, params=ANYMAL_PPO_PARAMS)
    except BaseException:
        import traceback; print("[anymal-dbg] run raised:"); traceback.print_exc()
    finally:
        rl.destroy_world(task.sim, task.runner)
