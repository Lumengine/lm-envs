"""ANYmal velocity-command locomotion on the USD world, trained by rl_games.

Production path (mirrors Samples/RlCartpoleUsd/cartpole_task.py): a VecTask-shaped
task the rl_games adapter (lm.rl/_rlgames.py) drives. The robot is the REAL anymal_c
(anymal.urdf -> USD via urdf_usd_converter, prepped to floating base + PD drives by
prep_anymal_usd.py). Task = IsaacGymEnvs Anymal: track a commanded base velocity
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


sys.path.insert(0, str(Path(__file__).resolve().parent))   # _bootstrap + prep_anymal_usd
import _bootstrap
_bootstrap.bootstrap()
import lm.rl as rl
import prep_anymal_usd

faulthandler.enable()

_ROBOT = _bootstrap.ASSETS / "anymal_converted" / "anymal.usda"
_WORLD = _bootstrap.ASSETS / "world_anymal_train.usd"   # generated (gitignored)

NUM_ENVS    = int(os.environ.get("LM_RL_NUM_ENVS", "256"))
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
CMD_X, CMD_Y, CMD_YAW = (-2.0, 2.0), (-1.0, 1.0), (-1.0, 1.0)
H_MIN_FRAC, UPRIGHT_MIN = 0.55, 0.6
MAX_EPISODE_LENGTH = 1000
# Net contact force on the base link (articulation link 0) above this => the trunk is on
# the ground = a fall (complements the height/upright test with a real contact signal).
BASE_CONTACT_FAIL_N = float(os.environ.get("LM_RL_BASE_CONTACT_FAIL_N", "1.0"))

# IsaacGymEnvs AnymalPPO.yaml training config, deep-merged into rl.train's base. The
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
    },
}}


def _quat_rotate_inv(q, v):
    import torch
    qv, qw = -q[:, 0:3], q[:, 3:4]
    t = 2.0 * torch.cross(qv, v, dim=1)
    return v + qw * t + torch.cross(qv, t, dim=1)


class AnymalTask(rl.VecTask):
    """ANYmal velocity-command locomotion on rl.VecTask: the base owns the step/reset
    loop; the hooks below define the obs/reward/reset and the PD-target action."""

    def __init__(self, num_envs=NUM_ENVS, headless=True):
        prep_anymal_usd.main()   # floating base + PD drives (idempotent)
        # instanceable=True: share ONE composed robot prototype across all envs instead of
        # expanding the 60-link subtree per env. At 2048 envs that is the difference between a
        # ~530k-prim stage (~20 GB RAM) and a ~2k-prim one. Anymal uses the default physics
        # material, so per-env friction DR still works (build-time per-articulation un-sharing);
        # the per-env collision id + root pose live on the instance prim, so they stay per-env.
        rl.author_world(_ROBOT, _WORLD, num_envs=int(num_envs), spacing=ENV_SPACING,
                        ground=True, ground_z=GROUND_Z, spawn_z=0.0, instanceable=True)
        # Scale the GPU contact-pair buffer with env count so large-N runs (1k-4k) don't
        # overflow PhysX's default ~1M contact cap (12-DOF legged robot on a ground plane).
        sim, runner = rl.create_world(
            _WORLD, num_envs=int(num_envs), dofs_per_actor=N_DOF,
            config=rl.SimConfig(substeps=SUBSTEPS, device="auto",
                                gpu_contact_buffer_multiplier=max(1.0, int(num_envs) / 512.0)),
            headless=headless, title="ANYmal (rl_games, vel-cmd)")
        sim.play()
        super().__init__(sim, runner, num_obs=48, num_actions=N_DOF, name="Anymal",
                         clip_obs=CLIP_OBS, max_episode_length=MAX_EPISODE_LENGTH,
                         seed=int(os.environ.get("LM_RL_SEED", "0")))
        self._nstep = 0
        self._drive = None   # set to the _DRIVE list (windowed play) -> live UI command

    # -- task hooks ---------------------------------------------------------

    def _capture(self):
        torch = self._torch; dev = self.device
        self._dof = self.sim.acquire_dof_state_tensor()
        self._root = self.sim.acquire_root_state_tensor()
        self._contact = self.sim.acquire_link_net_contact_force_tensor()   # (envs, links, 3)
        self.sim.refresh_dof_state_tensor(); self.sim.refresh_root_state_tensor()
        # PD-held stance + stand height are the references (DOF order opaque in the
        # 60-link tree; the settled state IS the default, in batch order).
        self._default_dof = self._dof[:, :N_DOF, 0].clone()
        self._world_down = torch.tensor([0.0, 0.0, -1.0], device=dev).repeat(self.num_envs, 1)
        self._cmd_scale = torch.tensor([LIN_VEL_SCALE, LIN_VEL_SCALE, ANG_VEL_SCALE], device=dev)
        self._prev_action = torch.zeros(self.num_envs, N_DOF, device=dev)
        self._commands = torch.zeros(self.num_envs, 3, device=dev)
        stand_h = float(self._root[:, 2].mean())
        self._h_min = stand_h * H_MIN_FRAC
        home = torch.zeros(self.num_envs, 13, device=dev)
        home[:, 0:2] = self._root[:, 0:2]
        home[:, 2] = stand_h
        home[:, 6] = 1.0
        self._home = home

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
        self._prev_action = actions
        targets = self._default_dof + ACTION_SCALE * actions
        self.sim.set_dof_position_target_tensor(targets)

    def _compute_observations(self):
        self.sim.refresh_root_state_tensor()
        root, dof = self._root, self._dof
        quat = root[:, 3:7]
        self._lin_body = _quat_rotate_inv(quat, root[:, 7:10])
        self._ang_body = _quat_rotate_inv(quat, root[:, 10:13])
        proj_grav = _quat_rotate_inv(quat, self._world_down)
        self._up_proj = -proj_grav[:, 2]
        o = self.obs_buf
        o[:, 0:3] = self._lin_body * LIN_VEL_SCALE
        o[:, 3:6] = self._ang_body * ANG_VEL_SCALE
        o[:, 6:9] = proj_grav
        o[:, 9:12] = self._commands * self._cmd_scale
        o[:, 12:24] = dof[:, :N_DOF, 0] - self._default_dof
        o[:, 24:36] = dof[:, :N_DOF, 1] * DOF_VEL_SCALE
        o[:, 36:48] = self._prev_action

    def _compute_reward(self):
        torch = self._torch
        lin_err = torch.sum((self._commands[:, :2] - self._lin_body[:, :2]) ** 2, dim=1)
        ang_err = (self._commands[:, 2] - self._ang_body[:, 2]) ** 2
        self.rew_buf = torch.clamp(
            torch.exp(-lin_err / TRACK_SIGMA) * REW_LIN_XY
            + torch.exp(-ang_err / TRACK_SIGMA) * REW_ANG_Z, min=0.0)

        height = self._root[:, 2]
        # Real foot/trunk contact: base link (articulation link 0) on the ground = a fall.
        self.sim.refresh_link_net_contact_force_tensor()
        base_contact = self._contact[:, 0, :].norm(dim=1)
        fail = (height < self._h_min) | (self._up_proj < UPRIGHT_MIN) | (base_contact > BASE_CONTACT_FAIL_N)
        self.reset_buf = fail.float()

        self._nstep += 1
        if self._nstep % 500 == 0:
            cmd = self._commands.mean(0)
            print(f"[anymal-dbg] step {self._nstep} | ep_len(mean)={float(self.progress_buf.mean()):.1f} "
                  f"| track_rew={float(self.rew_buf.mean()):.3f} | vx={float(self._lin_body[:,0].mean()):+.2f} "
                  f"(cmd_x~{float(cmd[0]):+.2f}) | upright={float(self._up_proj.mean()):.2f} "
                  f"| height={float(height.mean()):.2f} | base_contact>{BASE_CONTACT_FAIL_N:g}N="
                  f"{int((base_contact > BASE_CONTACT_FAIL_N).sum())} | fails={int(fail.sum())}", flush=True)

    def _reset_idx(self, ids):
        torch = self._torch; n = ids.numel(); dev = self.device
        self._root[ids] = self._home[ids]
        self._dof[ids, :N_DOF, 0] = self._default_dof[ids] * (0.5 + torch.rand(n, N_DOF, device=dev))
        self._dof[ids, :N_DOF, 1] = (torch.rand(n, N_DOF, device=dev) - 0.5) * 0.2
        self.sim.set_root_state_tensor_indexed(self._root, ids)
        self.sim.set_dof_state_tensor_indexed(self._dof, ids)
        self.sim.set_dof_position_target_tensor_indexed(self._default_dof, ids)
        self._prev_action[ids] = 0.0
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
    task = AnymalTask(num_envs=NUM_ENVS, headless=headless)
    try:
        if not headless:
            _frame_camera(task)
            task._drive = _DRIVE   # live UI driving: the slider panel commands all robots
            if hasattr(task.runner, "set_ui_callback"):
                task.runner.set_ui_callback(lambda: _draw_drive(task))
        if play_ckpt:
            games = int(os.environ.get("LM_RL_GAMES", "100000"))
            # Must rebuild the SAME network the checkpoint was trained with
            # ([256,128,64]) or the state_dict load mismatches — so play reuses
            # ANYMAL_PPO_PARAMS and just adds the player config.
            import copy
            pp = copy.deepcopy(ANYMAL_PPO_PARAMS)
            pp["params"]["config"]["player"] = {
                "games_num": games, "deterministic": True, "render": False}
            rl.play(task, play_ckpt, params=pp)
        else:
            rl.train(task, max_epochs=int(os.environ.get("LM_RL_EPOCHS", "1500")), seed=0,
                     horizon_length=24, mini_epochs=5, params=ANYMAL_PPO_PARAMS)
    except BaseException:
        import traceback; print("[anymal-dbg] run raised:"); traceback.print_exc()
    finally:
        rl.destroy_world(task.sim, task.runner)
