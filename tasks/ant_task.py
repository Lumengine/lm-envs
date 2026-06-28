"""ANT — a velocity-tracking locomotion task on the MuJoCo ant, IMPORTED FROM MJCF.

Demonstrates the rl.Mjcf import path end-to-end inside a real lm.rl task: ant.xml (MJCF)
-> mujoco-usd-converter -> UsdPhysics -> config-driven prep (assets/ant.rl.yaml: floating
base + PD DriveAPI on the 8 leg joints) -> ingest -> RL. Same VecTask scaffolding as
anymal_task, sized for the 8-DOF ant.

    LM_RL_VIEW=1 python ant_task.py                 # windowed: watch it train live
    python ant_task.py                              # headless train (rl_games)
    LM_RL_PLAY=runs/.../nn/Ant.pth python ant_task.py   # watch a trained policy
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
_CFG = _bootstrap.ASSETS / "ant.rl.yaml"           # floating base + PD drives (prep config)

NUM_ENVS    = int(os.environ.get("LM_RL_NUM_ENVS", "256"))
ENV_SPACING = float(os.environ.get("LM_RL_SPACING", "2.5"))
N_DOF       = 8
GROUND_Z    = 0.0
SPAWN_Z     = float(os.environ.get("LM_RL_ANT_SPAWN_Z", "0.55"))
SUBSTEPS    = 2

LIN_VEL_SCALE, ANG_VEL_SCALE = 2.0, 0.25
DOF_VEL_SCALE = 0.05
ACTION_SCALE  = 0.5
CLIP_OBS      = 5.0
TRACK_SIGMA   = 0.25
REW_LIN_XY, REW_ANG_Z = 1.0, 0.5
CMD_X, CMD_Y, CMD_YAW = (-1.5, 1.5), (-0.5, 0.5), (-1.0, 1.0)
H_MIN_FRAC, UPRIGHT_MIN = 0.5, 0.5
MAX_EPISODE_LENGTH = 1000
NUM_OBS = 36

ANT_PPO_PARAMS = {"params": {
    "network": {"mlp": {"units": [128, 64, 32]}},
    "config": {"reward_shaper": {"scale_value": 1.0}, "critic_coef": 2, "bounds_loss_coef": 0.001},
}}


def _quat_rotate_inv(q, v):
    import torch
    w = q[:, 3:4]; xyz = q[:, 0:3]
    t = 2.0 * torch.cross(xyz, v, dim=1)
    return v - w * t + torch.cross(xyz, t, dim=1)


class AntTask(rl.VecTask):
    """Track a commanded base velocity (vx, vy, yaw) on the MJCF-imported ant."""

    def __init__(self, num_envs=NUM_ENVS, headless=True):
        self.world = rl.World(num_envs=int(num_envs), env_spacing=ENV_SPACING)
        self.world.add_ground(z=GROUND_Z, friction=1.0)
        self.robot = self.world.add_robot(rl.Mjcf(str(_ANT), config=str(_CFG)), spawn_z=SPAWN_Z)
        sim, runner = self.world.build(
            headless=headless,
            config=rl.SimConfig(substeps=SUBSTEPS, device="auto",
                                gpu_contact_buffer_multiplier=max(1.0, int(num_envs) / 512.0)),
            title="Ant (MJCF, vel-cmd)")
        sim.play()
        super().__init__(sim, runner, num_obs=NUM_OBS, num_actions=N_DOF, name="Ant",
                         clip_obs=CLIP_OBS, max_episode_length=MAX_EPISODE_LENGTH,
                         seed=int(os.environ.get("LM_RL_SEED", "0")))
        self._nstep = 0
        self._drive = None

    # -- task hooks ---------------------------------------------------------

    def _capture(self):
        torch = self._torch; dev = self.device
        self._dof = self.sim.acquire_dof_state_tensor()
        self._root = self.sim.acquire_root_state_tensor()
        self._contact = self.sim.acquire_link_net_contact_force_tensor()
        self.sim.refresh_dof_state_tensor(); self.sim.refresh_root_state_tensor()
        self._default_dof = self.robot.default_dof_positions.unsqueeze(0).repeat(self.num_envs, 1)
        self._world_down = torch.tensor([0.0, 0.0, -1.0], device=dev).repeat(self.num_envs, 1)
        self._cmd_scale = torch.tensor([LIN_VEL_SCALE, LIN_VEL_SCALE, ANG_VEL_SCALE], device=dev)
        self._prev_action = torch.zeros(self.num_envs, N_DOF, device=dev)
        self._commands = torch.zeros(self.num_envs, 3, device=dev)
        self._h_min = self._root[:, 2] * H_MIN_FRAC
        home = torch.zeros(self.num_envs, 13, device=dev)
        home[:, 0:3] = self._root[:, 0:3]
        home[:, 6] = 1.0
        self._home = home

    def _sample_commands(self, ids):
        torch = self._torch; dev = self.device; n = ids.numel()
        if self._drive is not None:
            self._commands[ids, 0], self._commands[ids, 1], self._commands[ids, 2] = self._drive
            return
        fixed = os.environ.get("LM_RL_CMD")
        if fixed:
            c = [float(x) for x in fixed.split(",")]
            self._commands[ids, 0], self._commands[ids, 1], self._commands[ids, 2] = c[0], c[1], c[2]
            return
        self._commands[ids, 0] = torch.empty(n, device=dev).uniform_(*CMD_X)
        self._commands[ids, 1] = torch.empty(n, device=dev).uniform_(*CMD_Y)
        self._commands[ids, 2] = torch.empty(n, device=dev).uniform_(*CMD_YAW)

    def _pre_physics_step(self, actions):
        if self._drive is not None:
            self._commands[:, 0], self._commands[:, 1], self._commands[:, 2] = self._drive
        self._prev_action = actions
        self.sim.set_dof_position_target_tensor(self._default_dof + ACTION_SCALE * actions)

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
        o[:, 12:20] = dof[:, :N_DOF, 0] - self._default_dof
        o[:, 20:28] = dof[:, :N_DOF, 1] * DOF_VEL_SCALE
        o[:, 28:36] = self._prev_action

    def _compute_reward(self):
        torch = self._torch
        lin_err = torch.sum((self._commands[:, :2] - self._lin_body[:, :2]) ** 2, dim=1)
        ang_err = (self._commands[:, 2] - self._ang_body[:, 2]) ** 2
        self.rew_buf = torch.clamp(
            torch.exp(-lin_err / TRACK_SIGMA) * REW_LIN_XY
            + torch.exp(-ang_err / TRACK_SIGMA) * REW_ANG_Z, min=0.0)

        height = self._root[:, 2]
        self.sim.refresh_link_net_contact_force_tensor()
        base_contact = self._contact[:, 0, :].norm(dim=1)
        fail = (height < self._h_min) | (self._up_proj < UPRIGHT_MIN)
        self.reset_buf = fail.float()

        self._nstep += 1
        if self._nstep % 500 == 0:
            print(f"[ant-dbg] step {self._nstep} | ep_len(mean)={float(self.progress_buf.mean()):.1f} "
                  f"| track_rew={float(self.rew_buf.mean()):.3f} | vx={float(self._lin_body[:,0].mean()):+.2f} "
                  f"(cmd_x~{float(self._commands[:,0].mean()):+.2f}) | upright={float(self._up_proj.mean()):.2f} "
                  f"| height={float(height.mean()):.2f} | fails={int(fail.sum())}", flush=True)

    def _reset_idx(self, ids):
        torch = self._torch; n = ids.numel(); dev = self.device
        self._root[ids] = self._home[ids]
        self._dof[ids, :N_DOF, 0] = self._default_dof[ids] + (torch.rand(n, N_DOF, device=dev) - 0.5) * 0.1
        self._dof[ids, :N_DOF, 1] = (torch.rand(n, N_DOF, device=dev) - 0.5) * 0.2
        self.sim.set_root_state_tensor_indexed(self._root, ids)
        self.sim.set_dof_state_tensor_indexed(self._dof, ids)
        self.sim.set_dof_position_target_tensor_indexed(self._default_dof, ids)
        self._prev_action[ids] = 0.0
        self._sample_commands(ids)
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
            task._drive = [1.0, 0.0, 0.0]   # walk forward in the windowed view
        if play_ckpt:
            import copy
            pp = copy.deepcopy(ANT_PPO_PARAMS)
            pp["params"]["config"]["player"] = {
                "games_num": int(os.environ.get("LM_RL_GAMES", "100000")),
                "deterministic": True, "render": False}
            rl.play_rl_games(task, play_ckpt, params=pp)
        else:
            rl.train_rl_games(task, max_epochs=int(os.environ.get("LM_RL_EPOCHS", "1000")), seed=0,
                              horizon_length=24, mini_epochs=5, params=ANT_PPO_PARAMS)
    except BaseException:
        import traceback; print("[ant-dbg] run raised:"); traceback.print_exc()
    finally:
        rl.destroy_world(task.sim, task.runner)
