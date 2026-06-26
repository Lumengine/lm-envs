"""Cartpole vectorized task on the USD world, trained by rl_games (Phase 0 spike).

A `VecTask`-shaped task (the adapter in lm.rl/_rlgames.py wraps it for rl_games):
builds the USD world via rl.author_world + rl.create_world, exposes reset()/step()
over the RlSim GPU tensors, and reports failure vs timeout so the trainer bootstraps
correctly. Run headless:

    set LM_PHYSX_SHARE_CUDA_CONTEXT=1
    python Samples/RlCartpoleUsd/cartpole_task.py
"""

import faulthandler
import math
import os
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))   # for _bootstrap
import _bootstrap
_bootstrap.bootstrap()
import lm.rl as rl

faulthandler.enable()

_ROBOT_USD  = Path(os.environ.get(
    "LM_RL_ROBOT_USD", str(_bootstrap.ASSETS / "cartpole_converted" / "cartpole.usda")))
_WORLD_USD  = _bootstrap.ASSETS / "world.usd"   # generated (gitignored)

NUM_ENVS    = int(os.environ.get("LM_RL_NUM_ENVS", "512"))
# Envs share ONE PhysicsScene but never collide across cells: author_world marks each
# env and the physicsEngine ingest gives every body an env partition id, so the filter
# shader suppresses cross-env contacts (incl. on the direct-GPU articulation path).
# Spacing is therefore purely cosmetic (grid layout) — 4 keeps parity with the
# reference env without any risk of inter-env coupling.
ENV_SPACING = float(os.environ.get("LM_RL_SPACING", "4.0"))
NUM_DOFS    = 2
FORCE_MAG   = float(os.environ.get("LM_RL_FORCE", "400.0"))   # max push effort (action in [-1,1] -> [-FORCE,FORCE] N)
ANGLE_LIMIT = math.pi / 2
CART_LIMIT  = 3.0
CLIP_OBS    = 5.0
MAX_EPISODE_LENGTH = 500


class CartpoleTask(rl.VecTask):
    """Vectorized cartpole on the USD world (fixed-base slider + pole), on rl.VecTask:
    the base owns the step/reset loop; this fills the five task hooks."""

    def __init__(self, num_envs=NUM_ENVS, headless=True, num_states=0, config=None):
        rl.author_world(_ROBOT_USD, _WORLD_USD, num_envs=int(num_envs), spacing=ENV_SPACING)
        sim, runner = rl.create_world(
            _WORLD_USD, num_envs=int(num_envs), dofs_per_actor=NUM_DOFS,
            config=config or rl.SimConfig(substeps=2, device="auto"),
            headless=headless, title="Cartpole (rl_games)")
        sim.play()
        super().__init__(sim, runner, num_obs=4, num_actions=1, num_states=int(num_states),
                         name="Cartpole", clip_obs=CLIP_OBS, max_episode_length=MAX_EPISODE_LENGTH,
                         seed=int(os.environ.get("LM_RL_SEED", "0")))
        self._dof = None

    # -- task hooks ---------------------------------------------------------

    def _capture(self):
        self._dof = self.sim.acquire_dof_state_tensor()

    def _pre_physics_step(self, actions):
        forces = self._torch.zeros(self.num_envs, NUM_DOFS, device=self.device)
        forces[:, 0] = actions.reshape(-1) * FORCE_MAG     # action already clipped by the base
        self.sim.set_dof_actuation_force_tensor(forces)

    def _compute_observations(self):
        d, o = self._dof, self.obs_buf
        o[:, 0] = d[:, 0, 0]   # cart_pos
        o[:, 1] = d[:, 0, 1]   # cart_vel
        o[:, 2] = d[:, 1, 0]   # pole_angle
        o[:, 3] = d[:, 1, 1]   # pole_vel

    def _compute_reward(self):
        torch = self._torch
        o = self.obs_buf
        cart_pos, cart_vel, pole_angle, pole_vel = o[:, 0], o[:, 1], o[:, 2], o[:, 3]
        reward = 1.0 - pole_angle * pole_angle - 0.01 * cart_vel.abs() - 0.005 * pole_vel.abs()
        fail = (cart_pos.abs() > CART_LIMIT) | (pole_angle.abs() > ANGLE_LIMIT)
        self.rew_buf = torch.where(fail, torch.full_like(reward, -2.0), reward)
        self.reset_buf = fail.float()

    def _reset_idx(self, ids):
        torch = self._torch
        n = ids.numel()
        # Reference cartpole reset: dof positions ~U(-0.1, 0.1), vels ~U(-0.25, 0.25).
        self._dof[ids, :, 0] = 0.2 * (torch.rand(n, NUM_DOFS, device=self.device) - 0.5)
        self._dof[ids, :, 1] = 0.5 * (torch.rand(n, NUM_DOFS, device=self.device) - 0.5)
        self.sim.set_dof_state_tensor_indexed(self._dof, ids)
        self.progress_buf[ids] = 0.0


def _diag_dof_mapping(task):
    """Env-correctness test via the task's REAL reset()/step() (exactly what rl_games
    drives), after a proper warmup. Checks: reset applies, the action controls the cart,
    and which obs component responds (DOF order / orientation)."""
    from lm.rl._rlgames import _warmup
    torch = task._torch
    _warmup(task)

    # Physics sanity: uncontrolled fall time. Place the pole at 0.10 rad, zero
    # velocity, ZERO force, and count steps until it crosses the +/-90 deg limit.
    # A correctly-massed ~1 m pole (I~0.30 about the pivot) takes ~50 steps; a
    # broken/low inertia (wrong COM, fallback collider) topples in ~10.
    task.reset()
    task._dof[:, :, 0] = 0.0
    task._dof[:, :, 1] = 0.0
    task._dof[:, 1, 0] = 0.10          # pole angle = 0.10 rad, everything else at rest
    task.sim.set_dof_state_tensor(task._dof)
    task.sim.simulate(); task.sim.fetch_results(); task.sim.refresh_dof_state_tensor()
    # Raw step (NOT task.step) so the auto-reset can't contaminate the fall — just
    # zero force + simulate + read the unclamped pole angle until it crosses 90 deg.
    zero_forces = torch.zeros(task.num_envs, NUM_DOFS, device=task.device)
    fall = None
    for i in range(1, 401):
        task.sim.set_dof_actuation_force_tensor(zero_forces)
        task.sim.simulate(); task.sim.fetch_results(); task.sim.refresh_dof_state_tensor()
        ang = float(task._dof[0, 1, 0])   # pole angle, unclamped, no reset
        vel = float(task._dof[0, 1, 1])
        if i <= 60:
            print(f"[diag-fall] step {i:>3}: pole_ang={ang:+.3f} rad ({ang*57.3:+.0f} deg) vel={vel:+.3f}")
        if abs(ang) >= ANGLE_LIMIT - 1e-3:
            fall = i
            print(f"[diag-fall] >>> crossed +/-90 deg at step {i} "
                  f"(correct ~50, broken/too-small inertia ~10)")
            break
    if fall is None:
        print("[diag-fall] pole did NOT cross 90 deg in 400 steps (over-damped / COM at pivot?)")

    obs = task.reset()
    print(f"[diag] after reset: cart_pos(obs0)={float(obs[0,0]):+.3f} cart_vel(obs1)={float(obs[0,1]):+.3f} "
          f"pole_ang(obs2)={float(obs[0,2]):+.3f} pole_vel(obs3)={float(obs[0,3]):+.3f}  "
          f"(reset should give all ~small)")
    for sign in (+1.0, -1.0):
        obs = task.reset()
        print(f"[diag] --- constant action={sign:+.0f} (force {sign*FORCE_MAG:+.0f} N on dof0) ---")
        for i in range(1, 31):
            a = torch.full((task.num_envs, 1), sign, device=task.device)
            obs, rew, done, ex = task.step(a)
            if i % 6 == 0:
                print(f"[diag]  step {i:>2}: cart_pos={float(obs[0,0]):+.3f} cart_vel={float(obs[0,1]):+.3f} "
                      f"pole_ang={float(obs[0,2]):+.3f} pole_vel={float(obs[0,3]):+.3f} done={int(done[0])}")
    print("[diag] -> +force should drive cart_pos monotonically one way (cart slides); "
          "if instead pole_ang runs away with ~0 cart_pos, the DOF order is swapped.")


def _frame_camera(task):
    """Point the windowed viewer at the env grid (Z-up; envs spread on X-Y)."""
    v = task.runner
    if v is None or not hasattr(v, "set_camera"):
        return
    if task.num_envs == 1:
        # Cart slides along Y, pole pivots about X: look down +X at the Y-Z plane.
        v.set_camera(pos=(3.2, 0.0, 0.55), target=(0.0, 0.0, 0.55))
        return
    side = int(math.ceil(math.sqrt(task.num_envs)))
    ext = (side - 1) * ENV_SPACING
    cx = cy = ext / 2.0
    dist = ext * 0.9 + 8.0
    v.set_camera(pos=(cx - dist * 0.5, cy - dist * 0.7, dist * 0.7), target=(cx, cy, 0.6))


if __name__ == "__main__":
    play_ckpt = os.environ.get("LM_RL_PLAY")          # path to a .pth -> watch it windowed
    diag = os.environ.get("LM_RL_DIAG")
    view = os.environ.get("LM_RL_VIEW") == "1"        # windowed: watch training live
    trainer = os.environ.get("LM_RL_TRAINER", "rl_games")  # rl_games | rsl_rl | skrl
    force_headless = os.environ.get("LM_RL_HEADLESS") == "1"
    headless = force_headless or not (play_ckpt or view)
    task = CartpoleTask(num_envs=NUM_ENVS, headless=headless)

    if diag:
        try:
            _diag_dof_mapping(task)
        finally:
            rl.destroy_world(task.sim, task.runner)
        import sys as _s; _s.exit(0)

    try:
        if not headless:
            _frame_camera(task)
        if play_ckpt:
            if trainer == "rsl_rl":
                rl.play_rsl_rl(task, play_ckpt)
            elif trainer == "skrl":
                rl.play_skrl(task, play_ckpt, headless=headless)
            else:
                # Deterministic (no action noise) and keep playing many episodes so the
                # window stays up to watch; closing the window stops it gracefully.
                games_num = int(os.environ.get("LM_RL_GAMES", "100000"))
                deterministic = os.environ.get("LM_RL_DETERMINISTIC", "1") != "0"
                rl.play_rl_games(task, play_ckpt, params={"params": {"config": {
                    "player": {"games_num": games_num, "deterministic": deterministic, "render": False}}}})
        elif trainer == "rsl_rl":
            rl.train_rsl_rl(task, max_iterations=int(os.environ.get("LM_RL_EPOCHS", "200")), seed=0)
        elif trainer == "skrl":
            rl.train_skrl(task, timesteps=int(os.environ.get("LM_RL_TIMESTEPS", "100000")),
                          seed=0, headless=headless)
        else:
            rl.train_rl_games(task, max_epochs=int(os.environ.get("LM_RL_EPOCHS", "200")), seed=0)
    except BaseException:
        import traceback
        print("[task-dbg] run raised:")
        traceback.print_exc()
    finally:
        rl.destroy_world(task.sim, task.runner)
