"""Cartpole vectorized task on the USD world (the classic balance benchmark).

A `rl.VecTask` (the rl_games/rsl_rl/skrl adapters drive it): builds the USD world via
the `rl.World` facade (fixed-base, force-controlled), exposes reset()/step() over the
RlSim GPU tensors, and reports failure vs timeout so the trainer bootstraps correctly.

Run via the CLI:
    python train.py --task Cartpole
    python play.py  --task Cartpole --checkpoint runs/Cartpole_.../nn/Cartpole.pth
"""
import math
from pathlib import Path

from lumotion_envs._engine import ensure_engine
ensure_engine()
import lm.rl as rl
from lumotion_envs.config import CartpoleConfig

from lumotion_envs.assets import ASSETS
NUM_DOFS    = 2
ANGLE_LIMIT = math.pi / 2
CART_LIMIT  = 3.0
CLIP_OBS    = 5.0
MAX_EPISODE_LENGTH = 500


# Custom (local) terms — the managers work with task-specific term funcs, not just the
# shared lm.rl.observations/terminations libraries.
def _obs_cart_pos(task):   return task._dof[:, 0, 0]
def _obs_cart_vel(task):   return task._dof[:, 0, 1]
def _obs_pole_angle(task): return task._dof[:, 1, 0]
def _obs_pole_vel(task):   return task._dof[:, 1, 1]

OBS_TERMS = [
    rl.ObsTerm("cart_pos",   _obs_cart_pos),
    rl.ObsTerm("cart_vel",   _obs_cart_vel),
    rl.ObsTerm("pole_angle", _obs_pole_angle),
    rl.ObsTerm("pole_vel",   _obs_pole_vel),
]


def _term_out_of_bounds(task):
    d = task._dof
    return (d[:, 0, 0].abs() > CART_LIMIT) | (d[:, 1, 0].abs() > ANGLE_LIMIT)


def _reset_cartpole(task, env_ids):
    torch = task._torch
    n = env_ids.numel()
    # Reference cartpole reset: dof positions ~U(-0.1, 0.1), vels ~U(-0.25, 0.25).
    task._dof[env_ids, :, 0] = 0.2 * (torch.rand(n, NUM_DOFS, device=task.device) - 0.5)
    task._dof[env_ids, :, 1] = 0.5 * (torch.rand(n, NUM_DOFS, device=task.device) - 0.5)
    task.sim.set_dof_state_tensor_indexed(task._dof, env_ids)


class CartpoleTask(rl.VecTask):
    """Vectorized cartpole on the USD world (fixed-base slider + pole)."""

    def __init__(self, cfg: CartpoleConfig = None, *, num_envs=None, headless=None,
                 num_states=None, config=None):
        # Primary API: a CartpoleConfig. The keyword args are a back-compat path (used by
        # tests + direct construction); `config` overrides the internal SimConfig.
        self.cfg = cfg or CartpoleConfig()
        if num_envs is not None:
            self.cfg.num_envs = num_envs
        if headless is not None:
            self.cfg.headless = headless
        if num_states is not None:
            self.cfg.num_states = num_states
        robot_usd = self.cfg.robot_usd or str(ASSETS / "cartpole_converted" / "cartpole.usda")
        # Fixed-base, force-controlled: no ground, no PD prep (prep=False), DOFs by index.
        self.world = rl.World(num_envs=int(self.cfg.num_envs), env_spacing=self.cfg.env_spacing)
        self.world.add_robot(rl.Usd(robot_usd, prep=False, num_dofs=NUM_DOFS))
        sim, runner = self.world.build(
            headless=self.cfg.headless, config=config or rl.SimConfig(substeps=2, device="auto"),
            title="Cartpole (rl_games)")
        sim.play()
        super().__init__(sim, runner, num_obs=4, num_actions=1, num_states=int(self.cfg.num_states),
                         name="Cartpole", clip_obs=CLIP_OBS, max_episode_length=MAX_EPISODE_LENGTH,
                         seed=int(self.cfg.seed))
        self._dof = None

    # -- task hooks ---------------------------------------------------------

    def _capture(self):
        self._dof = self.sim.acquire_dof_state_tensor()
        self.observations = rl.ObservationManager(self, OBS_TERMS)
        self.terminations = rl.TerminationManager(self, [
            rl.TermTerm("out_of_bounds", _term_out_of_bounds)])
        self.events = rl.EventManager(self, [
            rl.EventTerm("reset", _reset_cartpole, mode="reset")])

    def _pre_physics_step(self, actions):
        forces = self._torch.zeros(self.num_envs, NUM_DOFS, device=self.device)
        forces[:, 0] = actions.reshape(-1) * self.cfg.force_mag   # action already clipped by the base
        self.sim.set_dof_actuation_force_tensor(forces)

    def _compute_observations(self):
        self.obs_buf[:] = self.observations.compute()

    def _compute_reward(self):
        torch = self._torch
        o = self.obs_buf
        cart_vel, pole_angle, pole_vel = o[:, 1], o[:, 2], o[:, 3]
        reward = 1.0 - pole_angle * pole_angle - 0.01 * cart_vel.abs() - 0.005 * pole_vel.abs()
        self.reset_buf = self.terminations.compute()
        self.rew_buf = torch.where(self.reset_buf.bool(), torch.full_like(reward, -2.0), reward)

    def _reset_idx(self, ids):
        self.events.reset(ids)
        self.progress_buf[ids] = 0.0


def _frame_camera(task):
    """Point the windowed viewer at the env grid (Z-up; envs spread on X-Y)."""
    v = task.runner
    if v is None or not hasattr(v, "set_camera"):
        return
    if task.num_envs == 1:
        v.set_camera(pos=(3.2, 0.0, 0.55), target=(0.0, 0.0, 0.55))
        return
    side = int(math.ceil(math.sqrt(task.num_envs)))
    ext = (side - 1) * task.cfg.env_spacing
    cx = cy = ext / 2.0
    dist = ext * 0.9 + 8.0
    v.set_camera(pos=(cx - dist * 0.5, cy - dist * 0.7, dist * 0.7), target=(cx, cy, 0.6))
