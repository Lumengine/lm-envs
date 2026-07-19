"""ANYmal-C velocity-command locomotion — a thin wrapper over the generic
LeggedVelocityTask (tasks/legged_velocity.py) bound to the AnymalConfig. The actual
task logic + reward/obs/termination/command/event terms live in legged_velocity; a
new quadruped is a config, not a new task body.

    python train.py --task Anymal
"""
from pathlib import Path

from lumengine_envs.tasks.legged_velocity import LeggedVelocityTask, LEGGED_PPO_PARAMS, _frame_camera  # noqa: F401
from lumengine_envs.config import AnymalConfig

# Back-compat module constants (used by tests that author worlds directly).
N_DOF = 12
ENV_SPACING = 4.0
ANYMAL_PPO_PARAMS = LEGGED_PPO_PARAMS


class AnymalTask(LeggedVelocityTask):
    """LeggedVelocityTask with the ANYmal-C config."""

    def __init__(self, cfg=None, *, num_envs=None, headless=None):
        super().__init__(cfg or AnymalConfig(), num_envs=num_envs, headless=headless)
