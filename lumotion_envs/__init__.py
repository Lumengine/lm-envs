"""Lumotion — reinforcement-learning environments for the Lumengine engine.

The engine ships the `lm.rl` facade (RlSim tensors, the World authoring layer, and
the rl_games / rsl_rl / skrl trainers); this package holds the TASKS, robot assets
and a small task registry + CLI that use it.

Public surface:
    from lumotion_envs.registry import REGISTRY, load_task

Train / play from the repo root:
    python train.py --task Anymal           # headless train
    python train.py --list                  # the task catalog
    python play.py  --task Anymal --checkpoint runs/.../nn/Anymal.pth
"""

from .registry import REGISTRY, TaskSpec, load_task

__all__ = ["REGISTRY", "TaskSpec", "load_task"]
