"""Task registry — the single catalog of LumengineEnvs tasks, with the metadata the
`train.py` / `play.py` CLIs need to build and train each one uniformly.

A `TaskSpec` is pure data (no engine import), so listing the catalog (`--list`) works
without CUDA or the engine. `load_task()` lazily imports the task module on demand and
returns its class + optional rl_games PPO params (referenced by attribute name so this
file stays import-light).

Migration note: tasks currently live as top-level modules under `tasks/` (e.g.
`anymal_task`). The CLIs put `tasks/` on `sys.path`. As tasks move into
`lumengine_envs/tasks/<domain>/`, only the `module` field here changes.
"""

import importlib
from dataclasses import dataclass, field

from .config import AntConfig, AnymalConfig, CartpoleConfig


@dataclass(frozen=True)
class TaskSpec:
    id: str                         # catalog name, e.g. "Anymal"
    module: str                     # importable module providing the task class
    cls: str                        # task class name in that module
    config_cls: type                # the (engine-free) config dataclass for this task
    domain: str                     # "classic" | "locomotion" | "manipulation" | "hands" | "aerial"
    desc: str                       # one-line human description
    default_envs: int               # num_envs when --num-envs is omitted
    max_epochs: int                 # rl_games epochs when --epochs is omitted
    train_kwargs: dict = field(default_factory=dict)   # extra train_rl_games kwargs (horizon_length, mini_epochs)
    ppo_attr: str | None = None     # module attribute holding the rl_games PPO params dict, or None for the default config


# The catalog. Keep ids stable — they are the public CLI handle and the baselines/ key.
REGISTRY: dict[str, TaskSpec] = {
    "Cartpole": TaskSpec(
        id="Cartpole", module="cartpole_task", cls="CartpoleTask", config_cls=CartpoleConfig,
        domain="classic", desc="Balance a pole on a force-controlled cart (fixed base).",
        default_envs=512, max_epochs=200),

    "Ant": TaskSpec(
        id="Ant", module="ant_task", cls="AntTask", config_cls=AntConfig,
        domain="locomotion", desc="MuJoCo ant (MJCF import) - run forward, torque control.",
        default_envs=4096, max_epochs=500,
        train_kwargs={"horizon_length": 24, "mini_epochs": 5}, ppo_attr="ANT_PPO_PARAMS"),

    "Anymal": TaskSpec(
        id="Anymal", module="anymal_task", cls="AnymalTask", config_cls=AnymalConfig,
        domain="locomotion", desc="ANYmal-C (URDF) - velocity-command walking (IsaacLab reward).",
        default_envs=4096, max_epochs=1500,
        train_kwargs={"horizon_length": 24, "mini_epochs": 5}, ppo_attr="ANYMAL_PPO_PARAMS"),
}


def load_task(spec: TaskSpec):
    """Import the task module and resolve (module, task_class, ppo_params). Triggers the
    engine bootstrap that the task module performs at import — call only after the caller
    has set LUMENGINE_ROOT (the CLIs do)."""
    module = importlib.import_module(spec.module)
    cls = getattr(module, spec.cls)
    ppo = getattr(module, spec.ppo_attr) if spec.ppo_attr else None
    return module, cls, ppo
