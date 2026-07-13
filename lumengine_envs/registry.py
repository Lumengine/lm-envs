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

from .config import (A1Config, AntConfig, AnymalConfig, CartpoleConfig, FrankaLiftConfig,
                     FrankaReachConfig, Go1Config, Go2Config, H1Config)


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

    "Go2": TaskSpec(
        id="Go2", module="legged_velocity", cls="LeggedVelocityTask", config_cls=Go2Config,
        domain="locomotion", desc="Unitree Go2 (URDF) - velocity-command walking (shared legged task).",
        default_envs=4096, max_epochs=1000,
        train_kwargs={"horizon_length": 24, "mini_epochs": 5}, ppo_attr="LEGGED_PPO_PARAMS"),

    "Go1": TaskSpec(
        id="Go1", module="legged_velocity", cls="LeggedVelocityTask", config_cls=Go1Config,
        domain="locomotion", desc="Unitree Go1 (MJCF, Menagerie) - velocity-command walking.",
        default_envs=4096, max_epochs=1000,
        train_kwargs={"horizon_length": 24, "mini_epochs": 5}, ppo_attr="LEGGED_PPO_PARAMS"),

    "A1": TaskSpec(
        id="A1", module="legged_velocity", cls="LeggedVelocityTask", config_cls=A1Config,
        domain="locomotion", desc="Unitree A1 (MJCF, Menagerie) - velocity-command walking.",
        default_envs=4096, max_epochs=1000,
        train_kwargs={"horizon_length": 24, "mini_epochs": 5}, ppo_attr="LEGGED_PPO_PARAMS"),

    "H1": TaskSpec(
        id="H1", module="legged_velocity", cls="LeggedVelocityTask", config_cls=H1Config,
        domain="locomotion", desc="Unitree H1 humanoid (MJCF, Menagerie) - bipedal velocity walking.",
        default_envs=4096, max_epochs=1500,
        train_kwargs={"horizon_length": 24, "mini_epochs": 5}, ppo_attr="LEGGED_PPO_PARAMS"),

    "FrankaReach": TaskSpec(
        id="FrankaReach", module="franka_reach", cls="FrankaReachTask", config_cls=FrankaReachConfig,
        domain="manipulation", desc="Franka Panda arm (MJCF, Menagerie) - end-effector reach to a random target.",
        default_envs=4096, max_epochs=300,
        train_kwargs={"horizon_length": 16, "mini_epochs": 5}, ppo_attr="FRANKA_PPO_PARAMS"),

    "FrankaLift": TaskSpec(
        id="FrankaLift", module="franka_lift", cls="FrankaLiftTask", config_cls=FrankaLiftConfig,
        domain="manipulation", desc="Franka Panda + gripper - grasp a cube and hold it at a goal height.",
        default_envs=4096, max_epochs=1000,
        train_kwargs={"horizon_length": 24, "mini_epochs": 5}, ppo_attr="FRANKA_LIFT_PPO_PARAMS"),
}


def load_task(spec: TaskSpec):
    """Import the task module and resolve (module, task_class, ppo_params). Triggers the
    engine bootstrap that the task module performs at import — call only after the caller
    has set LUMENGINE_ROOT (the CLIs do)."""
    module = importlib.import_module(spec.module)
    cls = getattr(module, spec.cls)
    ppo = getattr(module, spec.ppo_attr) if spec.ppo_attr else None
    return module, cls, ppo
