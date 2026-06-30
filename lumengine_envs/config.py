"""Typed task configuration — replaces the scattered ``os.environ`` knobs the tasks
used to read. Engine-free (no `lm.rl` import) so configs can be built and `--help`/
`--list` shown without CUDA or the engine.

Layering (lowest to highest precedence): dataclass defaults -> `configs/<task>.yaml`
-> CLI `--set key=value` / explicit flags. The CLIs (`train.py`/`play.py`) do the
layering via `build_config` and pass the result to the task constructor, which reads
`self.cfg.<field>`.
"""

from dataclasses import dataclass, fields


# ── config dataclasses (one per task; common knobs in the base) ──────────────

@dataclass
class BaseConfig:
    num_envs: int = 4096
    env_spacing: float = 4.0
    seed: int = 0
    headless: bool = True
    # Fixed velocity command "vx,vy,yaw" for locomotion tasks; "" = per-env random
    # (the training distribution). Set by play.py --cmd for a clean demo.
    cmd: str = ""


@dataclass
class CartpoleConfig(BaseConfig):
    num_envs: int = 512
    force_mag: float = 400.0          # action in [-1,1] -> [-force_mag, force_mag] N
    robot_usd: str = ""               # "" = the vendored cartpole.usda
    num_states: int = 0               # >0 = asymmetric (privileged) critic state dim


@dataclass
class AntConfig(BaseConfig):
    num_envs: int = 4096
    env_spacing: float = 2.5
    spawn_z: float = 0.62             # clear the feet off the ground at spawn
    torque_scale: float = 15.0        # action -> joint torque (N*m)
    armature: float = 0.5             # per-DOF joint inertia (stability under torque control)
    fall_drop: float = 0.5            # terminate if base drops this far below spawn
    upright_min: float = 0.2          # terminate if up-projection falls below this (flipped)


@dataclass
class LeggedConfig(BaseConfig):
    """Velocity-command quadruped locomotion — shared across legged robots. A specific
    robot is just a subclass that overrides the asset/robot defaults below."""
    num_envs: int = 4096
    env_spacing: float = 4.0
    # Robot identity (asset-relative paths under assets/). Morph type is inferred from the
    # extension: .usd/.usda -> Usd, .urdf -> Urdf, .xml -> Mjcf.
    name: str = "Legged"              # task name (checkpoint / run dir)
    robot: str = ""
    rl_yaml: str = ""                 # prep config (floating base + PD drives)
    num_dof: int = 12
    ground_z: float = -0.65           # ground plane height (so the feet rest on it)
    spawn_z: float = 0.0
    action_scale: float = 0.5         # target = action_scale * action + default stance
    # Reward recipe: "isaaclab" = the 11-term flat-velocity set (good for a high-authority
    # robot like anymal); "genesis" = the minimal 6-term set (tracking + light penalties +
    # similar_to_default) that lets a low-authority robot like Go2 actually walk.
    reward: str = "isaaclab"
    feet_air_time_threshold: float = 0.5   # swing-time (s) the foot-air-time reward credits past
    reset_mode: str = "scaled"        # "scaled" = default*U(0.5,1.5) (IsaacGymEnvs); "offset" = default+small noise
    foot_suffix: str = "FOOT"         # link-name suffix for feet (contact / air-time)
    thigh_suffix: str = "THIGH"       # link-name suffix for the undesired/knee-contact links
    base_contact_fail_n: float = 1.0  # base/knee contact force (N) above which = a fall
    upright_min: float = 0.5          # terminate if up-projection drops below this (tipped over)
    # "auto" = instanced only when headless (training); "on"/"off" force it.
    instance: str = "auto"
    # Terrain mode: "flat" | "noise" | "variants" | "curriculum".
    terrain: str = "flat"
    terrain_amp: float = 0.10         # noise amplitude (m) for terrain="noise"
    terrain_cells: int = 4            # noise base cells (more = tighter/steeper)
    terrain_strategy: str = "round_robin"   # per-env assignment for terrain="variants"
    curriculum_levels: int = 8
    curriculum_size: float = 8.0      # difficulty-tile side (m)
    curriculum_init: int = 1          # max initial difficulty level
    scatter: int = 0                  # N cylinder obstacles scattered per env (0 = none)


@dataclass
class AnymalConfig(LeggedConfig):
    name: str = "Anymal"
    robot: str = "anymal_converted/anymal.usda"
    rl_yaml: str = "anymal_c.rl.yaml"
    ground_z: float = -0.65
    action_scale: float = 0.5


@dataclass
class Go2Config(LeggedConfig):
    name: str = "Go2"
    robot: str = "go2/urdf/go2.urdf"
    rl_yaml: str = "go2.rl.yaml"
    env_spacing: float = 2.5
    ground_z: float = -0.40           # go2 is smaller than anymal (lower stance)
    action_scale: float = 0.5         # enough leg swing to step in our sim (paired with Kp40/Kd1.0)
    reward: str = "genesis"           # minimal recipe — Go2 walks; the anymal set smothers it
    feet_air_time_threshold: float = 0.2   # go2 trots fast — short swing; don't punish it
    reset_mode: str = "offset"        # spawn near the stance so the stiffer snap doesn't tip it


@dataclass
class Go1Config(LeggedConfig):
    name: str = "Go1"
    robot: str = "go1/go1.xml"        # MuJoCo Menagerie MJCF (BSD-3)
    rl_yaml: str = "go1.rl.yaml"
    env_spacing: float = 2.5
    ground_z: float = -0.30           # ~0.27 m standing stance (keyframe home)
    action_scale: float = 0.5
    reward: str = "genesis"           # same recipe that makes Go2 walk
    feet_air_time_threshold: float = 0.2
    reset_mode: str = "offset"
    # The MJCF has no foot body — the foot is a geom on the calf, so contact lands on the
    # calf link. (Reward recipe "genesis" ignores feet anyway; this only labels contacts.)
    foot_suffix: str = "calf"
    thigh_suffix: str = "thigh"


@dataclass
class A1Config(Go1Config):
    name: str = "A1"
    robot: str = "a1/a1.xml"          # MuJoCo Menagerie MJCF (BSD-3)
    rl_yaml: str = "a1.rl.yaml"


# ── layering helpers ─────────────────────────────────────────────────────────

def _coerce(value, to_type):
    """Coerce a (possibly string, from --set) value to the dataclass field type."""
    if isinstance(value, to_type) and to_type is not bool:
        return value
    if to_type is bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ("1", "true", "yes", "on")
    if to_type is int:
        return int(value)
    if to_type is float:
        return float(value)
    return str(value)


def apply_dict(cfg, overrides: dict):
    """Apply a {field: value} dict onto `cfg` in place, coercing by field type.
    Unknown keys raise (typo protection — the whole point of leaving env vars)."""
    types = {f.name: f.type for f in fields(cfg)}
    for k, v in overrides.items():
        if k not in types:
            raise KeyError(
                f"unknown config key {k!r} for {type(cfg).__name__}; "
                f"valid keys: {', '.join(sorted(types))}")
        ftype = types[k] if isinstance(types[k], type) else type(getattr(cfg, k))
        setattr(cfg, k, _coerce(v, ftype))
    return cfg


def apply_set(cfg, sets):
    """Apply a list of "key=value" strings (CLI --set) onto `cfg`."""
    overrides = {}
    for item in sets or []:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        k, v = item.split("=", 1)
        overrides[k.strip()] = v.strip()
    return apply_dict(cfg, overrides)


def load_yaml(path):
    import yaml
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def build_config(cfg_cls, *, yaml_path=None, sets=None, **flag_overrides):
    """Build a config: defaults -> yaml -> --set -> explicit flags. Flags whose value
    is None are ignored (so an unset CLI flag keeps the yaml/default value)."""
    cfg = cfg_cls()
    if yaml_path:
        apply_dict(cfg, load_yaml(yaml_path))
    apply_set(cfg, sets)
    apply_dict(cfg, {k: v for k, v in flag_overrides.items() if v is not None})
    return cfg
