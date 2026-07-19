"""Console entry points — `lumotion-train` / `lumotion-play` (pip install lumotion-envs), also
reachable as `python train.py` / `python play.py` from a repo checkout.

    lumotion-train --list                                   # the task catalog
    lumotion-train --task Anymal                            # headless train (rl_games)
    lumotion-train --task Anymal --num-envs 4096 --epochs 1500
    lumotion-train --task Ant --view                        # windowed: watch it train
    lumotion-train --task Anymal --set terrain=noise --set terrain_amp=0.15
    lumotion-play  --task Anymal --checkpoint runs/Anymal_.../nn/Anymal.pth --cmd 1,0,0

Config layering: dataclass defaults -> configs/<Task>.yaml (auto-loaded if
present) -> --set key=value -> explicit flags. `--list` works without the
engine. Engine location: wheel install (`pip install lumotion`) or a dev
checkout via LUMENGINE_ROOT — see `lumengine_envs._engine`.
"""
import argparse
import copy
from pathlib import Path

from lumengine_envs.config import build_config
from lumengine_envs.registry import REGISTRY, load_task

CONFIGS_DIR = Path(__file__).resolve().parent / "configs"


def _default_yaml(task_id):
    p = CONFIGS_DIR / f"{task_id}.yaml"
    return str(p) if p.is_file() else None


def _print_catalog():
    print(f"{'TASK':<12} {'DOMAIN':<13} {'ENVS':>6}  DESCRIPTION")
    print("-" * 78)
    for s in REGISTRY.values():
        print(f"{s.id:<12} {s.domain:<13} {s.default_envs:>6}  {s.desc}")
    print("\nPer-task knobs: lumotion-train --task <Name> --show-config")


def train_main():
    ap = argparse.ArgumentParser(description="Train a Lumotion task.")
    ap.add_argument("--task", help="task id (see --list)")
    ap.add_argument("--list", action="store_true", help="print the task catalog and exit")
    ap.add_argument("--show-config", action="store_true", help="print the task's config fields and exit")
    ap.add_argument("--num-envs", type=int, default=None, help="parallel env count")
    ap.add_argument("--epochs", type=int, default=None, help="training epochs")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--trainer", default="rl_games", choices=["rl_games", "rsl_rl", "skrl"])
    ap.add_argument("--view", action="store_true", help="windowed (watch training live)")
    ap.add_argument("--config", help="path to a YAML config (default: the packaged configs/<Task>.yaml if present)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override any config field (repeatable), e.g. --set terrain=noise")
    ap.add_argument("--resume-from", default=None,
                    help="checkpoint .pth to restore + continue training from (rl_games); "
                         "used by train_segments.py to chain a long run around a flaky "
                         "process death")
    args = ap.parse_args()

    if args.list or not args.task:
        _print_catalog()
        return
    if args.task not in REGISTRY:
        ap.error(f"unknown task {args.task!r}; choose from: {', '.join(REGISTRY)} (or --list)")
    spec = REGISTRY[args.task]

    if args.show_config:
        import dataclasses
        print(f"{spec.id} config ({spec.config_cls.__name__}):")
        for f in dataclasses.fields(spec.config_cls):
            print(f"  {f.name:<22} = {f.default!r}")
        return

    yaml_path = args.config or _default_yaml(spec.id)
    cfg = build_config(spec.config_cls, yaml_path=yaml_path, sets=args.set,
                       num_envs=args.num_envs, seed=args.seed, headless=not args.view)

    module, cls, ppo = load_task(spec)        # imports the task module (engine bootstrap)
    import lm.rl as rl

    epochs = args.epochs or spec.max_epochs
    task = cls(cfg)
    exit_code = 0
    try:
        if args.view and hasattr(module, "_frame_camera"):
            try:
                module._frame_camera(task)
            except Exception:
                pass
        if args.trainer == "rsl_rl":
            rl.train_rsl_rl(task, max_iterations=epochs, seed=cfg.seed)
        elif args.trainer == "skrl":
            horizon = spec.train_kwargs.get("horizon_length", 16)
            rl.train_skrl(task, timesteps=epochs * cfg.num_envs * horizon,
                          seed=cfg.seed, headless=cfg.headless)
        else:
            kw = dict(max_epochs=epochs, seed=cfg.seed, **spec.train_kwargs)
            if ppo is not None:
                kw["params"] = ppo
            if args.resume_from:
                kw["resume_from"] = args.resume_from
            rl.train_rl_games(task, **kw)
    except Exception:
        import traceback
        print("[train] run raised:")
        traceback.print_exc()
        exit_code = 1
    finally:
        rl.destroy_world(task.sim, task.runner)
    raise SystemExit(exit_code)


def play_main():
    ap = argparse.ArgumentParser(description="Play (watch) a trained Lumotion policy.")
    ap.add_argument("--task", required=True, help="task id (see lumotion-train --list)")
    ap.add_argument("--checkpoint", required=True, help="path to the trained checkpoint")
    ap.add_argument("--num-envs", type=int, default=None,
                    help="env count for the replay (default 16 for a clean view; a "
                         "--set num_envs=... also works — flags win over --set in the "
                         "config layering)")
    ap.add_argument("--trainer", default="rl_games", choices=["rl_games", "rsl_rl", "skrl"])
    ap.add_argument("--cmd", default=None, help='fixed velocity command "vx,vy,yaw" for locomotion')
    ap.add_argument("--headless", action="store_true", help="no window (benchmark only)")
    ap.add_argument("--no-realtime", action="store_true",
                    help="don't pace the replay to real time (uncapped, as fast as the display renders)")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="replay speed multiplier (1.0 = real time, 0.5 = slow-mo, 2.0 = 2x)")
    ap.add_argument("--config", help="path to a YAML config (default: the packaged configs/<Task>.yaml if present)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override any config field (repeatable)")
    args = ap.parse_args()

    if args.task not in REGISTRY:
        ap.error(f"unknown task {args.task!r}; choose from: {', '.join(REGISTRY)}")
    spec = REGISTRY[args.task]

    yaml_path = args.config or _default_yaml(spec.id)
    cfg = build_config(spec.config_cls, yaml_path=yaml_path, sets=args.set,
                       num_envs=args.num_envs, headless=args.headless, cmd=args.cmd)
    # Viewing default: 16 envs, but ONLY when neither --num-envs nor --set num_envs=
    # asked for something (a non-None flag default would silently override --set).
    if args.num_envs is None and not any(
            s.split("=", 1)[0].strip() == "num_envs" for s in args.set):
        cfg.num_envs = 16

    module, cls, ppo = load_task(spec)
    import lm.rl as rl

    task = cls(cfg)
    exit_code = 0
    try:
        if not cfg.headless and hasattr(module, "_frame_camera"):
            try:
                module._frame_camera(task)
            except Exception:
                pass
        if args.trainer == "rsl_rl":
            rl.play_rsl_rl(task, args.checkpoint)
        elif args.trainer == "skrl":
            rl.play_skrl(task, args.checkpoint, headless=cfg.headless)
        else:
            # Rebuild the SAME network the checkpoint was trained with, plus a
            # deterministic player config.
            params = copy.deepcopy(ppo) if ppo else {}
            params.setdefault("params", {}).setdefault("config", {})["player"] = {
                "games_num": 100000, "deterministic": True, "render": False}
            rl.play_rl_games(task, args.checkpoint, params=params,
                             realtime=not (cfg.headless or args.no_realtime), speed=args.speed)
    except Exception:
        import traceback
        print("[play] run raised:")
        traceback.print_exc()
        exit_code = 1
    finally:
        rl.destroy_world(task.sim, task.runner)
    raise SystemExit(exit_code)
