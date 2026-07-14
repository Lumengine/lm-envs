#!/usr/bin/env python
"""Unified training CLI for LumengineEnvs.

    set LUMENGINE_ROOT=C:\\path\\to\\Lumengine    & set LM_PHYSX_SHARE_CUDA_CONTEXT=1
    python train.py --list                                   # the task catalog
    python train.py --task Anymal                            # headless train (rl_games)
    python train.py --task Anymal --num-envs 4096 --epochs 1500
    python train.py --task Ant --view                        # windowed: watch it train
    python train.py --task Anymal --set terrain=noise --set terrain_amp=0.15
    python train.py --task Anymal --config configs/Anymal.yaml --trainer rsl_rl

Config layering: dataclass defaults -> configs/<Task>.yaml (auto-loaded if present)
-> --set key=value -> explicit flags. `--list` works without the engine.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))             # lumengine_envs
sys.path.insert(0, str(ROOT / "tasks"))   # task modules + _bootstrap

from lumengine_envs.config import build_config            # noqa: E402
from lumengine_envs.registry import REGISTRY, load_task   # noqa: E402


def _print_catalog():
    print(f"{'TASK':<12} {'DOMAIN':<13} {'ENVS':>6}  DESCRIPTION")
    print("-" * 78)
    for s in REGISTRY.values():
        print(f"{s.id:<12} {s.domain:<13} {s.default_envs:>6}  {s.desc}")
    print("\nPer-task knobs: python train.py --task <Name> --show-config")


def main():
    ap = argparse.ArgumentParser(description="Train a LumengineEnvs task.")
    ap.add_argument("--task", help="task id (see --list)")
    ap.add_argument("--list", action="store_true", help="print the task catalog and exit")
    ap.add_argument("--show-config", action="store_true", help="print the task's config fields and exit")
    ap.add_argument("--num-envs", type=int, default=None, help="parallel env count")
    ap.add_argument("--epochs", type=int, default=None, help="training epochs")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--trainer", default="rl_games", choices=["rl_games", "rsl_rl", "skrl"])
    ap.add_argument("--view", action="store_true", help="windowed (watch training live)")
    ap.add_argument("--config", help="path to a YAML config (default: configs/<Task>.yaml if present)")
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
    except BaseException:
        import traceback
        print("[train] run raised:")
        traceback.print_exc()
    finally:
        rl.destroy_world(task.sim, task.runner)


def _default_yaml(task_id):
    p = ROOT / "configs" / f"{task_id}.yaml"
    return str(p) if p.is_file() else None


if __name__ == "__main__":
    main()
