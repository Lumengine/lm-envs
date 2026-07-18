#!/usr/bin/env python
"""Play (watch) a trained LumengineEnvs policy — windowed by default.

    set LUMENGINE_ROOT=C:\\path\\to\\Lumengine    & set LM_PHYSX_SHARE_CUDA_CONTEXT=1
    python play.py --task Anymal --checkpoint runs/Anymal_.../nn/Anymal.pth
    python play.py --task Anymal --checkpoint <ckpt> --cmd 1,0,0   # all robots walk forward
    python play.py --task Ant --checkpoint runs/Ant_.../nn/Ant.pth --num-envs 16

Replay with the SAME --trainer the checkpoint was trained with (the network differs
per trainer). rl_games checkpoints are runs/.../nn/*.pth.
"""
import argparse
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tasks"))

from lumengine_envs.config import build_config            # noqa: E402
from lumengine_envs.registry import REGISTRY, load_task   # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Play a trained LumengineEnvs policy.")
    ap.add_argument("--task", required=True, help="task id (see train.py --list)")
    ap.add_argument("--checkpoint", required=True, help="path to the trained checkpoint")
    ap.add_argument("--num-envs", type=int, default=None,
                    help="env count for the replay (default 16 for a clean view; a "
                         "--set num_envs=... also works — the old default silently "
                         "overrode it, flags win over --set in the config layering)")
    ap.add_argument("--trainer", default="rl_games", choices=["rl_games", "rsl_rl", "skrl"])
    ap.add_argument("--cmd", default=None, help='fixed velocity command "vx,vy,yaw" for locomotion')
    ap.add_argument("--headless", action="store_true", help="no window (benchmark only)")
    ap.add_argument("--no-realtime", action="store_true",
                    help="don't pace the replay to real time (uncapped, as fast as the display renders)")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="replay speed multiplier (1.0 = real time, 0.5 = slow-mo, 2.0 = 2x)")
    ap.add_argument("--config", help="path to a YAML config (default: configs/<Task>.yaml if present)")
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
    finally:
        rl.destroy_world(task.sim, task.runner)


def _default_yaml(task_id):
    p = ROOT / "configs" / f"{task_id}.yaml"
    return str(p) if p.is_file() else None


if __name__ == "__main__":
    main()
