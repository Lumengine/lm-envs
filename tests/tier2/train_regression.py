"""Tier 2 — mini-train regression: a short rl_games training must IMPROVE the
mean per-step reward. This is the tier that catches the historical bring-up
bugs (lost resets, poisoned home pose, dead drives): with any of them, reward
goes flat or NaN instead of up.

One task per process (engine contract); driven by run_engine_tests.py --train:

    set LUMENGINE_ROOT=...
    python tests/tier2/train_regression.py --task Cartpole

Method: wrap task.step to record the batch-mean reward of every policy step,
train for --epochs, then compare the first and last windows. Trainer-agnostic
(reads the task side, not rl_games logs).
"""
import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.environ.setdefault("LM_PHYSX_SHARE_CUDA_CONTEXT", "1")

# Per-task budgets: epochs kept short — this is a regression gate, not a baseline.
# improve_factor: required last-window mean vs first-window mean (signed-safe check).
BUDGETS = {
    "Cartpole": dict(epochs=30, num_envs=512, window=200),
    "Ant":      dict(epochs=40, num_envs=1024, window=200),
    "Go2":      dict(epochs=60, num_envs=1024, window=200),
}
DEFAULT_BUDGET = dict(epochs=60, num_envs=1024, window=200)


def _fail(msg):
    print(f"FAIL: {msg}")
    sys.exit(1)


def run(task_id: str, epochs: int | None, num_envs: int | None) -> int:
    from lumotion_envs.config import build_config
    from lumotion_envs.registry import REGISTRY, load_task

    if task_id not in REGISTRY:
        _fail(f"unknown task {task_id!r}")
    spec = REGISTRY[task_id]
    budget = BUDGETS.get(task_id, DEFAULT_BUDGET)
    epochs = epochs or budget["epochs"]
    num_envs = num_envs or budget["num_envs"]
    window = budget["window"]

    cfg_probe = spec.config_cls()
    for field in ("robot", "rl_yaml", "cabinet", "cabinet_yaml"):
        rel = getattr(cfg_probe, field, "") or ""
        if rel and not (REPO / "assets" / rel).exists():
            print(f"SKIP: {task_id}: missing asset assets/{rel} (run scripts/fetch_assets.py)")
            return 0

    import torch
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available")
        return 0

    _, task_cls, ppo = load_task(spec)
    task = task_cls(build_config(spec.config_cls, num_envs=num_envs,
                                 seed=0, headless=True))

    # Record the batch-mean reward of every policy step, trainer-agnostically.
    rewards = []
    orig_step = task.step

    def recording_step(actions):
        obs, rew, reset, extras = orig_step(actions)
        rewards.append(float(rew.mean().item()))
        return obs, rew, reset, extras

    task.step = recording_step

    import lm.rl as rl
    kw = dict(max_epochs=epochs, seed=0, **spec.train_kwargs)
    if ppo is not None:
        kw["params"] = ppo
    rl.train_rl_games(task, **kw)

    if len(rewards) < 2 * window:
        _fail(f"too few recorded steps ({len(rewards)}) for window={window} — "
              f"training ended early or the wrapper was bypassed")
    import math
    if any(not math.isfinite(r) for r in rewards):
        _fail("non-finite reward encountered during training")

    first = sum(rewards[:window]) / window
    last = sum(rewards[-window:]) / window
    print(f"[tier2] {task_id}: mean step-reward first {window}: {first:.4f} "
          f"-> last {window}: {last:.4f} over {len(rewards)} steps")
    if not last > first:
        _fail(f"reward did not improve (first={first:.4f}, last={last:.4f}) — "
              f"training regression")

    print(f"PASS: {task_id}: reward improved {first:.4f} -> {last:.4f} "
          f"({epochs} epochs x {num_envs} envs)")
    rl.destroy_world(task.sim, task.runner)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--num-envs", type=int)
    args = ap.parse_args()
    sys.exit(run(args.task, args.epochs, args.num_envs))


if __name__ == "__main__":
    main()
