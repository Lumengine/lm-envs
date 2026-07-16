"""Tier 1 — generic per-robot ingest smoke: build a small world for one registry
task, warm the direct-GPU batch up, run N steps, and assert the invariants any
healthy task must satisfy. One task per process (engine contract); driven by
scripts/run_engine_tests.py --smoke, or directly:

    set LUMENGINE_ROOT=...
    python tests/tier1/smoke_ingest.py --task Go2 --num-envs 4 --steps 300

Prints "SKIP: ..." + exit 0 when the environment can't run it (no CUDA, missing
fetched asset); any assertion or crash exits non-zero.
"""
import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tasks"))

# The direct-GPU torch interop needs the shared CUDA context; set it before any
# engine import so users (and CI) don't have to remember the env var.
os.environ.setdefault("LM_PHYSX_SHARE_CUDA_CONTEXT", "1")

WARMUP_FRAMES = 4000


def _fail(msg):
    print(f"FAIL: {msg}")
    sys.exit(1)


def _finite(torch, t, what, step):
    if t is None:
        return
    if not torch.isfinite(t).all():
        bad = (~torch.isfinite(t)).sum().item()
        _fail(f"{what} has {bad} non-finite values at step {step}")


def run(task_id: str, num_envs: int, steps: int) -> int:
    from lumengine_envs.registry import REGISTRY, load_task

    if task_id not in REGISTRY:
        _fail(f"unknown task {task_id!r}; ids: {', '.join(sorted(REGISTRY))}")
    spec = REGISTRY[task_id]

    # Missing fetched assets are a SKIP here (tier0 already fails loudly on them);
    # this keeps --smoke usable on a fresh clone before fetch_assets ran.
    cfg_probe = spec.config_cls()
    for field in ("robot", "rl_yaml", "cabinet", "cabinet_yaml"):
        rel = getattr(cfg_probe, field, "") or ""
        if rel and not (REPO / "assets" / rel).exists():
            print(f"SKIP: {task_id}: missing asset assets/{rel} (run scripts/fetch_assets.py)")
            return 0

    import torch
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available (direct-GPU batch required)")
        return 0

    from lumengine_envs.config import build_config
    _, task_cls, _ = load_task(spec)
    cfg = build_config(spec.config_cls, num_envs=num_envs, headless=True)
    task = task_cls(cfg)

    # Warm up until the direct-GPU batch materializes (pattern shared by all tasks).
    for _ in range(WARMUP_FRAMES):
        task.warmup_step()
        task.runner.run()
        if task.ready:
            break
    if not task.ready:
        _fail(f"direct-GPU batch never became ready within {WARMUP_FRAMES} frames")

    obs = task.reset()
    _finite(torch, obs, "reset() obs", 0)
    if obs.shape[0] != num_envs:
        _fail(f"obs rows {obs.shape[0]} != num_envs {num_envs}")

    resets_seen = 0
    for step in range(1, steps + 1):
        actions = torch.zeros((num_envs, task.num_actions), device=task.device)
        if step > steps // 2:      # second half: small random actions
            actions.uniform_(-0.25, 0.25)
        obs, rew, reset, extras = task.step(actions)
        if step % 25 == 0 or step == steps:
            _finite(torch, obs, "obs", step)
            _finite(torch, rew, "reward", step)
            dof = task.sim.acquire_dof_state_tensor()
            task.sim.refresh_dof_state_tensor()
            _finite(torch, dof, "dof_state", step)
        resets_seen += int(reset.sum().item())

    print(f"PASS: {task_id}: {steps} steps x {num_envs} envs, "
          f"{resets_seen} env-resets, obs {tuple(obs.shape)} finite")
    # Teardown last — destroy_world hard-exits the process (exit code 0).
    import lm.rl as rl
    rl.destroy_world(task.sim, task.runner)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--num-envs", type=int, default=4)
    ap.add_argument("--steps", type=int, default=300)
    args = ap.parse_args()
    sys.exit(run(args.task, args.num_envs, args.steps))


if __name__ == "__main__":
    main()
