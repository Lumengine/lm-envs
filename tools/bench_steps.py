"""Performance micro-bench — env-steps/s vs num_envs, the standing instrument
for the productization plan's Phase 2 (and the honest answer to marketing FPS).

One world per process (engine contract): with several --envs values this script
re-executes itself once per value and aggregates the RESULT lines.

    set LUMENGINE_ROOT=...
    python tools/bench_steps.py --task Ant --envs 256,1024,4096 --steps 200

Reports both policy-steps/s and env-steps/s (= num_envs * policy-steps/s), plus
substeps/s (* control_freq_inv) so comparisons against substep-counting claims
(Genesis) are explicit. Reference target: Isaac Gym class is ~150-500k
env-steps/s on Ant @ 4096 on one GPU.
"""
import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tasks"))
os.environ.setdefault("LM_PHYSX_SHARE_CUDA_CONTEXT", "1")


def bench_one(task_id: str, num_envs: int, steps: int, warmup_steps: int) -> None:
    from lumengine_envs.registry import REGISTRY, load_task
    import torch
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available")
        sys.exit(0)

    from lumengine_envs.config import build_config
    spec = REGISTRY[task_id]
    _, task_cls, _ = load_task(spec)
    task = task_cls(build_config(spec.config_cls, num_envs=num_envs, headless=True))
    for _ in range(4000):
        task.warmup_step()
        task.runner.run()
        if task.ready:
            break
    assert task.ready, "batch never ready"
    task.reset()

    actions = torch.zeros((num_envs, task.num_actions), device=task.device)
    for _ in range(warmup_steps):
        task.step(actions)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        task.step(actions)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    decim = getattr(task, "control_freq_inv", 1)
    sps = steps / dt
    print(f"RESULT task={task_id} envs={num_envs} steps={steps} wall={dt:.3f}s "
          f"policy_sps={sps:.1f} env_sps={sps * num_envs:.0f} "
          f"env_substeps={sps * num_envs * decim:.0f} decimation={decim}")
    import lm.rl as rl
    rl.destroy_world(task.sim, task.runner)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="Ant")
    ap.add_argument("--envs", default="256,1024,4096",
                    help="comma-separated num_envs values (one subprocess each)")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--_one", type=int, help=argparse.SUPPRESS)   # internal: single run
    args = ap.parse_args()

    if args._one is not None:
        bench_one(args.task, args._one, args.steps, args.warmup)
        return

    env_counts = [int(x) for x in args.envs.split(",") if x.strip()]
    rows = []
    for n in env_counts:
        print(f"[bench] {args.task} @ {n} envs ...", flush=True)
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--task", args.task,
             "--steps", str(args.steps), "--warmup", str(args.warmup), "--_one", str(n)],
            capture_output=True, text=True, cwd=str(REPO))
        out = (proc.stdout or "") + (proc.stderr or "")
        m = re.search(r"RESULT .*", out)
        if proc.returncode != 0 or not m:
            print(f"[bench] {n} envs FAILED:\n" + "\n".join(out.splitlines()[-15:]))
            continue
        print("[bench] " + m.group(0))
        rows.append(m.group(0))

    if rows:
        print("\n=== bench summary ===")
        print(f"{'envs':>8} {'policy sps':>12} {'env-steps/s':>14} {'substeps/s':>14}")
        for r in rows:
            kv = dict(p.split("=") for p in r.split()[1:])
            print(f"{kv['envs']:>8} {float(kv['policy_sps']):>12.1f} "
                  f"{float(kv['env_sps']):>14.0f} {float(kv['env_substeps']):>14.0f}")
        print("\nScaling read: env-steps/s should GROW with envs until the GPU saturates;"
              "\nan early plateau = CPU/sync-bound (see plan 003, Phase 2).")


if __name__ == "__main__":
    main()
