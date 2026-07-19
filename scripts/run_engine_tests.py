"""Engine-test runner — one subprocess per test, because the engine's contract
is ONE WORLD PER PROCESS (`rl.destroy_world` hard-exits). A shared pytest
session would die after the first engine test; this runner gives each test a
fresh process and aggregates the results.

Usage:
    set LUMENGINE_ROOT=C:\\path\\to\\Lumengine
    python scripts/run_engine_tests.py                 # all tests/test_rl_*.py
    python scripts/run_engine_tests.py --only tensors  # filename filter
    python scripts/run_engine_tests.py --smoke         # Tier 1 ingest smoke, all tasks
    python scripts/run_engine_tests.py --smoke --only Cartpole,Go2

Exit code: 0 = all passed/skipped, 1 = at least one failure.
A test that prints "SKIP" and returns 0 is counted as a skip.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TESTS = REPO / "tests"


def _engine_ready() -> bool:
    root = os.environ.get("LUMENGINE_ROOT")
    if not root:
        return False
    cfg = os.environ.get("LUMENGINE_BUILD_CONFIG", "Release")
    return (Path(root) / "build" / cfg / "python").exists()


def _run_one(label: str, cmd: list, timeout: int):
    t0 = time.time()
    # The direct-GPU torch interop needs the shared CUDA context. The tiered
    # harness sets it itself; the legacy contract tests (tests/test_rl_*.py)
    # expect the CALLER to — so the runner guarantees it for every child.
    env = dict(os.environ)
    env.setdefault("LM_PHYSX_SHARE_CUDA_CONTEXT", "1")
    try:
        proc = subprocess.run(cmd, cwd=str(REPO), timeout=timeout,
                              capture_output=True, text=True, env=env)
        dt = time.time() - t0
        out = (proc.stdout or "") + (proc.stderr or "")
        if proc.returncode == 0:
            status = "SKIP" if "SKIP" in out else "PASS"
        else:
            status = "FAIL"
        return status, dt, out
    except subprocess.TimeoutExpired as e:
        out = ((e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes)
               else (e.stdout or ""))
        return "TIMEOUT", time.time() - t0, out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", help="comma-separated substring filter (filenames or task ids)")
    ap.add_argument("--smoke", action="store_true",
                    help="run the Tier 1 ingest smoke per registry task instead of tests/test_rl_*.py")
    ap.add_argument("--train", action="store_true",
                    help="run the Tier 2 mini-train regression (default tasks: Cartpole; "
                         "use --only to pick others)")
    ap.add_argument("--timeout", type=int, default=1200, help="per-test timeout (s)")
    ap.add_argument("--num-envs", type=int, default=4, help="(--smoke) envs per task")
    ap.add_argument("--steps", type=int, default=300, help="(--smoke) steps per task")
    ap.add_argument("-v", "--verbose", action="store_true", help="print each test's output")
    args = ap.parse_args()

    if not _engine_ready():
        print("run_engine_tests: LUMENGINE_ROOT is not set (or the engine is not built) "
              "— nothing to run.\n    set LUMENGINE_ROOT=C:\\path\\to\\Lumengine")
        sys.exit(2)

    filters = [f.strip().lower() for f in (args.only or "").split(",") if f.strip()]

    jobs = []  # (label, cmd)
    if args.train:
        sys.path.insert(0, str(REPO))
        from lumotion_envs.registry import REGISTRY
        task_ids = ([t for t in REGISTRY if any(f in t.lower() for f in filters)]
                    if filters else ["Cartpole"])
        for task_id in task_ids:
            jobs.append((f"train:{task_id}",
                         [sys.executable, str(TESTS / "tier2" / "train_regression.py"),
                          "--task", task_id]))
    elif args.smoke:
        sys.path.insert(0, str(REPO))
        from lumotion_envs.registry import REGISTRY
        for task_id in REGISTRY:
            if filters and not any(f in task_id.lower() for f in filters):
                continue
            jobs.append((f"smoke:{task_id}",
                         [sys.executable, str(TESTS / "tier1" / "smoke_ingest.py"),
                          "--task", task_id, "--num-envs", str(args.num_envs),
                          "--steps", str(args.steps)]))
    else:
        for p in sorted(TESTS.glob("test_rl_*.py")):
            if filters and not any(f in p.name.lower() for f in filters):
                continue
            jobs.append((p.name, [sys.executable, str(p)]))

    if not jobs:
        print("run_engine_tests: no tests matched the filter")
        sys.exit(2)

    results = []
    for label, cmd in jobs:
        print(f"[run] {label} ...", flush=True)
        status, dt, out = _run_one(label, cmd, args.timeout)
        results.append((label, status, dt))
        print(f"[run] {label}: {status} ({dt:.1f}s)")
        if args.verbose or status in ("FAIL", "TIMEOUT"):
            tail = "\n".join(out.splitlines()[-30:])
            print("      " + "\n      ".join(tail.splitlines()) if tail else "      (no output)")

    print("\n=== summary ===")
    width = max(len(label) for label, *_ in results)
    counts = {}
    for label, status, dt in results:
        counts[status] = counts.get(status, 0) + 1
        print(f"  {label:<{width}}  {status:<7}  {dt:7.1f}s")
    print("  " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    sys.exit(1 if (counts.get("FAIL", 0) + counts.get("TIMEOUT", 0)) else 0)


if __name__ == "__main__":
    main()
