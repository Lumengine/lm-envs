"""Pre-commit ritual — everything a single dev machine should run before
pushing, in one command. No CI infrastructure required.

    set LUMENGINE_ROOT=C:\\path\\to\\Lumengine
    python scripts/preflight.py            # quick: tier0 + 3-task smoke + Cartpole train (~90s)
    python scripts/preflight.py --full     # tier0 + all-task smoke + Cartpole train (~2-3 min)

Exit 0 = safe to push. The long soak is NOT part of this — run it overnight
before milestones: python tests/soak/soak_task.py --task FrankaLift --hours 12
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "scripts" / "run_engine_tests.py"

# Quick smoke picks one task per asset path: vendored USD, URDF, MJCF (+contact-heavy).
QUICK_SMOKE = "Cartpole,Go2,FrankaLift"


def run_stage(label, cmd):
    print(f"\n=== [{label}] {' '.join(str(c) for c in cmd)}")
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=str(REPO)).returncode
    print(f"=== [{label}] {'OK' if rc == 0 else f'FAILED (rc={rc})'} ({time.time()-t0:.0f}s)")
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="smoke every registry task")
    ap.add_argument("--bench", action="store_true",
                    help="add the perf gate (Ant@4096 must clear 300k env-steps/s — "
                         "a floor far below the thermal band, catching real "
                         "regressions only). Meant for the nightly, not every push.")
    args = ap.parse_args()

    stages = [
        ("tier0", [sys.executable, "-m", "pytest", "-q"]),
        ("smoke", [sys.executable, str(RUNNER), "--smoke"]
                  + ([] if args.full else ["--only", QUICK_SMOKE])),
        ("train", [sys.executable, str(RUNNER), "--train"]),
    ]
    if args.bench:
        stages.append(("bench", [sys.executable, str(REPO / "tools" / "bench_steps.py"),
                                 "--task", "Ant", "--envs", "4096",
                                 "--min-env-sps", "300000"]))
    failed = [label for label, cmd in stages if run_stage(label, cmd) != 0]
    print("\n=== preflight:", "PASS — safe to push" if not failed
          else f"FAIL — {', '.join(failed)}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
