"""Endurance soak — run a contact-heavy task for hours and fail on ANY of:
native death (the process exits non-zero by itself), non-finite tensors,
or unbounded memory growth. The nightly answer to "does a long training
survive?" (historically: PhysX GPU contact-gen faults, host event leaks,
unbounded caches).

    set LUMENGINE_ROOT=...
    python tests/soak/soak_task.py --task FrankaLift --hours 12
    python tests/soak/soak_task.py --task AllegroCube --hours 12

Heartbeats print every --report-s seconds: steps, steps/s, CUDA memory, host
working set. Memory check: the LAST heartbeat's host RSS must not exceed the
first post-warmup heartbeat by more than --max-rss-growth-mb.
"""
import argparse
import ctypes
import ctypes.wintypes as wt
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tasks"))
os.environ.setdefault("LM_PHYSX_SHARE_CUDA_CONTEXT", "1")


def _host_rss_mb() -> float:
    """Process working set (MB) via Win32 (no psutil dependency)."""
    class PMC(ctypes.Structure):
        _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t)]
    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    k32 = ctypes.windll.kernel32
    k32.GetCurrentProcess.restype = wt.HANDLE            # 64-bit pseudo-handle
    fn = getattr(k32, "K32GetProcessMemoryInfo", None) or ctypes.windll.psapi.GetProcessMemoryInfo
    fn.argtypes = [wt.HANDLE, ctypes.POINTER(PMC), wt.DWORD]
    fn.restype = wt.BOOL
    if not fn(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
        raise OSError("GetProcessMemoryInfo failed")
    return pmc.WorkingSetSize / (1024 * 1024)


def _fail(msg):
    print(f"FAIL: {msg}")
    sys.exit(1)


def run(task_id, hours, num_envs, report_s, max_rss_growth_mb) -> int:
    from lumengine_envs.config import build_config
    from lumengine_envs.registry import REGISTRY, load_task
    import torch
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available")
        return 0

    spec = REGISTRY[task_id]
    _, task_cls, _ = load_task(spec)
    task = task_cls(build_config(spec.config_cls, num_envs=num_envs,
                                 seed=0, headless=True))
    for _ in range(4000):
        task.warmup_step()
        task.runner.run()
        if task.ready:
            break
    assert task.ready, "batch never ready"
    task.reset()

    deadline = time.time() + hours * 3600.0
    t_last, n_last, step = time.time(), 0, 0
    baseline_rss = None
    actions = torch.zeros((num_envs, task.num_actions), device=task.device)
    print(f"[soak] {task_id} x {num_envs} envs for {hours:.1f}h — started")

    while time.time() < deadline:
        actions.uniform_(-0.5, 0.5)     # random policy = maximum contact churn
        obs, rew, reset, _ = task.step(actions)
        step += 1
        if step % 200 == 0:
            if not (torch.isfinite(obs).all() and torch.isfinite(rew).all()):
                _fail(f"non-finite obs/reward at step {step}")
        now = time.time()
        if now - t_last >= report_s:
            sps = (step - n_last) / (now - t_last)
            # Device-level usage (mem_get_info sees PhysX/engine allocations too;
            # torch.cuda.memory_allocated only counts torch's own caching allocator).
            free_b, total_b = torch.cuda.mem_get_info()
            cuda_mb = (total_b - free_b) / (1024 * 1024)
            rss = _host_rss_mb()
            if baseline_rss is None:
                baseline_rss = rss
            print(f"[soak] step={step} sps={sps:.1f} cuda={cuda_mb:.0f}MB "
                  f"rss={rss:.0f}MB (+{rss - baseline_rss:.0f})", flush=True)
            if rss - baseline_rss > max_rss_growth_mb:
                _fail(f"host RSS grew {rss - baseline_rss:.0f} MB > "
                      f"{max_rss_growth_mb} MB budget — leak")
            t_last, n_last = now, step

    print(f"PASS: {task_id}: {step} steps over {hours:.1f}h, "
          f"final RSS growth {(_host_rss_mb() - (baseline_rss or 0)):.0f} MB")
    import lm.rl as rl
    rl.destroy_world(task.sim, task.runner)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="FrankaLift")
    ap.add_argument("--hours", type=float, default=12.0)
    ap.add_argument("--num-envs", type=int, default=1024)
    ap.add_argument("--report-s", type=float, default=60.0)
    ap.add_argument("--max-rss-growth-mb", type=float, default=2048.0)
    args = ap.parse_args()
    sys.exit(run(args.task, args.hours, args.num_envs, args.report_s,
                 args.max_rss_growth_mb))


if __name__ == "__main__":
    main()
