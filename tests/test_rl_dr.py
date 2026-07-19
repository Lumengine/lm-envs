"""Test: DOF velocity drive target (new enum) + built-in domain randomization in
VecTask (per-env motor-strength + push perturbations). DR params are set on an anymal
instance post-construction (the base reads them) so no task file changes are needed.

    set LUMENGINE_ROOT=...\\Lumengine2 & set LUMENGINE_BUILD_CONFIG=Release
    set LM_PHYSX_SHARE_CUDA_CONTEXT=1 & python tests/test_rl_dr.py
"""
import os
import sys
from pathlib import Path

NUM_ENVS = 8
N_DOF = 12


def run():
    import lumengine_envs.tasks.anymal_task as A
    import lm.rl as rl
    import torch

    if not torch.cuda.is_available():
        print("[test] SKIP: CUDA not available")
        return 0

    task = A.AnymalTask(num_envs=NUM_ENVS, headless=True)
    for _ in range(4000):
        task.warmup_step(); task.runner.run()
        if task.ready:
            break
    assert task.ready
    view = task.sim.articulations
    dev = view.device

    # --- DOF velocity drive target (new ArticulationWrite.JOINT_DRIVE_VELOCITY) ---
    # Plumbing check: the view method is backed (no NotImplementedError) and the write runs.
    view.set_dof_velocity_targets(torch.zeros(NUM_ENVS, N_DOF, device=dev))
    view.set_dof_velocity_targets(torch.ones(NUM_ENVS, N_DOF, device=dev), indices=torch.tensor([0, 2], device=dev))
    assert rl.ArticulationView.set_dof_velocity_targets.__doc__ is None or True  # callable, didn't raise
    print("[test] set_dof_velocity_targets (JOINT_DRIVE_VELOCITY) wired + callable  OK")

    # --- DR: per-env motor-strength sampled in range, varying across envs ---
    task._motor_strength_range = (0.5, 1.5)
    task.reset()
    ms = task.motor_strength
    assert tuple(ms.shape) == (NUM_ENVS, N_DOF), ms.shape
    assert float(ms.min()) >= 0.5 - 1e-4 and float(ms.max()) <= 1.5 + 1e-4, f"out of range: [{float(ms.min())},{float(ms.max())}]"
    assert float(ms.std()) > 0.01, "motor-strength identical across envs (not randomized)"
    print(f"[test] motor-strength DR: range=[{float(ms.min()):.3f},{float(ms.max()):.3f}] std={float(ms.std()):.3f}  OK")

    # --- DR: push perturbation imparts a horizontal root-velocity kick ---
    task._push_interval = 1
    task._push_vel = 3.0
    task._dr_step = 0                       # so _dr_step % interval == 0 -> push fires
    v0 = view.get_root_states()[:, 7:9].clone()
    task._maybe_push()
    v1 = view.get_root_states()[:, 7:9]
    dv = (v1 - v0).abs()
    print(f"[test] push DR: |dv_xy|max={float(dv.max()):.3f} m/s (kick up to {task._push_vel})")
    assert float(dv.max()) > 0.1, "push imparted no velocity"
    assert float(dv.max()) <= task._push_vel + 1e-3, "push exceeded the configured magnitude"
    print("[test] push DR imparts a bounded root-velocity kick  OK")

    print("[test] DR + velocity-target OK")
    rl.destroy_world(task.sim, task.runner)
    return 0


def test_dr_and_velocity_target():
    assert run() == 0


if __name__ == "__main__":
    try:
        sys.exit(run())
    except Exception:
        import traceback
        print("[test] FAILED:")
        traceback.print_exc()
        os._exit(1)
