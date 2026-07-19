"""Phase 1 tensor-surface smoke test for the lm.rl façade (Isaac-parity).

Builds a small anymal world headless, warms the direct-GPU batch up, then acquires
+ refreshes every façade tensor and asserts shapes / finiteness / known values.

Run directly:
    set LUMENGINE_ROOT=...\\Lumengine2
    set LUMENGINE_BUILD_CONFIG=Release
    set LM_PHYSX_SHARE_CUDA_CONTEXT=1
    python tests/test_rl_tensors.py

Requires CUDA at runtime (the direct-GPU batch). Skips cleanly if torch has no CUDA.
NOTE: `rl.destroy_world` hard-exits the process (os._exit), so this test runs the
assertions FIRST and only tears down at the very end.
"""
import os
import sys
from pathlib import Path

NUM_ENVS = 4
N_DOF = 12
N_LINKS = 60   # anymal_c composed link count


def run():
    import lumengine_envs.tasks.anymal_task as A
    import lm.rl as rl
    import torch

    if not torch.cuda.is_available():
        print("[test] SKIP: CUDA not available (direct-GPU batch required)")
        return 0

    task = A.AnymalTask(num_envs=NUM_ENVS, headless=True)
    sim = task.sim
    for _ in range(4000):
        task.warmup_step()
        task.runner.run()
        if task.ready:
            break
    assert task.ready, "direct-GPU batch never became ready"
    n, d = sim.num_envs, sim.num_dofs
    assert n == NUM_ENVS and d == N_DOF, f"unexpected dims n={n} d={d}"

    # Acquire the full façade surface.
    dof = sim.acquire_dof_state_tensor()
    root = sim.acquire_root_state_tensor()
    applied = sim.acquire_applied_dof_force_tensor()
    body = sim.acquire_rigid_body_state_tensor()
    fsens = sim.acquire_link_incoming_joint_force_tensor()
    jac = sim.acquire_jacobian_tensor()
    mass = sim.acquire_mass_matrix_tensor()
    ncf = sim.acquire_net_contact_force_tensor()
    roots = sim.get_actor_entities()

    # Write a known per-DOF effort so the applied-force readback is deterministic.
    sim.set_dof_actuation_force_tensor(torch.full((n, d), 2.5, device=sim.device))
    sim.simulate(); sim.fetch_results()
    for r in (sim.refresh_dof_state_tensor, sim.refresh_root_state_tensor,
              sim.refresh_applied_dof_force_tensor, sim.refresh_rigid_body_state_tensor,
              sim.refresh_link_incoming_joint_force_tensor, sim.refresh_jacobian_tensor,
              sim.refresh_mass_matrix_tensor, sim.refresh_net_contact_force_tensor):
        r()

    # Shapes.
    assert tuple(dof.shape) == (n, d, 2), dof.shape
    assert tuple(root.shape) == (n, 13), root.shape
    assert tuple(applied.shape) == (n, d), applied.shape
    assert tuple(body.shape) == (n, N_LINKS, 13), body.shape
    assert tuple(fsens.shape) == (n, N_LINKS, 6), fsens.shape
    assert jac.shape[0] == n and jac.ndim == 2, jac.shape
    assert mass.shape == (n, (d + 6) ** 2), mass.shape           # (maxDofs+6)^2
    assert ncf is None, "anymal is a pure articulation -> no free rigid bodies"
    assert len(roots) == n, len(roots)

    # Finiteness.
    for name, t in (("dof", dof), ("root", root), ("applied", applied),
                    ("body", body), ("fsens", fsens), ("jac", jac), ("mass", mass)):
        assert torch.isfinite(t).all(), f"{name} has non-finite values"

    # Applied-effort readback is exactly what we wrote (eJOINT_FORCE is the applied force).
    assert torch.allclose(applied, torch.full_like(applied, 2.5), atol=1e-4), \
        f"applied dof force readback mismatch: absmax={float(applied.abs().max())}"

    # Quaternion of root state is unit-norm (layout sanity).
    qn = root[:, 3:7].norm(dim=1)
    assert torch.allclose(qn, torch.ones_like(qn), atol=1e-3), f"root quat not unit: {qn}"

    print("[test] PHASE1 tensor surface OK")
    rl.destroy_world(task.sim, task.runner)   # hard-exits the process
    return 0


def test_phase1_tensor_surface():
    """pytest entry point (requires the lm.rl runtime env + CUDA)."""
    assert run() == 0


if __name__ == "__main__":
    try:
        sys.exit(run())
    except Exception:
        import traceback
        print("[test] FAILED:")
        traceback.print_exc()
        os._exit(1)
