"""Per-env CONTACT-material DR (+ per-DOF joint friction smoke).

The material test proves per-env INDEPENDENCE: after the build-time material
un-sharing, giving env 0 near-zero friction and env 1 high friction produces a
clearly different slide distance under the same horizontal push.

TIMING CONTRACT (direct-GPU): PhysX's GPU contact pipeline captures materials at
the FIRST simulate() and ignores all later material updates, so material DR is a
STARTUP operation — applied here right after the world builds, BEFORE the warmup
steps. (A post-step call is dropped with a warning by the facade; the old version
of this test did exactly that and could never differentiate.)

The assertion is direction-agnostic (|low - high| gap): an articulated robot's
slide distance is not monotonic in friction (frictionless feet splay and the
robot bellies out; grippy feet convert the push into a stumble), so the test
requires a significant SEPARATION, not a particular ordering.

    set LUMENGINE_ROOT=...\\Lumengine & set LUMENGINE_BUILD_CONFIG=Release
    set LM_PHYSX_SHARE_CUDA_CONTEXT=1 & python tests/test_rl_material_dr.py
"""
import os
import sys
from pathlib import Path

NUM_ENVS = 4
N_DOF = 12
MIN_GAP = 0.05     # meters of slide-distance separation between the two groups


def run():
    import lumengine_envs.tasks.anymal_task as A
    import lm.rl as rl
    import torch

    if not torch.cuda.is_available():
        print("[test] SKIP: CUDA not available")
        return 0

    task = A.AnymalTask(num_envs=NUM_ENVS, headless=True)
    sim = task.sim

    # STARTUP material DR: articulations exist right after the build ticks, before
    # any simulate(). Half the envs near-frictionless, half grippy.
    for _ in range(2000):
        sim.pump_scene()
        task.runner.run()
        if sim._batch_count_ready():
            break
    assert sim._batch_count_ready(), "articulations never built"
    fr = torch.tensor([0.0] * (NUM_ENVS // 2) + [8.0] * (NUM_ENVS - NUM_ENVS // 2))
    sim.set_material_properties(fr)
    print("[test] per-env materials set BEFORE the first simulate (startup DR)")

    for _ in range(4000):
        task.warmup_step()
        task.runner.run()
        if task.ready:
            break
    assert task.ready
    view = sim.articulations
    dev = view.device

    for _ in range(20):
        sim.simulate(); sim.fetch_results()

    # Joint friction: runs on the live articulation (drive-side, like the drive params).
    view.set_friction_coefficients(torch.full((NUM_ENVS, N_DOF), 0.5, device=dev))
    print("[test] set_friction_coefficients (joint) runs OK")

    # Same lateral push for everyone; the two friction groups must separate.
    root = view.get_root_states().clone(); root[:, 7] = 2.5; root[:, 8:13] = 0.0
    view.set_root_states(root)
    xy0 = view.get_root_states()[:, 0:2].clone()
    for _ in range(45):
        sim.simulate(); sim.fetch_results()
    dist = (view.get_root_states()[:, 0:2] - xy0).norm(dim=1)
    lo = float(dist[:NUM_ENVS // 2].mean()); hi = float(dist[NUM_ENVS // 2:].mean())
    print(f"[test] per-env friction slide: fr=0 group {lo:.3f} vs fr=8 group {hi:.3f} "
          f"(gap {abs(lo - hi):.3f})")
    assert abs(lo - hi) > MIN_GAP, (
        f"per-env contact friction not differentiating: low={lo} high={hi}")
    print("[test] -> per-env contact friction works (build-time material un-sharing)")

    print("[test] PER-ENV MATERIAL + JOINT FRICTION DR OK")
    rl.destroy_world(task.sim, task.runner)
    return 0


def test_material_and_joint_friction_dr():
    assert run() == 0


if __name__ == "__main__":
    try:
        sys.exit(run())
    except Exception:
        import traceback
        print("[test] FAILED:")
        traceback.print_exc()
        os._exit(1)
