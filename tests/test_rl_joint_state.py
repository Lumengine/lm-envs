"""Phase F test: UsdPhysics JointStateAPI authored initial joint position is honored
by the USD->PhysX ingest (previously dropped -> articulation started at zero).

Authors `state:angular:physics:position = 20 deg` on the cartpole pole revolute joint,
ingests, and checks the pole DOF starts at ~0.349 rad (not 0). Pure ingest check.

    set LUMENGINE_ROOT=...\\Lumengine2 & set LUMENGINE_BUILD_CONFIG=Release
    set LM_PHYSX_SHARE_CUDA_CONTEXT=1 & python tests/test_rl_joint_state.py
"""
import math
import os
import sys
from pathlib import Path
from lumotion_envs._engine import ensure_engine
ensure_engine()
from lumotion_envs import assets as _assets
import lumotion as rl

ROBOT = _assets.ASSETS / "cartpole_converted" / "cartpole.usda"
WORLD = _assets.ASSETS / "world_jointstate_test.usd"
POLE_ANGLE_DEG = 20.0
NUM_ENVS = 2
POLE_DOF = 1   # cartpole DOF order: 0 = cart (prismatic), 1 = pole (revolute)


def run():
    import torch
    if not torch.cuda.is_available():
        print("[test] SKIP: CUDA not available")
        return 0

    # Author the env grid, then override the pole revolute joints with an initial angle.
    rl.author_world(ROBOT, WORLD, num_envs=NUM_ENVS, spacing=4.0)

    from pxr import Usd, UsdPhysics, Sdf, Tf
    stage = Usd.Stage.Open(str(WORLD))
    n_revolute = 0
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.RevoluteJoint):
            prim.CreateAttribute("state:angular:physics:position",
                                 Sdf.ValueTypeNames.Float).Set(POLE_ANGLE_DEG)
            n_revolute += 1
    stage.GetRootLayer().Save()
    assert n_revolute == NUM_ENVS, f"expected {NUM_ENVS} revolute joints, found {n_revolute}"
    print(f"[test] authored state:angular:physics:position={POLE_ANGLE_DEG} deg on "
          f"{n_revolute} pole joints")

    sim, runner = rl.create_world(str(WORLD), num_envs=NUM_ENVS, dofs_per_actor=2,
                                  config=rl.SimConfig(substeps=2, device="auto"),
                                  headless=True, title="JointState ingest test")
    sim.play()
    for _ in range(4000):
        sim.simulate(); sim.fetch_results(); runner.run()
        if sim._batch_ready():
            break
    assert sim._batch_ready(), "batch never became ready"

    dof = sim.acquire_dof_state_tensor()
    sim.refresh_dof_state_tensor()
    pole = dof[:, POLE_DOF, 0]
    expected = math.radians(POLE_ANGLE_DEG)
    print(f"[test] pole DOF after ingest = {[round(float(v), 3) for v in pole]} rad "
          f"(expected ~{expected:.3f}; without the fix it would be ~0)")

    # The authored ~0.349 rad must show up (it falls slightly over the warmup steps, so
    # allow a band that still clearly excludes the old 0).
    assert (pole > 0.25).all() and (pole < 0.45).all(), \
        f"JointStateAPI initial position not honored: {pole}"
    print("[test] PHASE_F JointStateAPI initial joint position OK")
    rl.destroy_world(sim, runner)
    return 0


def test_phase_f_joint_state():
    assert run() == 0


if __name__ == "__main__":
    try:
        sys.exit(run())
    except Exception:
        import traceback
        print("[test] FAILED:")
        traceback.print_exc()
        os._exit(1)
