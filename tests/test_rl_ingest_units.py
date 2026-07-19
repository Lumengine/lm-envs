"""M1 fidelity test: the USD->PhysX ingest converts schema-DEGREE angular units to
PhysX RADIANS. Pins schema conformance end-to-end: a third-party asset authored per
the UsdPhysics/PhysxSchema unit conventions must simulate at the authored magnitudes.

Authors a minimal floating-base pendulum robot directly with pxr, schema-conformant:
  - `physics:angularVelocity = (0, 0, 90)` deg/s on the base body, and
  - a DriveAPI("angular") with `targetVelocity = 90` deg/s and damping authored
    per-DEGREE (torque per deg/s),
builds a 2-env world (gravity zeroed so nothing falls), steps, and asserts from the
tensors that BOTH measured angular velocities are ~pi/2 rad/s (10% band: drive
settle). On a pre-fix ingest those attributes pass through raw (degrees read as
radians) and measure ~57x too fast — this test FAILS there and PASSES post-fix.

    set LUMENGINE_ROOT=...\\Lumengine-rl & set LUMENGINE_BUILD_CONFIG=Release
    set LM_PHYSX_SHARE_CUDA_CONTEXT=1 & python tests/test_rl_ingest_units.py
"""
import math
import os
import sys
from pathlib import Path
from lumotion_envs._engine import ensure_engine
ensure_engine()
from lumotion_envs import assets as _assets
import lm.rl as rl

ROBOT = _assets.ASSETS / "pendulum_units_test.usda"
WORLD = _assets.ASSETS / "world_ingest_units_test.usd"
NUM_ENVS = 2
ANGVEL_DEG_S = 90.0          # schema deg/s -> pi/2 rad/s effective
DAMPING_PER_DEG = 5.0        # schema torque per deg/s -> 286.5 N.m.s/rad effective
EXPECTED = math.radians(ANGVEL_DEG_S)


def author_robot():
    """A 1-DOF floating-base pendulum, authored in SCHEMA units (degrees). The base
    is deliberately heavy (inertia 10) vs the arm (0.01) so the drive's reaction
    torque cannot visibly perturb the base spin we assert on."""
    # The ENGINE's pxr must be the one imported (a system usdex pxr found first
    # would poison the process's USD DLLs and break the lm.lumydra import later).
    from lm.rl._sim import _prepare_usd_runtime
    _prepare_usd_runtime()
    from pxr import Usd, UsdGeom, UsdPhysics, Gf

    if ROBOT.exists():
        os.remove(ROBOT)
    stage = Usd.Stage.CreateNew(str(ROBOT))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/pendulum")
    stage.SetDefaultPrim(root.GetPrim())

    base = UsdGeom.Xform.Define(stage, "/pendulum/base")
    UsdPhysics.ArticulationRootAPI.Apply(base.GetPrim())
    rb = UsdPhysics.RigidBodyAPI.Apply(base.GetPrim())
    # Schema: physics:angularVelocity is DEGREES/second (about +Z here).
    rb.CreateAngularVelocityAttr(Gf.Vec3f(0.0, 0.0, ANGVEL_DEG_S))
    bmass = UsdPhysics.MassAPI.Apply(base.GetPrim())
    bmass.CreateMassAttr(100.0)
    bmass.CreateDiagonalInertiaAttr(Gf.Vec3f(10.0, 10.0, 10.0))
    bgeom = UsdGeom.Cube.Define(stage, "/pendulum/base/geom")
    bgeom.CreateSizeAttr(0.3)
    UsdPhysics.CollisionAPI.Apply(bgeom.GetPrim())

    # Arm well above the base cube so the two links can never touch as it spins.
    arm = UsdGeom.Xform.Define(stage, "/pendulum/arm")
    arm.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.4))
    UsdPhysics.RigidBodyAPI.Apply(arm.GetPrim())
    amass = UsdPhysics.MassAPI.Apply(arm.GetPrim())
    amass.CreateMassAttr(0.1)
    amass.CreateDiagonalInertiaAttr(Gf.Vec3f(0.01, 0.01, 0.01))
    ageom = UsdGeom.Cube.Define(stage, "/pendulum/arm/geom")
    ageom.CreateSizeAttr(0.1)
    UsdPhysics.CollisionAPI.Apply(ageom.GetPrim())

    # Revolute about Z: with gravity zeroed the drive is the only torque source.
    j = UsdPhysics.RevoluteJoint.Define(stage, "/pendulum/joint")
    j.CreateBody0Rel().SetTargets(["/pendulum/base"])
    j.CreateBody1Rel().SetTargets(["/pendulum/arm"])
    j.CreateAxisAttr("Z")
    j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.4))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    # Schema: angular DriveAPI velocity target in deg/s, damping per deg/s.
    drive = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "angular")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(0.0)
    drive.CreateDampingAttr(DAMPING_PER_DEG)
    drive.CreateTargetVelocityAttr(ANGVEL_DEG_S)
    stage.GetRootLayer().Save()


def run():
    import torch
    if not torch.cuda.is_available():
        print("[test] SKIP: CUDA not available")
        return 0

    author_robot()
    rl.author_world(ROBOT, WORLD, num_envs=NUM_ENVS, spacing=4.0)

    # Zero gravity so the floating base holds still and only the authored angular
    # quantities move anything (author_world hardcodes 9.81).
    from pxr import Usd, UsdPhysics
    stage = Usd.Stage.Open(str(WORLD))
    UsdPhysics.Scene(stage.GetPrimAtPath("/World/PhysicsScene")) \
        .CreateGravityMagnitudeAttr(0.0)
    stage.GetRootLayer().Save()

    sim, runner = rl.create_world(str(WORLD), num_envs=NUM_ENVS, dofs_per_actor=1,
                                  config=rl.SimConfig(substeps=2, device="auto"),
                                  headless=True, title="Ingest units test")
    sim.play()
    for _ in range(4000):
        sim.simulate(); sim.fetch_results(); runner.run()
        if sim._batch_ready():
            break
    assert sim._batch_ready(), "batch never became ready"

    dof = sim.acquire_dof_state_tensor()
    root = sim.acquire_root_state_tensor()
    # A short settle: the velocity drive converges in a few steps (tau = I/d ~ ms).
    for _ in range(60):
        sim.simulate(); sim.fetch_results(); runner.run()
    sim.refresh_dof_state_tensor()
    sim.refresh_root_state_tensor()

    joint_vel = dof[:, 0, 1]           # drive-held joint velocity (rad/s)
    base_spin = root[:, 12]            # root angular velocity Z (rad/s)
    print(f"[test] joint velocity = {[round(float(v), 4) for v in joint_vel]} rad/s, "
          f"base spin Z = {[round(float(v), 4) for v in base_spin]} rad/s "
          f"(expected ~{EXPECTED:.4f}; a degree-blind ingest measures ~{ANGVEL_DEG_S:.0f})")

    lo, hi = 0.9 * EXPECTED, 1.1 * EXPECTED
    assert ((joint_vel > lo) & (joint_vel < hi)).all(), \
        f"DriveAPI angular targetVelocity/damping not converted deg->rad: {joint_vel}"
    assert ((base_spin > lo) & (base_spin < hi)).all(), \
        f"physics:angularVelocity not converted deg->rad: {base_spin}"
    print("[test] M1 ingest angular-unit conversion OK")
    rl.destroy_world(sim, runner)
    return 0


def test_ingest_units():
    assert run() == 0


if __name__ == "__main__":
    try:
        sys.exit(run())
    except Exception:
        import traceback
        print("[test] FAILED:")
        traceback.print_exc()
        os._exit(1)
