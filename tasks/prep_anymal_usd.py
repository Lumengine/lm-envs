"""Post-process the urdf_usd_converter output for the real anymal_c so our engine can
drive it as a FLOATING-base, PD-controlled RL robot:

  1. Deactivate the `root_joint` fixed joint the converter adds to anchor the base to
     the world -> the articulation becomes free-floating (required for locomotion).
  2. Apply a PD position DriveAPI (UsdPhysics) to the 12 actuated revolute joints with
     the IsaacGym ANYmal default standing stance (bent-knee) as the drive target, so
     an implicit PD holds the stance from frame 1.

Edits Payload/Physics.usda in place (idempotent). Run with the engine's pxr.

    python prep_anymal_usd.py
"""
import math
import os
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))   # for _bootstrap
import _bootstrap
_bootstrap.bootstrap()
import lm.rl as rl

_PHYS = _bootstrap.ASSETS / "anymal_converted" / "Payload" / "Physics.usda"

# IsaacGymEnvs ANYmal default joint angles (rad). Mapped by joint NAME -> the converted
# USD uses the exact same names (LF/RF/LH/RH x HAA/HFE/KFE).
_DEFAULT_RAD = {
    "LF_HAA": 0.03,  "LF_HFE": 0.4,  "LF_KFE": -0.8,
    "RF_HAA": -0.03, "RF_HFE": 0.4,  "RF_KFE": -0.8,
    "LH_HAA": 0.03,  "LH_HFE": -0.4, "LH_KFE": 0.8,
    "RH_HAA": -0.03, "RH_HFE": -0.4, "RH_KFE": 0.8,
}
KP, KD, MAX_FORCE = 85.0, 2.0, 80.0   # IsaacGym anymal control + URDF effort limit


def main():
    rl._world._prepare_usd_runtime()
    from pxr import Usd, UsdPhysics, Sdf

    stage = Usd.Stage.Open(str(_PHYS))
    if stage is None:
        raise RuntimeError(f"cannot open {_PHYS}")

    n_drives = 0
    root_done = False
    # TraverseAll (not Traverse) so an already-deactivated root_joint is still visited
    # on a re-run -> the pass stays idempotent.
    for prim in stage.TraverseAll():
        name = prim.GetName()
        if prim.IsA(UsdPhysics.FixedJoint) and name == "root_joint":
            prim.SetActive(False)   # free the base -> floating articulation
            root_done = True
            continue
        if prim.IsA(UsdPhysics.RevoluteJoint) and name in _DEFAULT_RAD:
            deg = math.degrees(_DEFAULT_RAD[name])
            d = UsdPhysics.DriveAPI.Apply(prim, "angular")   # angular position drive
            d.CreateTypeAttr("force")
            d.CreateStiffnessAttr(KP)
            d.CreateDampingAttr(KD)
            d.CreateMaxForceAttr(MAX_FORCE)
            d.CreateTargetPositionAttr(deg)                  # UsdPhysics angular = degrees
            n_drives += 1

    stage.GetRootLayer().Save()
    print(f"[prep] root_joint deactivated: {root_done}")
    print(f"[prep] PD drives applied to {n_drives}/12 actuated joints "
          f"(Kp={KP}, Kd={KD}, maxForce={MAX_FORCE})")
    if n_drives != 12 or not root_done:
        raise SystemExit("[prep] INCOMPLETE — expected 12 drives + root_joint off")


if __name__ == "__main__":
    main()
