"""Does the converted H1 USD have collision geometry on the feet (ankle_link)?
Lists, per rigid body, the child collision prims (UsdPhysics.CollisionAPI) + their
UsdGeom types. If ankle_link has no collider, the robot sinks through the ground at
the feet -> the collapse is a missing-foot-collider bug, not a balance problem."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lumotion_envs._engine import ensure_engine
ensure_engine()
from lumotion_envs import assets as _assets
import lm.rl as rl
from lumotion_envs.config import H1Config
from lumotion_envs.tasks.legged_velocity import _make_morph
ASSETS = _assets.ASSETS

cfg = H1Config()
morph = _make_morph(ASSETS / cfg.robot, ASSETS / cfg.rl_yaml)
usd = morph.resolve()
print("prepped USD:", usd)

from pxr import Usd, UsdPhysics, UsdGeom
stage = Usd.Stage.Open(usd)

# Find rigid bodies (links) and, under each, any collider prims.
bodies = [p for p in stage.TraverseAll() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
print(f"\n{len(bodies)} rigid bodies. Per body: collider prims (type) [purpose]\n")
for b in bodies:
    colliders = []
    for c in Usd.PrimRange(b):
        if c.HasAPI(UsdPhysics.CollisionAPI):
            t = c.GetTypeName()
            approx = ""
            mc = UsdPhysics.MeshCollisionAPI.Get(stage, c.GetPath())
            if mc:
                a = mc.GetApproximationAttr()
                approx = f" approx={a.Get()}" if a and a.HasAuthoredValue() else ""
            colliders.append(f"{c.GetName()}({t}{approx})")
    tag = "  <-- FOOT" if "ankle" in b.GetName().lower() else ""
    flag = "" if colliders else "   *** NO COLLIDER ***"
    print(f"  {b.GetName():28} : {', '.join(colliders) if colliders else '(none)'}{flag}{tag}")
