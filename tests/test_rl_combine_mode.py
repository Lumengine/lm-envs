"""Material friction/restitution combine modes (PhysxSchema physxMaterial:*CombineMode):
the ingest parses them onto PhysicsMaterialComponent and CollisionShapeFactory applies them
to the PxMaterial. This validates the public SDK binding (enum + getters/setters) round-trips
— the ingest parse + PxMaterial application is exercised by C++.

    set LUMENGINE_ROOT=...\\Lumengine2 & set LUMENGINE_BUILD_CONFIG=Release
    python tests/test_rl_combine_mode.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tasks"))
import _bootstrap
_bootstrap.bootstrap()


def run():
    import lm.rl  # noqa: F401 — loads the engine DLL chain physx.pyd depends on
    import lm.physx as p

    # Enum is bound with PhysX's PxCombineMode ordering.
    assert int(p.PhysicsCombineMode.AVERAGE) == 0
    assert int(p.PhysicsCombineMode.MIN) == 1
    assert int(p.PhysicsCombineMode.MULTIPLY) == 2
    assert int(p.PhysicsCombineMode.MAX) == 3

    # Default is AVERAGE; setters/getters round-trip.
    m = p.PhysicsMaterialComponent(0.5, 0.5, 0.0)
    assert m.get_friction_combine_mode() == p.PhysicsCombineMode.AVERAGE
    assert m.get_restitution_combine_mode() == p.PhysicsCombineMode.AVERAGE
    m.set_friction_combine_mode(p.PhysicsCombineMode.MIN)
    m.set_restitution_combine_mode(p.PhysicsCombineMode.MAX)
    assert m.get_friction_combine_mode() == p.PhysicsCombineMode.MIN
    assert m.get_restitution_combine_mode() == p.PhysicsCombineMode.MAX
    print("[test] PhysicsCombineMode enum + friction/restitution combine setters/getters OK")

    print("[test] MATERIAL COMBINE-MODE BINDING OK")
    return 0


def test_combine_mode_binding():
    assert run() == 0


if __name__ == "__main__":
    try:
        _code = run()
    except Exception:
        import traceback
        print("[test] FAILED:")
        traceback.print_exc()
        os._exit(1)
    os._exit(_code)
