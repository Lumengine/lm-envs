"""Diagnostic for the H1 biped spawn collapse. Answers three questions with data,
not guesses:

  (A) Did prep apply the per-joint gains + the bent-stance targets? -> read back the
      prepped USD DriveAPI (stiffness/damping/target) per revolute joint prim, and
      compare the prim NAMES to the h1.rl.yaml short-names (a converter rename would
      silently drop both the group gains AND the default_joint_angles to 0).
  (B) Where does the robot actually spawn / settle? -> base z, per-foot z, per-link z,
      and which links register ground contact, BOTH at first-ready AND after holding
      the default PD stance for 200 steps.
  (C) Is default_dof_positions the bent stance or all-zeros? -> print the vector.

    set LUMENGINE_ROOT=...  & set LM_PHYSX_SHARE_CUDA_CONTEXT=1
    py -3.11 tasks/_h1_probe.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tasks"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import _bootstrap
_bootstrap.bootstrap()
import lm.rl as rl
import torch
from lumengine_envs.config import H1Config
from legged_velocity import LeggedVelocityTask, _quat_rotate_inv


def _read_prepped_drives(morph):
    """(A)+(C-source): open the prepped USD and dump per-joint prim name + applied
    DriveAPI gains/target. Reveals a converter rename (names not matching the yaml)."""
    usd_path = morph.resolve()
    meta = morph.meta
    print(f"\n=== (A) PREPPED USD: {usd_path}")
    print(f"    meta num_dofs={meta.get('num_dofs')}  actuated={len(meta.get('actuated_names', []))}")
    print(f"    meta default_joint_angles (name->rad): {meta.get('default_joint_angles')}")
    from pxr import Usd, UsdPhysics
    stage = Usd.Stage.Open(usd_path)
    print(f"\n    per revolute-joint prim  [name : Kp / Kd / maxF / target(deg)]")
    for prim in stage.TraverseAll():
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        d = UsdPhysics.DriveAPI.Get(prim, "angular")
        if not d:
            print(f"      {prim.GetName():28} : (no DriveAPI)")
            continue
        def g(attr):
            a = attr()
            return a.Get() if a and a.HasAuthoredValue() else None
        print(f"      {prim.GetName():28} : "
              f"{g(d.GetStiffnessAttr)} / {g(d.GetDampingAttr)} / "
              f"{g(d.GetMaxForceAttr)} / {g(d.GetTargetPositionAttr)}")


def _snapshot(task, label):
    """(B): base/foot/link z + per-link contact at the current state."""
    task.sim.refresh_root_state_tensor()
    task.sim.refresh_rigid_body_state_tensor()
    task.sim.refresh_link_net_contact_force_tensor()
    root = task._root
    rigid = task._rigid_state          # (envs, links, 13) [pos, quat, lin, ang]
    contact = task._contact            # (envs, links, 3)
    lm = task.robot.view.link_map
    inv = {i: n for n, i in lm.items()}
    up = (-_quat_rotate_inv(root[:, 3:7], task._world_down)[:, 2]).mean()
    print(f"\n=== (B) {label}")
    print(f"    base z(mean)={float(root[:,2].mean()):+.3f}  up_proj(mean)={float(up):.3f}")
    fmag = contact.norm(dim=2).mean(dim=0)
    linkz = rigid[:, :, 2].mean(dim=0)
    print(f"    per-link  z(mean) / contact|F|(N)   (contact>0.5N flagged '*'):")
    for i in range(linkz.shape[0]):
        flag = "*" if float(fmag[i]) > 0.5 else " "
        print(f"    {flag} link[{i:2}] {inv.get(i,'?'):26} z={float(linkz[i]):+.3f}  F={float(fmag[i]):7.2f}")
    feet = task.feet_indices
    print(f"    feet_indices={[inv.get(int(i)) for i in feet]} "
          f"foot z(mean)={[round(float(rigid[:,int(i),2].mean()),3) for i in feet]} "
          f"foot F={[round(float(contact[:,int(i),:].norm(dim=1).mean()),1) for i in feet]}")


def main():
    cfg = H1Config(num_envs=16, headless=True)
    task = LeggedVelocityTask(cfg=cfg)

    # (A) prepped-USD read-back BEFORE any stepping
    _read_prepped_drives(task.robot.morph)

    # warm to ready (materializes the articulation), capture tensors WITHOUT the
    # 80-step settle so we see the raw spawn — call the tensor-acquire part by hand.
    for _ in range(400):
        task.sim.simulate(); task.sim.fetch_results()
        if task.runner is not None:
            task.runner.run()
        if task.ready:
            break
    # acquire tensors + feet indices the way _capture does, but snapshot BEFORE settling
    task._dof = task.sim.acquire_dof_state_tensor()
    task._root = task.sim.acquire_root_state_tensor()
    task._contact = task.sim.acquire_link_net_contact_force_tensor()
    task._rigid_state = task.sim.acquire_rigid_body_state_tensor()
    task._world_down = torch.tensor([0.0, 0.0, -1.0], device=task.device).repeat(task.num_envs, 1)
    task._default_dof = task.robot.default_dof_positions.unsqueeze(0).repeat(task.num_envs, 1)
    lm = task.robot.view.link_map
    foot = cfg.foot_suffix.upper()
    task.feet_indices = torch.tensor(
        [i for n, i in lm.items() if n.upper().endswith(foot)], device=task.device, dtype=torch.long)

    # (C) is the default stance the bent pose or zeros?
    print(f"\n=== (C) default_dof_positions (rad, DOF order):")
    dof = task.robot.default_dof_positions
    n2d = task.robot.view._name_to_dof
    d2n = {i: n for n, i in n2d.items()}
    print("    " + "  ".join(f"{d2n.get(i,'?')}={float(dof[i]):+.2f}" for i in range(dof.numel())))

    _snapshot(task, "AT FIRST READY (raw spawn, no settle)")

    # hold the default stance and watch it settle (or collapse)
    for k in range(1, 201):
        task.sim.set_dof_position_target_tensor(task._default_dof)
        task.sim.simulate(); task.sim.fetch_results()
        if task.runner is not None:
            task.runner.run()
        if k in (20, 60, 200):
            task.sim.refresh_root_state_tensor()
            print(f"    [settle step {k:3}] base z(mean)={float(task._root[:,2].mean()):+.3f}")
    _snapshot(task, "AFTER 200-STEP HOLD OF DEFAULT STANCE")

    rl.destroy_world(task.sim, task.runner)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback; traceback.print_exc()
