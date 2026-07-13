"""Diagnostic: does the anymal stand on its FEET (foot links bearing load) when holding
the default PD stance? Prints per-link net contact-force magnitude + which links the task
picked as feet / undesired. Tells whether feet_air_time=0 is a detection bug or a learned
posture.

    set LUMENGINE_ROOT=...  & set LM_PHYSX_SHARE_CUDA_CONTEXT=1
    py -3.11 tasks/_anymal_probe.py
"""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap
_bootstrap.bootstrap()
import lm.rl as rl
import torch
import anymal_task as AT


def main():
    task = AT.AnymalTask(num_envs=16, headless=True)
    # warm up until ready (captures tensors + feet/undesired indices)
    for _ in range(400):
        task.sim.simulate(); task.sim.fetch_results()
        if task.runner is not None:
            task.runner.run()
        if task.ready:
            break
    if not task._captured:
        task._capture(); task._captured = True

    lm_map = task.robot.view.link_map
    inv = {i: n for n, i in lm_map.items()}
    print("feet_indices     :", [inv.get(int(i)) for i in task.feet_indices])
    print("undesired (pen.) :", [inv.get(int(i)) for i in task.undesired_contact_indices])
    print("knee/term (thigh):", [inv.get(int(i)) for i in task.knee_indices])

    # Hold the default stance for a while, then read per-link contact.
    default = task._default_dof
    for _ in range(200):
        task.sim.set_dof_position_target_tensor(default)
        task.sim.simulate(); task.sim.fetch_results()
        if task.runner is not None:
            task.runner.run()
    task.sim.refresh_root_state_tensor()
    task.sim.refresh_link_net_contact_force_tensor()
    contact = task._contact            # (envs, links, 3)
    fmag = contact.norm(dim=2).mean(dim=0)   # mean over envs -> (links,)
    root = task._root
    print(f"\nbase root z (mean)={float(root[:,2].mean()):.3f}  up_proj(mean)="
          f"{float((-AT._quat_rotate_inv(root[:,3:7], task._world_down)[:,2]).mean()):.2f}")
    print("\nper-link mean |contact force| (N), links with >0.5 N:")
    for i in range(fmag.shape[0]):
        f = float(fmag[i])
        if f > 0.5:
            print(f"  link[{i:2}] {inv.get(i,'?'):20} = {f:8.2f} N")
    feet_f = contact[:, task.feet_indices, :].norm(dim=2).mean(dim=0)
    print("\nfoot contact forces (mean, N):", [round(float(x), 1) for x in feet_f])
    rl.destroy_world(task.sim, task.runner)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback; traceback.print_exc()
