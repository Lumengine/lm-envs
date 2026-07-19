"""Decisive test: hold the default PD stance and record base z + per-foot contact
force EVERY step. If the feet bear weight (~half body weight each) before the base
tips -> colliders/contact are fine and the collapse is pure biped balance (the
open-loop settle tips a statically-unstable inverted pendulum). If the feet never
bear load while the base sinks through the 0.095 spawn gap -> a contact bug."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lumengine_envs._engine import ensure_engine
ensure_engine()
import lm.rl as rl
import torch
from lumengine_envs.config import H1Config
from lumengine_envs.tasks.legged_velocity import LeggedVelocityTask, _quat_rotate_inv


def main():
    task = LeggedVelocityTask(cfg=H1Config(num_envs=16, headless=True))
    for _ in range(400):
        task.sim.simulate(); task.sim.fetch_results()
        if task.runner is not None:
            task.runner.run()
        if task.ready:
            break
    task._dof = task.sim.acquire_dof_state_tensor()
    task._root = task.sim.acquire_root_state_tensor()
    task._contact = task.sim.acquire_link_net_contact_force_tensor()
    task._rigid_state = task.sim.acquire_rigid_body_state_tensor()
    task._world_down = torch.tensor([0.0, 0.0, -1.0], device=task.device).repeat(task.num_envs, 1)
    default = task.robot.default_dof_positions.unsqueeze(0).repeat(task.num_envs, 1)
    lm = task.robot.view.link_map
    feet = torch.tensor([i for n, i in lm.items() if n.upper().endswith("ANKLE_LINK")],
                        device=task.device, dtype=torch.long)

    print("step | base_z | up_proj | footL_z footR_z | footL_F footR_F | totalFootF")
    for k in range(60):
        task.sim.set_dof_position_target_tensor(default)
        task.sim.simulate(); task.sim.fetch_results()
        if task.runner is not None:
            task.runner.run()
        task.sim.refresh_root_state_tensor()
        task.sim.refresh_rigid_body_state_tensor()
        task.sim.refresh_link_net_contact_force_tensor()
        root, rigid, contact = task._root, task._rigid_state, task._contact
        up = float((-_quat_rotate_inv(root[:, 3:7], task._world_down)[:, 2]).mean())
        fz = rigid[:, feet, 2].mean(dim=0)
        ff = contact[:, feet, :].norm(dim=2).mean(dim=0)
        if k < 20 or k % 5 == 0:
            print(f"{k:4} | {float(root[:,2].mean()):+.3f} | {up:+.2f} | "
                  f"{float(fz[0]):+.3f} {float(fz[1]):+.3f} | "
                  f"{float(ff[0]):7.1f} {float(ff[1]):7.1f} | {float(ff.sum()):7.1f}")
    rl.destroy_world(task.sim, task.runner)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback; traceback.print_exc()
