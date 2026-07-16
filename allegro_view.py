"""Windowed Allegro hand + cube — watch the palm-up hand and the cube behavior."""
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "tasks"))
import _bootstrap; _bootstrap.bootstrap()
ASSETS=_bootstrap.ASSETS
import torch, lm.rl as rl
world = rl.World(num_envs=1, env_spacing=0.6)
world.add_ground(z=-0.5, friction=1.0)
hand = world.add_robot(rl.Mjcf(str(ASSETS/"allegro/right_hand.xml"), prep=True, config=str(ASSETS/"allegro.rl.yaml")), spawn_z=0.4, rpy=(0.0,-math.pi/2,0.0))
S=0.045
world.add_static(rl.Box(size=(S,S,S), dynamic=True, color=(0.85,0.25,0.15),
                        solver_position_iterations=16, solver_velocity_iterations=1,
                        max_depenetration_velocity=5.0, max_linear_velocity=10.0),
                 at=(0.02,0.0,0.46), per_env=True)
sim,runner = world.build(headless=False, title="Allegro + cube",
                         config=rl.SimConfig(dt=1/60,substeps=2,device="auto",
                                             gpu_found_lost_pairs_capacity=2_000_000))
sim.play()
i=0
while not sim._batch_ready():
    sim.simulate(); sim.fetch_results()
    if runner: runner.run()
    i+=1
    if i>4000: sys.exit(3)
print("PROBE: scene ready — regarde la fenêtre (main paume-en-haut + cube)")
k=0
try:
    while True:
        sim.simulate(); sim.fetch_results()
        if runner: runner.run()
        k+=1
except KeyboardInterrupt:
    pass
