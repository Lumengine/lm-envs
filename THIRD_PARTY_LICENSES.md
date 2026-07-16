# Third-party asset licenses

LumengineEnvs itself is Apache-2.0 (see `LICENSE`). The robot models and assets it
uses remain under their **own** licenses, listed here. Each *vendored* asset also
keeps its original license file in its `assets/<asset>/` directory.

Robot models are sourced from their **upstream URDF/MJCF descriptions** and converted
to USD locally (via the engine's converters). **No assets are taken from NVIDIA Isaac
Sim or Omniverse Nucleus** (those packages are proprietary and not redistributable).

**Status legend:** `vendored` = committed here with its license · `fetched` =
downloaded by `scripts/fetch_assets.py` (not committed) · `planned` = not yet added.

| Asset | License | Origin | Status |
|---|---|---|---|
| Cartpole | Zlib / BSD-3 | classic control (Bullet / NVIDIA) | vendored |
| Ant (`ant.xml`) | MIT | OpenAI Gym (github.com/openai/gym) | vendored |
| ANYmal-C | BSD-3-Clause (© ANYbotics AG) | github.com/ANYbotics/anymal_c_simple_description | fetched |
| Humanoid | Apache-2.0 | DeepMind dm_control | planned |
| Walker | Apache-2.0 | DeepMind dm_control / Genesis | planned |
| Unitree Go2 | BSD-3-Clause (© Unitree) | Unitree (unitree_ros) via Genesis; see assets/go2/LICENSE | vendored |
| Unitree A1 / Go1 | BSD-3-Clause (© Unitree) | MuJoCo Menagerie; see assets/a1/LICENSE, assets/go1/LICENSE | vendored |
| Unitree B1 | BSD-3-Clause (© Unitree) | MuJoCo Menagerie | planned |
| Unitree H1 | BSD-3-Clause (© Unitree) | MuJoCo Menagerie; see assets/h1/LICENSE | vendored |
| Unitree G1 | BSD-3-Clause (© Unitree) | unitree_ros / MuJoCo Menagerie | planned |
| Franka Panda | Apache-2.0 (© Franka Robotics) | MuJoCo Menagerie (franka_emika_panda); see assets/franka/LICENSE | vendored |
| KUKA iiwa | Apache-2.0 / BSD-2 | ROS-Industrial / TU Munich | planned |
| Universal Robots UR5e | BSD-3-Clause | ROS-Industrial (description only) | planned |
| Kinova Gen3 / Jaco2 | BSD-3-Clause (© Kinova) | Kinova ROS | planned |
| Shadow Hand | Apache-2.0 | shadow-robot/sr_common + OpenAI | planned |
| Allegro Hand | BSD-3-Clause (© 2016 SimLab / Wonik Robotics) | MuJoCo Menagerie (wonik_allegro); see assets/allegro/LICENSE | vendored |
| Sektion cabinet | BSD-3-Clause (© NVIDIA) | github.com/isaac-sim/IsaacGymEnvs (repo, not Isaac Sim); see assets/sektion_cabinet/PROVENANCE.md | vendored |
| Trifinger | BSD-3-Clause (© Max Planck) | github.com/rr-learning/rrc_simulation | planned |
| Crazyflie (drone) | Apache-2.0 / MIT | Bitcraze / Genesis | planned |
| YCB objects | CC-BY-4.0 (data), MIT (code) | Yale-CMU-Berkeley (ycbbenchmarks.org) | planned |
| Factory / IndustReal parts | BSD-3-Clause (+ NIST attribution) | NVIDIA IsaacGymEnvs / NIST | planned |

## Attribution requirements (per license)

- **BSD / MIT / Zlib**: retain the copyright notice + license text (kept in each
  `assets/<asset>/`).
- **Apache-2.0**: retain copyright + NOTICE; mark any modified asset files.
- **CC-BY-4.0 (YCB)**: cite the Yale-CMU-Berkeley Object and Model Set —
  Calli et al., *The YCB Object and Model Set*, and https://www.ycbbenchmarks.com.

## Explicitly NOT included

NVIDIA Isaac Sim / Omniverse Nucleus assets (proprietary); manufacturer-restricted
robots whose only available package is proprietary (e.g. Boston Dynamics Spot,
Universal Robots' Omniverse package, Rethink Sawyer); and CMU motion-capture data
(AMP motions — academic-use only).
