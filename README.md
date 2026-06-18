# Particle Filter Localization

Monte Carlo Localization (MCL) for F1tenth / Pacifica on **ROS 2 (Foxy/Humble)**. Fuses any `sensor_msgs/LaserScan`-compatible 2D LiDAR with wheel odometry to maintain a probability distribution (particle cloud) over robot pose (x, y, θ) on a pre-built occupancy grid.

[![YouTube Demo](./media/thumb.jpg)](https://www.youtube.com/watch?v=-c_0hSjgLYw)

For high efficiency in Python it uses NumPy arrays and [RangeLibc](https://github.com/f1tenth/range_libc) for fast 2D ray casting. A GPU ray-casting backend (`rmgpu`) via CUDA is supported for maximum throughput on Jetson platforms.

---

## Table of Contents

- [Changes from upstream](#changes-from-upstream)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Architecture](#architecture)
- [ROS Interface](#ros-interface)
- [TF Tree](#tf-tree)
- [Global Localization](#global-localization)
- [Ray-Casting Backends](#ray-casting-backends)
- [Sensor Model Variants](#sensor-model-variants)
- [Performance Notes](#performance-notes)
- [Future Work](#future-work)
- [Citation](#citation)

---

## Changes from upstream

This fork targets **real-robot deployment** on platforms with a complete ROS 2 TF tree
(e.g. Pacifica / Autonole). The upstream `foxy-devel` / `humble-devel` branches work in
simulation but have several issues that surface only on hardware. The changes below fix
those issues and make the node a drop-in replacement for any nav-stack-compatible robot.

### Bug fixes

#### TF tree corruption (critical)
The upstream node publishes a `map → laser` transform directly. On a robot whose URDF
already defines `odom → base_link → sensor_kit → laser`, this creates a cycle in the TF
tree and breaks every downstream node (`amcl`, `nav2`, `move_base`) that tries to look
up `odom → laser` or `map → base_link`.

**Fix:** the node now publishes `map → odom` by composing its `map → laser` estimate
with the robot's own `laser → odom` TF chain. This is the nav-stack standard: the robot
owns `odom → base_link`; the PF corrects accumulated drift by publishing `map → odom`.
The old `map → laser` / `map → base_link` modes are still available via the
`publish_map_to_odom` parameter.

#### Hardcoded frame names
The upstream code uses literal strings `'/map'` and `'/laser'` throughout — including
the leading `/` that ROS 2 deprecates. These must match your robot's URDF exactly or
every published message lands in the wrong frame.

**Fix:** all frame names are now ROS parameters (`global_frame_id`, `base_frame_id`,
`odom_frame_id`). The laser frame is **auto-detected** from the incoming `LaserScan`
header, and the odom / base frames are **auto-detected** from the incoming `Odometry`
message, so the node adapts to any robot's TF tree without manual config changes.

#### LiDAR subscription QoS mismatch
The upstream subscription uses `reliable` QoS. Most real LiDAR drivers (YDLIDAR,
Hokuyo, Velodyne) publish on `best_effort`. The mismatch silently drops every scan —
the node appears to run but never receives data.

**Fix:** the LaserScan subscription now uses `BEST_EFFORT` QoS, matching standard
hardware drivers.

#### NumPy scalar serialization in fake scan
`LaserScan.angle_min/max/angle_increment/range_max` were published as NumPy scalars.
On some ROS 2 / rcl_interfaces builds this raises a `TypeError` at serialize time,
crashing the visualization.

**Fix:** all `LaserScan` fields are explicitly cast to Python `float` (scalar fields)
or `.tolist()` (range array) before publishing.

#### Missing default parameter values
The upstream `declare_parameter()` calls have no defaults, so the node raises
`ParameterUninitializedException` and crashes if launched without a config file (e.g.
during integration testing or with a partial YAML overlay).

**Fix:** all parameters have inline defaults matching `config/localize.yaml`.

### Improvements

#### TF lookup fallback
If the `laser → odom` TF lookup fails at startup (e.g. `static_transformations` is not
yet running), the node falls back to the pose from the latest `/odom` topic message
rather than silently dropping the transform broadcast.

#### Static TF cache
When `static_laser_to_base_link: True` (default), the `laser → base_link` TF is looked
up once and cached for the lifetime of the node. This avoids one TF buffer query per
MCL tick (~40 Hz) and removes a potential source of jitter.

#### Topic names updated for real hardware
Default `scan_topic` and `odometry_topic` are updated to `scan_filtered` and
`odometry/local` — the actual topics produced by the Autonole / Pacifica platform's
LiDAR pipeline and EKF. The original names (`scan`, `odom`) are kept as inline comments
for simulation use.

---

## Installation

### Dependencies

**Map server (nav2):**
```bash
sudo apt update
rosdep update
source /opt/ros/${ROS_DISTRO}/setup.bash
rosdep install --from-paths src --ignore-src -r -y -q
```

**[RangeLibc](https://github.com/f1tenth/range_libc) — required C++ ray-casting library with Python bindings:**
```bash
git clone https://github.com/f1tenth/range_libc.git -b humble-devel
cd range_libc && mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc) && sudo make install
python3 -m pip install Cython==3.0.12
cd ../pywrapper && WITH_CUDA=ON python3 setup.py install --user
```

> Drop `WITH_CUDA=ON` if you do not have a CUDA-capable GPU. The `rmgpu` backend will not be available, but all CPU backends still work.

### Build

Run from the **ROS 2 workspace root** (not this package directory):
```bash
colcon build --packages-select particle_filter
source install/setup.bash
```

---

## Usage

```bash
ros2 launch particle_filter localize_launch.py
```

Override the map at launch time:
```bash
ros2 launch particle_filter localize_launch.py map_name:=my_map
```

Once running, open **RViz** to visualize the map, particle cloud, inferred pose, and fake scan. Use the **"2D Pose Estimate"** tool from the RViz toolbar to initialize particle locations.

If `set_initial_pose: False` and no pose estimate is provided within `global_loc_timeout` seconds, the node automatically triggers hybrid global localization (see [Global Localization](#global-localization)).

---

## Configuration

All parameters live in `config/localize.yaml`. An example sensor-specific overlay is in `config/ydlidar_x4.yaml` (overrides only sensor model params for the YDLIDAR X4).

### General parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `scan_topic` | `scan` | LiDAR scan topic |
| `odometry_topic` | `odom` | Odometry topic |
| `scan_qos_reliability` | `best_effort` | `best_effort` for real hardware; `reliable` for simulation |
| `max_particles` | `2000` | Number of particles; trades accuracy for CPU cost |
| `angle_step` | `18` | Subsample every Nth LiDAR beam before scoring |
| `range_method` | `rmgpu` | Ray-casting backend (see [Ray-Casting Backends](#ray-casting-backends)) |
| `rangelib_variant` | `2` | Sensor model computation path (see [Sensor Model Variants](#sensor-model-variants)) |
| `mcl_hz` | `40.0` | MCL timer rate (Hz); decoupled from odometry rate |
| `scan_max_age` | `0.5` | Skip MCL cycle if cached scan is older than this (seconds) |
| `viz_throttle` | `4` | Publish visualization every Nth MCL cycle |
| `seed` | `-1` | RNG seed; `>= 0` for reproducible runs, `-1` for random |
| `tf_broadcast` | `True` | Set `False` to suppress all `/tf` publishing |

### Sensor model parameters

These depend on your specific sensor. The values below are tuned for the YDLIDAR X4 and are provided as a starting point — adjust them to match your sensor's noise characteristics.

| Parameter | Example value | Description |
| --- | --- | --- |
| `z_hit` | `0.75` | Probability of hitting the intended surface |
| `z_short` | `0.01` | Short-reading probability (e.g. crosstalk, glass) |
| `z_max` | `0.07` | Max-range miss probability |
| `z_rand` | `0.12` | Random noise return probability |
| `sigma_hit` | `4.0` | Sensor noise standard deviation **in pixels** |
| `lambda_short` | `0.05` | Exponential rate for the short-reading beam component (Thrun model) |
| `max_range` | `10.0` | Maximum valid range in metres |

> **Pixel units for `sigma_hit`**: multiply your sensor's range accuracy (metres) by `1 / map_resolution`. For example, ±0.10 m at 0.05 m/px ≈ 2 px; `4.0` is a conservative starting value.

### Motion model parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `motion_dispersion_x` | `0.05` | Per-step noise in forward direction |
| `motion_dispersion_y` | `0.025` | Per-step noise in lateral direction |
| `motion_dispersion_theta` | `0.25` | Per-step noise in heading |

### Frame IDs

| Parameter | Default | Description |
| --- | --- | --- |
| `global_frame_id` | `map` | Fixed map frame |
| `odom_frame_id` | `odom` | Odom frame; set to `''` to publish `map → base_link` directly |
| `base_frame_id` | `base_link` | Robot base frame |
| `static_laser_to_base_link` | `True` | Cache the `laser → base_link` TF once; avoids repeated lookups |

---

## Architecture

### MCL algorithm (one cycle of `_mcl_timer` at `mcl_hz` Hz)

1. **Resample** — draw `max_particles` indices using systematic (low-variance) resampling weighted by particle weights
2. **Motion model** — apply accumulated odometry delta in car-local frame + Gaussian noise per axis
3. **Sensor model** — ray-cast each particle via RangeLibc; score against a precomputed `P(observed | true_range)` LUT
4. **Normalize** — divide weights by sum; compute `inferred_pose = Σ wᵢ · pᵢ`
5. **Publish** — broadcast TF (`map → odom` or `map → base_link`) and optional odometry topic
6. **Diagnostics** — compute effective particle count `N_eff = 1 / Σ wᵢ²`; warn if `N_eff < 10 % of max_particles`

### Concurrency model

MCL is **timer-driven**. `odomCB` accumulates odometry deltas between ticks (no MCL triggered from odometry). `lidarCB` caches the latest scan only. The node runs on a `SingleThreadedExecutor`.

Two locks provide defensive concurrency:
- `state_lock` — guards the particle array and weights; MCL cycle is skipped (with a warning) if the lock cannot be acquired immediately
- `_odom_lock` — guards odometry accumulation and snapshot in `update()`

### Map setup

Maps live in `maps/<name>.yaml` + `maps/<name>.pgm`. The launch file reads the default map name from `map_server.ros__parameters.map` in `localize.yaml` and falls back to `levine`. Override at launch time with `map_name:=<name>`.

---

## ROS Interface

### Subscribed topics

| Topic | Type | Description |
| --- | --- | --- |
| `scan` (configurable) | `sensor_msgs/LaserScan` | LiDAR scan input |
| `odom` (configurable) | `nav_msgs/Odometry` | Wheel odometry |
| `initialpose` | `geometry_msgs/PoseWithCovarianceStamped` | Manual pose initialization from RViz |
| `clicked_point` | `geometry_msgs/PointStamped` | Triggers hybrid global localization |

### Published topics

| Topic | Type | Description |
| --- | --- | --- |
| `pf/viz/inferred_pose` | `geometry_msgs/PoseStamped` | Best-estimate pose |
| `pf/viz/particles` | `geometry_msgs/PoseArray` | Full particle cloud |
| `pf/viz/fake_scan` | `sensor_msgs/LaserScan` | Ray-cast scan at inferred pose |
| `pf/pose/odom` | `nav_msgs/Odometry` | Estimated odometry (if `publish_odom: 1`) |
| `/tf` | — | `map → odom` or `map → base_link` |

### Services

| Service | Type | Description |
| --- | --- | --- |
| `global_localize` | `std_srvs/Empty` | Trigger hybrid global localization manually |

```bash
ros2 service call /global_localize std_srvs/srv/Empty
```

---

## TF Tree

`publish_tf()` selects the transform to broadcast based on which frame IDs are configured:

| `odom_frame_id` | `base_frame_id` | Published transform | Notes |
| --- | --- | --- | --- |
| set (e.g. `odom`) | set | `map → odom` | **Nav-stack standard**; requires existing `odom → base_link` |
| `''` | set (e.g. `base_link`) | `map → base_link` | Uses cached static `laser → base_link` TF |
| `''` | `''` | `map → laser` | Fallback only; non-standard, breaks most nav stacks |

The default config sets both frame IDs, so the node publishes the nav-stack-standard `map → odom` transform. The physical robot TF tree (`base_link → sensor_kit → lidar/IMU/wheels`) is published separately by `launch/static_transformations.launch.py`.

---

## Global Localization

When the robot pose is unknown (no `/initialpose` received), the node can locate itself automatically from the current laser scan using **hybrid global localization**:

1. Build a coarse candidate grid over all free map cells at `global_loc_coarse_res` metre and `global_loc_theta_res` degree spacing
2. Score every candidate by ray-casting against the live scan using the existing sensor model LUT
3. Pick the top-K highest-scoring, spatially separated hypotheses (non-maximum suppression with `global_loc_min_dist` radius)
4. Seed `max_particles / K` particles around each hypothesis with small Gaussian noise; hand off to normal MCL

### Triggers

- **Automatic** — fires after `global_loc_timeout` seconds at startup if no `/initialpose` message arrives
- **RViz click** — click any point in the map (publishes to `clicked_point`)
- **ROS 2 service** — `ros2 service call /global_localize std_srvs/srv/Empty`

### Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `global_loc_coarse_res` | `0.3` | Metres between x/y candidates in coarse grid |
| `global_loc_theta_res` | `30.0` | Degrees between candidate orientations |
| `global_loc_top_k` | `3` | Number of best hypotheses to seed particles around |
| `global_loc_min_dist` | `1.0` | Minimum metres between chosen hypotheses |
| `global_loc_max_candidates` | `50000` | Safety cap; resolution is doubled automatically if exceeded |
| `global_loc_timeout` | `5.0` | Seconds to wait for `/initialpose` before auto-triggering |

### Estimated cost (20 × 20 m indoor map, 0.05 m/px, 35 beams)

| Backend | Approx. time |
| --- | --- |
| `cddt` (CPU) | ~0.3 – 0.8 s |
| `rmgpu` (GPU) | < 0.1 s |

---

## Ray-Casting Backends

Controlled by the `range_method` parameter:

| Method | Description |
| --- | --- |
| `cddt` | **Default** — fast with low initialization time |
| `pcddt` | Pruned CDDT; slightly faster at runtime |
| `glt` | Giant LUT — fastest CPU option; slow startup |
| `rmgpu` | GPU ray marching via CUDA — fastest overall (requires CUDA build of RangeLibc) |
| `bl` | Bresenham's line — simplest; reference/educational use |

![Range Method Performance Comparison](./media/comparison.png)

---

## Sensor Model Variants

Controlled by the `rangelib_variant` parameter:

| Value | Method | Notes |
| --- | --- | --- |
| 0 | Python loop over LUT | Slow; educational |
| 1 | `calc_range_many` + `eval_sensor_model` | Moderate |
| 2 | `calc_range_repeat_angles` + `eval_sensor_model` | **Default** — works with `rmgpu` |
| 3 | `calc_range_repeat_angles_eval_sensor_model` (one-shot) | **Does not work with `rmgpu`** |
| 4 | Radial CDDT optimization | CDDT / PCDDT only |

---

## Performance Notes

- **Executor**: `SingleThreadedExecutor` is used (switched from `MultiThreadedExecutor` after observing 446 % CPU on a Jetson Orin Nano). Do not re-introduce `MultiThreadedExecutor` without profiling first.
- **Resampling**: Systematic (low-variance) resampler replaces multinomial `np.random.choice`; O(N) cost with significantly reduced particle impoverishment.
- **Pre-allocation**: Working arrays are allocated at init time; MCL cycles use in-place NumPy operations to avoid per-cycle heap allocation.
- **Visualization throttle**: `viz_throttle: 4` publishes visualization every 4th MCL cycle (~10 Hz at 40 Hz MCL).
- **Scan staleness guard**: MCL cycle is skipped with a log warning if the cached scan is older than `scan_max_age` seconds.
- **N_eff diagnostic**: Effective particle count `N_eff = 1 / Σ wᵢ²` is logged every 10 cycles; a warning fires if it drops below 10 % of `max_particles`, indicating weight collapse.

---

## Future Work

### Near-term

- **Automatic kidnapped-robot recovery** — the node already computes `N_eff = 1 / Σ wᵢ²` every 10 MCL cycles and logs a warning when it drops below 10 % of `max_particles`. The missing piece is acting on that signal: when `N_eff` falls below a configurable threshold, automatically invoke `_hybrid_global_localize()` to re-seed particles from scratch. The global localizer already exists and works; this only requires wiring the two together and adding a `kidnap_recovery_threshold` parameter (e.g. `0.05`).
- **Speed-dependent motion noise** — scale dispersion by vehicle speed from odometry; currently dispersion is constant per timestep
- **Dynamic parameter updates** — runtime tuning of `max_particles`, `motion_dispersion_*`, `z_hit`, `sigma_hit` via `add_on_set_parameters_callback` without restarting the node
- **QoS parameterization for non-scan topics** — `scan_qos_reliability` is implemented; `/odom` and other subscriptions use hardcoded QoS

### Longer-term

- **EDT / PyTorch ray marching backend** — replace RangeLibc with a Euclidean Distance Transform sphere-tracer implemented in PyTorch (`torch.vmap`), enabling GPU-accelerated localization without per-architecture CUDA compilation. The sensor model interface is already factored to support a backend swap without changing the MCL logic.

### Cleanup

- Remove the four vestigial `MutuallyExclusiveCallbackGroup`s left over from the `MultiThreadedExecutor` era (harmless but unused under `SingleThreadedExecutor`)

---

## Docs

A mathematical derivation of MCL is in [docs/Lab5.pdf](docs/Lab5.pdf) along with [RangeLibc documentation](docs/RangeLibcUsageandInformation.pdf).

---

## Citation

This library accompanies the following [publication](https://arxiv.org/abs/1705.01167):

```bibtex
@article{walsh17,
    author = {Corey Walsh and Sertac Karaman},
    title  = {CDDT: Fast Approximate 2D Ray Casting for Accelerated Localization},
    volume = {abs/1705.01167},
    url    = {https://arxiv.org/abs/1705.01167},
    year   = {2017}
}
```