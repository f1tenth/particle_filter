# MIT License

# Copyright (c) 2025 Boluwatife Olabiran

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the 'Software'), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import time
from threading import Lock

import numpy as np
import range_libc
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from rclpy.executors import SingleThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from std_srvs.srv import Empty
from tf2_ros import Buffer, TransformListener, TransformBroadcaster
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import (PoseStamped, PoseWithCovarianceStamped, PoseArray,
                                Quaternion, PolygonStamped, PointStamped)
from nav_msgs.msg import Odometry
from nav_msgs.srv import GetMap

from particle_filter import utils as Utils
from particle_filter.utils import (
    quat_trans_to_matrix, quaternion_from_euler, pose_msg_to_matrix,
    transformstamped_to_matrix, quaternion_from_matrix, matrix_to_transformstamped,
    cov_mat_to_list,
)

VAR_NO_EVAL_SENSOR_MODEL = 0
VAR_CALC_RANGE_MANY_EVAL_SENSOR = 1
VAR_REPEAT_ANGLES_EVAL_SENSOR = 2
VAR_REPEAT_ANGLES_EVAL_SENSOR_ONE_SHOT = 3
VAR_RADIAL_CDDT_OPTIMIZATIONS = 4


class ParticleFilter(Node):
    """Monte Carlo Localization fusing 2D LiDAR with wheel odometry."""

    def __init__(self):
        super().__init__('particle_filter')

        # ── parameters ────────────────────────────────────────────────────────
        self.declare_parameter('angle_step', 18)
        self.declare_parameter('max_particles', 4000)
        self.declare_parameter('max_viz_particles', 60)
        self.declare_parameter('squash_factor', 2.2)
        self.declare_parameter('max_range', 10.0)
        self.declare_parameter('theta_discretization', 112)
        self.declare_parameter('range_method', 'rmgpu')
        self.declare_parameter('rangelib_variant', 2)
        self.declare_parameter('fine_timing', 0)
        self.declare_parameter('publish_odom', 1)
        self.declare_parameter('viz', 1)
        self.declare_parameter('viz_throttle', 4)
        self.declare_parameter('z_short', 0.01)
        self.declare_parameter('z_max', 0.07)
        self.declare_parameter('z_rand', 0.12)
        self.declare_parameter('z_hit', 0.75)
        self.declare_parameter('sigma_hit', 4.0)
        self.declare_parameter('lambda_short', 0.05)
        self.declare_parameter('motion_dispersion_x', 0.05)
        self.declare_parameter('motion_dispersion_y', 0.025)
        self.declare_parameter('motion_dispersion_theta', 0.25)
        # Motion gate (AMCL update_min_d/_a): skip resample+motion-noise when the
        # accumulated odom delta is below these, so a stationary robot does not
        # random-walk from per-tick noise. 0.0 disables the gate (update every tick).
        self.declare_parameter('update_min_d', 0.02)
        self.declare_parameter('update_min_a', 0.01)
        self.declare_parameter('scan_topic', 'scan')
        self.declare_parameter('odometry_topic', 'odom')
        self.declare_parameter('scan_qos_reliability', 'best_effort')
        self.declare_parameter('odom_qos_reliability', 'reliable')
        self.declare_parameter('scan_max_age', 0.5)
        self.declare_parameter('odom_max_age', 0.5)
        self.declare_parameter('mcl_hz', 40.0)
        self.declare_parameter('global_frame_id', 'map')
        self.declare_parameter('odom_frame_id', '')
        self.declare_parameter('base_frame_id', 'base_link')
        self.declare_parameter('static_laser_to_base_link', True)
        self.declare_parameter('transform_tolerance', 0.5)
        # TF republish is normally driven by the odom rate (we publish whenever the
        # odom stamp advances), which avoids TF_REPEATED_DATA drops when the MCL
        # timer runs faster than odom. The heartbeat re-stamps the last transform if
        # this many seconds pass with no fresh odom, so map->odom does not expire
        # while odom is slow (but still alive). 0.0 disables it.
        self.declare_parameter('tf_heartbeat_period', 0.2)
        self.declare_parameter('tf_broadcast', True)
        # Comparison aid (non-standard, default off): in map->base mode, freeze the
        # map->odom correction at the first fix and express the published pose
        # relative to the odom origin, so pf/pose/odom starts coincident with raw
        # odometry and diverges only by the drift the PF corrects. AMCL never does
        # this (it publishes a live map->odom TF); only enable for offline odom-vs-PF
        # trajectory comparison. Requires base_frame_id set (map->base mode).
        self.declare_parameter('anchor_to_odom_origin', False)
        self.declare_parameter('set_initial_pose', False)
        self.declare_parameter('initial_pose.x', 0.0)
        self.declare_parameter('initial_pose.y', 0.0)
        self.declare_parameter('initial_pose.z', 0.0)
        self.declare_parameter('initial_pose.yaw', 0.0)
        self.declare_parameter('seed', -1)
        # Task 9: hybrid global localization
        self.declare_parameter('global_loc_coarse_res', 0.3)
        self.declare_parameter('global_loc_theta_res', 30.0)
        self.declare_parameter('global_loc_top_k', 3)
        self.declare_parameter('global_loc_min_dist', 1.0)
        self.declare_parameter('global_loc_max_candidates', 50000)
        self.declare_parameter('global_loc_timeout', 5.0)

        self.ANGLE_STEP = self.get_parameter('angle_step').value
        self.MAX_PARTICLES = self.get_parameter('max_particles').value
        self.MAX_VIZ_PARTICLES = self.get_parameter('max_viz_particles').value
        self.INV_SQUASH_FACTOR = 1.0 / self.get_parameter('squash_factor').value
        self.MAX_RANGE_METERS = self.get_parameter('max_range').value
        self.THETA_DISCRETIZATION = self.get_parameter('theta_discretization').value
        self.WHICH_RM = self.get_parameter('range_method').value
        self.RANGELIB_VAR = self.get_parameter('rangelib_variant').value
        self.SHOW_FINE_TIMING = self.get_parameter('fine_timing').value
        self.PUBLISH_ODOM = self.get_parameter('publish_odom').value
        self.DO_VIZ = self.get_parameter('viz').value
        self.VIZ_THROTTLE = self.get_parameter('viz_throttle').value
        self.Z_SHORT = self.get_parameter('z_short').value
        self.Z_MAX = self.get_parameter('z_max').value
        self.Z_RAND = self.get_parameter('z_rand').value
        self.Z_HIT = self.get_parameter('z_hit').value
        self.SIGMA_HIT = self.get_parameter('sigma_hit').value
        self.LAMBDA_SHORT = self.get_parameter('lambda_short').value
        self.MOTION_DISPERSION_X = self.get_parameter('motion_dispersion_x').value
        self.MOTION_DISPERSION_Y = self.get_parameter('motion_dispersion_y').value
        self.MOTION_DISPERSION_THETA = self.get_parameter('motion_dispersion_theta').value
        self.UPDATE_MIN_D = self.get_parameter('update_min_d').value
        self.UPDATE_MIN_A = self.get_parameter('update_min_a').value
        self.GLOBAL_FRAME_ID = self.get_parameter('global_frame_id').value
        self.ODOM_FRAME_ID = self.get_parameter('odom_frame_id').value
        self.BASE_FRAME_ID = self.get_parameter('base_frame_id').value
        self.LASER_FRAME_ID = ''
        self.STATIC_LASER_TO_BASE_LINK = self.get_parameter('static_laser_to_base_link').value
        self.TRANSFORM_TOLERANCE = self.get_parameter('transform_tolerance').value
        self.TF_HEARTBEAT_PERIOD = self.get_parameter('tf_heartbeat_period').value
        self.tf_broadcast = self.get_parameter('tf_broadcast').value
        self.ANCHOR_TO_ODOM_ORIGIN = self.get_parameter('anchor_to_odom_origin').value
        self.set_initial_pose = self.get_parameter('set_initial_pose').value
        self.SCAN_MAX_AGE_SEC = self.get_parameter('scan_max_age').value
        self.ODOM_MAX_AGE_SEC = self.get_parameter('odom_max_age').value
        self.MCL_HZ = self.get_parameter('mcl_hz').value
        scan_qos_rel_str = self.get_parameter('scan_qos_reliability').value
        odom_qos_rel_str = self.get_parameter('odom_qos_reliability').value
        self.GLOBAL_LOC_COARSE_RES = self.get_parameter('global_loc_coarse_res').value
        self.GLOBAL_LOC_THETA_RES = self.get_parameter('global_loc_theta_res').value
        self.GLOBAL_LOC_TOP_K = self.get_parameter('global_loc_top_k').value
        self.GLOBAL_LOC_MIN_DIST = self.get_parameter('global_loc_min_dist').value
        self.GLOBAL_LOC_MAX_CANDIDATES = self.get_parameter('global_loc_max_candidates').value
        self.GLOBAL_LOC_TIMEOUT = self.get_parameter('global_loc_timeout').value

        # anchor_to_odom_origin only takes effect in map→base mode; warn if the
        # configured frames will select map→odom or the map→laser fallback instead.
        if self.ANCHOR_TO_ODOM_ORIGIN:
            _will_be_base = bool(self.BASE_FRAME_ID) and not (
                self.ODOM_FRAME_ID and self.ODOM_FRAME_ID != self.GLOBAL_FRAME_ID)
            if not _will_be_base:
                self.get_logger().warn(
                    'anchor_to_odom_origin is set but the frame config does not '
                    'select map→base mode (need base_frame_id set and '
                    'odom_frame_id empty/equal to global_frame_id); anchoring '
                    'will be ignored')

        # E-7: reproducible RNG seed
        seed = self.get_parameter('seed').value
        if seed >= 0:
            np.random.seed(seed)            # legacy API: covers resampling, particle init, viz
        # Fix 4: independent Generator for motion noise (supports out= parameter)
        self._rng = np.random.default_rng(seed if seed >= 0 else None)

        # ── callback groups (P-1) ─────────────────────────────────────────────
        self._lidar_group = MutuallyExclusiveCallbackGroup()
        self._odom_group = MutuallyExclusiveCallbackGroup()
        self._click_group = MutuallyExclusiveCallbackGroup()
        self._mcl_group = MutuallyExclusiveCallbackGroup()

        # ── MCL state ─────────────────────────────────────────────────────────
        self.MAX_RANGE_PX = None
        self.iters = 0
        self.map_info = None
        self.permissible_region = None
        self.map_initialized = False
        self.lidar_initialized = False
        self.odom_initialized = False
        self.laser_angles = None
        self.downsampled_angles = None
        self.downsampled_ranges = None
        self.range_method = None
        self.last_stamp = None
        self.first_sensor_update = True
        self.last_pose = None
        self.odom_pose = None
        self.current_speed = 0.0
        # Frozen inv(map->odom) correction for anchor_to_odom_origin; set once at
        # the first map->base publish, then held constant.
        self._odom_anchor_inv = None
        self.cov_3x3 = np.zeros((3, 3))

        # F-3: readiness flags; MCL timer checks all before running
        self._map_ready = False
        self._laser_ready = False
        self._pose_inited = False
        self._warned_not_ready = False

        # F-4: scan staleness tracking
        self._last_scan_stamp = None
        self._last_odom_stamp = None

        # TF publish pacing: republish only when the odom stamp advances (tracks the
        # odom rate, not the MCL timer) plus a low-rate heartbeat to keep map->odom
        # alive when odom is slow. _last_tf_pub_time is wall/sim time of the last
        # broadcast; None until the first publish.
        self._last_published_stamp = None
        self._last_tf_pub_time = None

        # E-5: range_min populated from each scan message
        self._range_min = 0.0

        # P-6: viz throttle counter
        self._viz_counter = 0

        # E-3: N_eff iteration counter
        self._n_eff_iter = 0

        # cached transforms (static)
        self.base_frame_to_scan_tf = None   # T_base_scan
        self.laser_to_base_frame_tf = None  # T_scan_base = inv(T_base_scan)

        # reused homogeneous matrix for map→scan estimate (avoids per-cycle alloc)
        self.T_map_to_scan = np.eye(4, dtype=np.float64)

        self.state_lock = Lock()
        self._odom_lock = Lock()

        # pre-allocated buffers for motion model and systematic resampling (P-4)
        self.local_deltas = np.zeros((self.MAX_PARTICLES, 3))
        self._proposal = np.zeros((self.MAX_PARTICLES, 3))

        # Fix 4: pre-allocated noise buffer; filled via rng.standard_normal(out=) each cycle
        self._noise = np.empty((self.MAX_PARTICLES, 3), dtype=np.float64)
        self._noise_scale = np.array(
            [self.MOTION_DISPERSION_X, self.MOTION_DISPERSION_Y, self.MOTION_DISPERSION_THETA],
            dtype=np.float64)

        # Fix 3: pre-allocated SE(3) inverse buffer; avoids LAPACK call in publish_tf
        self._T_odom_scan_inv = np.eye(4, dtype=np.float64)

        # accumulated odometry delta; reset to zero after each MCL tick (P-2)
        self.odometry_data = np.zeros(3)

        # sensor-model buffers (allocated on first update, Step 2)
        self.queries = None
        self.ranges = None
        self.tiled_angles = None
        self.sensor_model_table = None
        self._sensor_model_dispatch = None

        self.inferred_pose = None
        self.particle_indices = np.arange(self.MAX_PARTICLES)
        self.particles = np.zeros((self.MAX_PARTICLES, 3))
        self.weights = np.ones(self.MAX_PARTICLES) / float(self.MAX_PARTICLES)

        self.smoothing = Utils.CircularArray(10)
        self.timer = Utils.Timer(10)

        # G-8: global-loc range buffer; deferred to lidarCB (downsampled_angles not yet known)
        self._global_ranges_buf = None
        # G-5: set when uniform-spread fallback ran before scan arrived; cleared on first scan
        self._pending_global_loc = False

        # ── startup sequence ───────────────────────────────────────────────────
        self.map_client = self.create_client(GetMap, 'map_server/map')
        self.get_omap()
        self.precompute_sensor_model()
        # Step 1: unified init; scan not ready yet → uniform spread + _pending_global_loc
        self._reset_particles()

        if self.set_initial_pose:
            q = quaternion_from_euler(
                0.0, 0.0, self.get_parameter('initial_pose.yaw').value)
            ip = PoseWithCovarianceStamped()
            ip.header.frame_id = self.GLOBAL_FRAME_ID
            ip.header.stamp = self.get_clock().now().to_msg()
            ip.pose.pose.position.x = self.get_parameter('initial_pose.x').value
            ip.pose.pose.position.y = self.get_parameter('initial_pose.y').value
            ip.pose.pose.position.z = self.get_parameter('initial_pose.z').value
            ip.pose.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
            self._reset_particles(ip.pose.pose)

        # ── publishers ────────────────────────────────────────────────────────
        self.pose_pub = self.create_publisher(PoseStamped, 'pf/viz/inferred_pose', 1)
        self.particle_pub = self.create_publisher(PoseArray, 'pf/viz/particles', 1)
        self.pub_fake_scan = self.create_publisher(LaserScan, 'pf/viz/fake_scan', 1)
        self.rect_pub = self.create_publisher(PolygonStamped, 'pf/viz/poly1', 1)
        if self.PUBLISH_ODOM:
            self.odom_pub = self.create_publisher(Odometry, 'pf/pose/odom', 1)

        self.pub_tf = TransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ── scan / odom QoS ───────────────────────────────────────────────────
        qos_rel = (QoSReliabilityPolicy.BEST_EFFORT
                   if scan_qos_rel_str.lower() == 'best_effort'
                   else QoSReliabilityPolicy.RELIABLE)
        odom_qos_rel = (QoSReliabilityPolicy.BEST_EFFORT
                        if odom_qos_rel_str.lower() == 'best_effort'
                        else QoSReliabilityPolicy.RELIABLE)

        # ── subscribers ───────────────────────────────────────────────────────
        self.laser_sub = self.create_subscription(
            LaserScan,
            self.get_parameter('scan_topic').value,
            self.lidarCB,
            QoSProfile(depth=1, reliability=qos_rel),
            callback_group=self._lidar_group)
        # depth=10: buffers odom messages queued during ~25 ms MCL cycle
        self.odom_sub = self.create_subscription(
            Odometry,
            self.get_parameter('odometry_topic').value,
            self.odomCB,
            QoSProfile(depth=10, reliability=odom_qos_rel),
            callback_group=self._odom_group)
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped, 'initialpose', self.clicked_pose, 1,
            callback_group=self._click_group)
        self.click_sub = self.create_subscription(
            PointStamped, 'clicked_point', self.clicked_pose, 1,
            callback_group=self._click_group)

        # ── G-6: global localization service ──────────────────────────────────
        self._global_loc_srv = self.create_service(
            Empty, 'global_localize', self._global_localize_srv_cb,
            callback_group=self._click_group)

        # ── MCL timer (P-2) ───────────────────────────────────────────────────
        self._mcl_timer = self.create_timer(
            1.0 / self.MCL_HZ, self._mcl_timer_cb,
            callback_group=self._mcl_group)

        # ── G-7: one-shot startup timer; fires if no pose init within timeout ─
        if not self.set_initial_pose:
            self._startup_timer = self.create_timer(
                self.GLOBAL_LOC_TIMEOUT, self._startup_global_loc_cb,
                callback_group=self._click_group)

        self.get_logger().info('Finished initializing, waiting on messages...')

    # ── helpers ───────────────────────────────────────────────────────────────

    def _get_stamp(self, stamp=None):
        if stamp is None:
            return self.get_clock().now().to_msg()
        return stamp

    def _lookup_tf(self, source_frame, target_frame, timestamp=None, timeout=0.05):
        if timestamp is None:
            timestamp = rclpy.time.Time()
        try:
            return self.tf_buffer.lookup_transform(
                target_frame, source_frame, timestamp,
                rclpy.duration.Duration(seconds=timeout))
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(f'TF {source_frame}→{target_frame}: {e}')
            return None

    def _cache_base_to_scan_tf(self, timestamp=None):
        """Look up base_link → laser and cache; skips if already cached and static (P-3)."""
        if self.base_frame_to_scan_tf is not None and self.STATIC_LASER_TO_BASE_LINK:
            return
        tf = self._lookup_tf(self.BASE_FRAME_ID, self.LASER_FRAME_ID,
                             timestamp, self.TRANSFORM_TOLERANCE)
        if tf is not None:
            self.base_frame_to_scan_tf = transformstamped_to_matrix(tf)
            self.laser_to_base_frame_tf = np.linalg.inv(self.base_frame_to_scan_tf)

    # ── map loading ───────────────────────────────────────────────────────────

    def get_omap(self):
        """Fetch the occupancy grid from map_server and initialise the RangeLibc backend."""
        while not self.map_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for map_server/map service...')
        future = self.map_client.call_async(GetMap.Request())
        rclpy.spin_until_future_complete(self, future)
        map_msg = future.result().map
        self.map_info = map_msg.info

        oMap = range_libc.PyOMap(map_msg)
        self.MAX_RANGE_PX = int(self.MAX_RANGE_METERS / self.map_info.resolution)

        self.get_logger().info(f'Initialising range method: {self.WHICH_RM}')
        if self.WHICH_RM == 'bl':
            self.range_method = range_libc.PyBresenhamsLine(oMap, self.MAX_RANGE_PX)
        elif 'cddt' in self.WHICH_RM:
            self.range_method = range_libc.PyCDDTCast(
                oMap, self.MAX_RANGE_PX, self.THETA_DISCRETIZATION)
            if self.WHICH_RM == 'pcddt':
                self.get_logger().info('Pruning CDDT...')
                self.range_method.prune()
        elif self.WHICH_RM == 'rm':
            self.range_method = range_libc.PyRayMarching(oMap, self.MAX_RANGE_PX)
        elif self.WHICH_RM == 'rmgpu':
            self.range_method = range_libc.PyRayMarchingGPU(oMap, self.MAX_RANGE_PX)
        elif self.WHICH_RM == 'glt':
            self.range_method = range_libc.PyGiantLUTCast(
                oMap, self.MAX_RANGE_PX, self.THETA_DISCRETIZATION)
        self.get_logger().info('Map loaded')

        array_255 = np.array(map_msg.data).reshape(
            (map_msg.info.height, map_msg.info.width))
        self.permissible_region = (array_255 == 0)
        self.map_initialized = True
        self._map_ready = True

    # ── sensor model ──────────────────────────────────────────────────────────

    def precompute_sensor_model(self):
        """
        Build sensor_model_table[observed_px, computed_px] = P(observed | computed).
        Beam model: z_hit (Gaussian) + z_short (exponential, BUG-10) + z_max + z_rand.
        sigma_hit is in pixels (map_resolution × metres).
        lambda_short is the exponential rate for the z_short beam component.
        """
        self.get_logger().info('Precomputing sensor model')
        table_width = int(self.MAX_RANGE_PX) + 1
        self.sensor_model_table = np.zeros((table_width, table_width))

        for d in range(table_width):
            norm = 0.0
            for r in range(table_width):
                z = float(r - d)
                prob = (self.Z_HIT
                        * np.exp(-(z * z) / (2.0 * self.SIGMA_HIT ** 2))
                        / (self.SIGMA_HIT * np.sqrt(2.0 * np.pi)))
                # d > 0 is guaranteed when r < d; canonical exponential z_short model
                if r < d:
                    prob += self.Z_SHORT * self.LAMBDA_SHORT * np.exp(-self.LAMBDA_SHORT * r)
                if r == self.MAX_RANGE_PX:
                    prob += self.Z_MAX
                if r < self.MAX_RANGE_PX:
                    prob += self.Z_RAND / float(self.MAX_RANGE_PX)
                norm += prob
                self.sensor_model_table[r, d] = prob
            self.sensor_model_table[:, d] /= norm

        if self.RANGELIB_VAR > 0:
            self.range_method.set_sensor_model(self.sensor_model_table)

    # ── particle initialisation (Step 1: unified entry point) ─────────────────

    def _reset_particles(self, pose=None):
        """Unified init: pose set → Gaussian around pose; pose=None → hybrid global
        search if scan ready, else uniform spread (upgraded on first scan via G-5)."""
        if pose is not None:
            self._pending_global_loc = False
            with self.state_lock:
                self.weights[:] = 1.0 / self.MAX_PARTICLES
                self.particles[:, 0] = (pose.position.x
                                        + np.random.normal(0.0, 0.5, self.MAX_PARTICLES))
                self.particles[:, 1] = (pose.position.y
                                        + np.random.normal(0.0, 0.5, self.MAX_PARTICLES))
                self.particles[:, 2] = (Utils.quaternion_to_angle(pose.orientation)
                                        + np.random.normal(0.0, 0.4, self.MAX_PARTICLES))
                self.get_logger().info(
                    f'Pose init: [{pose.position.x:.2f}, {pose.position.y:.2f}]')
            self._pose_inited = True
        elif self._laser_ready:
            # G-5: scan available — run coarse search immediately
            self._hybrid_global_localize(self.downsampled_ranges)
        else:
            # G-5: scan not yet available — uniform spread so MCL can start, upgrade later
            with self.state_lock:
                py, px = np.where(self.permissible_region)
                idx = np.random.randint(0, len(px), size=self.MAX_PARTICLES)
                states = np.zeros((self.MAX_PARTICLES, 3))
                states[:, 0] = px[idx]
                states[:, 1] = py[idx]
                states[:, 2] = np.random.uniform(0.0, 2.0 * np.pi, self.MAX_PARTICLES)
                Utils.map_to_world(states, self.map_info)
                self.particles = states
                self.weights[:] = 1.0 / self.MAX_PARTICLES
                self.get_logger().info('Global particle initialisation (uniform; scan not ready)')
            self._pending_global_loc = True
            # _pose_inited stays False until hybrid search succeeds or manual pose arrives

    # ── MCL core ──────────────────────────────────────────────────────────────

    def _systematic_resample(self, weights):
        """O(N) systematic (low-variance) resampling (E-1)."""
        n = len(weights)
        positions = (np.arange(n) + np.random.uniform()) / n
        cumsum = np.cumsum(weights)
        return np.searchsorted(cumsum, positions)

    def motion_model(self, proposal_dist, action):
        """Apply odometry action in each particle's local frame, then add Gaussian noise."""
        cosines = np.cos(proposal_dist[:, 2])
        sines = np.sin(proposal_dist[:, 2])
        self.local_deltas[:, 0] = cosines * action[0] - sines * action[1]
        self.local_deltas[:, 1] = sines * action[0] + cosines * action[1]
        self.local_deltas[:, 2] = action[2]
        proposal_dist += self.local_deltas
        # Fix 4: single PRNG call into pre-allocated buffer; zero heap allocations per cycle
        self._rng.standard_normal(out=self._noise)   # shape (MAX_PARTICLES, 3), N(0,1)
        self._noise *= self._noise_scale             # broadcast scale per axis in place
        proposal_dist += self._noise

    # ── sensor model dispatch (Step 2) ────────────────────────────────────────

    def _init_sensor_buffers(self, num_rays):
        """Allocate MCL ray-cast buffers and build dispatch dict on first sensor update."""
        if self.RANGELIB_VAR <= 1:
            self.queries = np.zeros((num_rays * self.MAX_PARTICLES, 3), dtype=np.float32)
        else:
            self.queries = np.zeros((self.MAX_PARTICLES, 3), dtype=np.float32)
        self.ranges = np.zeros(num_rays * self.MAX_PARTICLES, dtype=np.float32)
        self.tiled_angles = np.tile(self.downsampled_angles, self.MAX_PARTICLES)
        self._sensor_model_dispatch = {
            VAR_NO_EVAL_SENSOR_MODEL:               self._sm_v0,
            VAR_CALC_RANGE_MANY_EVAL_SENSOR:        self._sm_v1,
            VAR_REPEAT_ANGLES_EVAL_SENSOR:          self._sm_v2,
            VAR_REPEAT_ANGLES_EVAL_SENSOR_ONE_SHOT: self._sm_v3,
            VAR_RADIAL_CDDT_OPTIMIZATIONS:          self._sm_v4,
        }
        self.first_sensor_update = False

    def _sm_v0(self, proposal_dist, obs, weights, num_rays):
        """Python loop over sensor model table; slow, educational."""
        self.queries[:, 0] = np.repeat(proposal_dist[:, 0], num_rays)
        self.queries[:, 1] = np.repeat(proposal_dist[:, 1], num_rays)
        self.queries[:, 2] = np.repeat(proposal_dist[:, 2], num_rays) + self.tiled_angles
        self.range_method.calc_range_many(self.queries, self.ranges)
        obs_px = np.clip(obs / self.map_info.resolution, 0, self.MAX_RANGE_PX)
        rng_px = np.clip(self.ranges / self.map_info.resolution, 0, self.MAX_RANGE_PX)
        intobs = np.rint(obs_px).astype(np.uint16)
        intrng = np.rint(rng_px).astype(np.uint16)
        for i in range(self.MAX_PARTICLES):
            w = np.prod(
                self.sensor_model_table[intobs, intrng[i * num_rays:(i + 1) * num_rays]])
            weights[i] = np.power(w, self.INV_SQUASH_FACTOR)

    def _sm_v1(self, proposal_dist, obs, weights, num_rays):
        """calc_range_many + eval_sensor_model."""
        self.queries[:, 0] = np.repeat(proposal_dist[:, 0], num_rays)
        self.queries[:, 1] = np.repeat(proposal_dist[:, 1], num_rays)
        self.queries[:, 2] = np.repeat(proposal_dist[:, 2], num_rays) + self.tiled_angles
        self.range_method.calc_range_many(self.queries, self.ranges)
        self.range_method.eval_sensor_model(
            obs, self.ranges, weights, num_rays, self.MAX_PARTICLES)
        np.power(weights, self.INV_SQUASH_FACTOR, weights)

    def _sm_v2(self, proposal_dist, obs, weights, num_rays):
        """calc_range_repeat_angles + eval_sensor_model. Default for rmgpu."""
        self.queries[:, :] = proposal_dist[:, :]
        self.range_method.calc_range_repeat_angles(
            self.queries, self.downsampled_angles, self.ranges)
        self.range_method.eval_sensor_model(
            obs, self.ranges, weights, num_rays, self.MAX_PARTICLES)
        np.power(weights, self.INV_SQUASH_FACTOR, weights)
        if self.SHOW_FINE_TIMING and self.iters % 10 == 0:
            self.get_logger().info(f'sensor_model variant={self.RANGELIB_VAR}')

    def _sm_v3(self, proposal_dist, obs, weights, num_rays):
        """calc_range_repeat_angles_eval_sensor_model one-shot; incompatible with rmgpu."""
        self.queries[:, :] = proposal_dist[:, :]
        self.range_method.calc_range_repeat_angles_eval_sensor_model(
            self.queries, self.downsampled_angles, obs, weights)
        np.power(weights, self.INV_SQUASH_FACTOR, weights)

    def _sm_v4(self, proposal_dist, obs, weights, num_rays):
        """Radial CDDT optimization; silently degrades to variant 2 for non-CDDT backends."""
        if 'cddt' not in self.WHICH_RM:
            self.get_logger().warn(
                'rangelib_variant 4 requires cddt/pcddt; falling back to variant 2')
            self.RANGELIB_VAR = VAR_REPEAT_ANGLES_EVAL_SENSOR
            self._sm_v2(proposal_dist, obs, weights, num_rays)
            return
        self.queries[:, :] = proposal_dist[:, :]
        self.range_method.calc_range_many_radial_optimized(
            num_rays, self.downsampled_angles[0], self.downsampled_angles[-1],
            self.queries, self.ranges)
        self.range_method.eval_sensor_model(
            obs, self.ranges, weights, num_rays, self.MAX_PARTICLES)
        np.power(weights, self.INV_SQUASH_FACTOR, weights)

    def sensor_model(self, proposal_dist, obs, weights):
        """
        Score particles via RangeLibc ray-casting + beam model LUT.
        rangelib_variant selects the ray-casting strategy (see CLAUDE.md for table).
        Variant 3 is incompatible with rmgpu and raises AssertionError at startup.
        """
        assert not (self.RANGELIB_VAR == VAR_REPEAT_ANGLES_EVAL_SENSOR_ONE_SHOT
                    and self.WHICH_RM == 'rmgpu'), \
            'rangelib_variant 3 is incompatible with rmgpu; use variant 2'

        num_rays = self.downsampled_angles.shape[0]
        if self.first_sensor_update:
            self._init_sensor_buffers(num_rays)

        fn = self._sensor_model_dispatch.get(self.RANGELIB_VAR)
        if fn is None:
            self.get_logger().error(f'Unknown rangelib_variant {self.RANGELIB_VAR}; set 0–4')
            return
        fn(proposal_dist, obs, weights, num_rays)

    def MCL(self, action, obs):
        """One MCL step: resample → motion model → sensor model → normalise."""
        # E-1: systematic (low-variance) resampling; P-4: copy into pre-allocated buffer
        proposal_indices = self._systematic_resample(self.weights)
        np.take(self.particles, proposal_indices, axis=0, out=self._proposal)
        self.motion_model(self._proposal, action)
        self.sensor_model(self._proposal, obs, self.weights)
        self.weights /= np.sum(self.weights)
        np.copyto(self.particles, self._proposal)

        # E-3: effective particle count diagnostic every 10 iters
        self._n_eff_iter += 1
        if self._n_eff_iter % 10 == 0:
            n_eff = 1.0 / np.sum(self.weights ** 2)
            if n_eff < self.MAX_PARTICLES * 0.1:
                self.get_logger().warn(
                    f'N_eff={n_eff:.0f} < {self.MAX_PARTICLES * 0.1:.0f}; '
                    'filter may be degenerate')

    def expected_pose(self):
        # x, y: weighted linear mean. theta: weighted *circular* mean — a linear
        # average is wrong across the ±pi wrap and biased whenever the cloud is
        # spread in heading, which corrupts the published yaw.
        pose = np.empty(3)
        pose[0] = np.dot(self.particles[:, 0], self.weights)
        pose[1] = np.dot(self.particles[:, 1], self.weights)
        pose[2] = np.arctan2(
            np.dot(np.sin(self.particles[:, 2]), self.weights),
            np.dot(np.cos(self.particles[:, 2]), self.weights))
        return pose

    # ── callbacks ─────────────────────────────────────────────────────────────

    def lidarCB(self, msg):
        """Store latest scan; on first message initialise angle/buffer arrays."""
        self.LASER_FRAME_ID = msg.header.frame_id

        if self.BASE_FRAME_ID and self.BASE_FRAME_ID == self.LASER_FRAME_ID:
            self.get_logger().warn(
                f'base_frame_id == laser_frame_id ({self.LASER_FRAME_ID}); '
                'clearing base_frame_id')
            self.BASE_FRAME_ID = ''

        if self.BASE_FRAME_ID:
            self._cache_base_to_scan_tf(
                timestamp=rclpy.time.Time.from_msg(msg.header.stamp))

        # P-8: guard against misconfigured angle_step
        assert len(msg.ranges) > self.ANGLE_STEP, \
            f'angle_step={self.ANGLE_STEP} >= num_ranges={len(msg.ranges)}'

        if not isinstance(self.laser_angles, np.ndarray):
            self.get_logger().info('Received first LiDAR message')
            self.laser_angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
            self.downsampled_angles = self.laser_angles[::self.ANGLE_STEP].astype(np.float32)
            self.viz_queries = np.zeros((self.downsampled_angles.shape[0], 3), dtype=np.float32)
            self.viz_ranges = np.zeros(self.downsampled_angles.shape[0], dtype=np.float32)
            self.get_logger().info(
                f'{self.downsampled_angles.shape[0]} downsampled rays '
                f'(angle_step={self.ANGLE_STEP})')
            # G-8: allocate global-loc buffer now that num_beams is known
            self._global_ranges_buf = np.zeros(
                self.GLOBAL_LOC_MAX_CANDIDATES * self.downsampled_angles.shape[0],
                dtype=np.float32)

        # E-5: read range_min from message; clip readings to valid sensor range.
        # nan_to_num first: np.clip propagates NaN unchanged, which would poison
        # the weight vector or cause undefined behaviour in RangeLibc C++ variants.
        self._range_min = msg.range_min
        self.downsampled_ranges = np.clip(
            np.nan_to_num(
                np.array(msg.ranges[::self.ANGLE_STEP], dtype=np.float32),
                nan=self.MAX_RANGE_METERS,
                posinf=self.MAX_RANGE_METERS,
                neginf=self._range_min,
            ),
            self._range_min,
            self.MAX_RANGE_METERS)

        self._last_scan_stamp = msg.header.stamp  # F-4
        self.lidar_initialized = True
        self._laser_ready = True

        # G-5: upgrade from uniform spread to hybrid global search on first scan
        if self._pending_global_loc:
            self._pending_global_loc = False
            self._hybrid_global_localize(self.downsampled_ranges)

    def odomCB(self, msg):
        """
        Accumulate car-local odometry delta between consecutive messages.
        The MCL timer drives updates at fixed Hz; this callback only accumulates (P-2).
        """
        self.odom_pose = msg.pose.pose
        position = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        orientation = Utils.quaternion_to_angle(msg.pose.pose.orientation)
        pose = np.array([position[0], position[1], orientation])
        self.current_speed = msg.twist.twist.linear.x

        with self._odom_lock:
            if isinstance(self.last_pose, np.ndarray):
                rot = Utils.rotation_matrix(-self.last_pose[2])
                delta = position - self.last_pose[:2]
                local_delta = rot @ delta
                # accumulate so no delta is lost between MCL timer ticks
                self.odometry_data[0] += local_delta[0]
                self.odometry_data[1] += local_delta[1]
                # Fix 7: angle_diff handles ±pi wrap (raw difference can be ±2pi)
                self.odometry_data[2] += Utils.angle_diff(orientation, self.last_pose[2])
                self.last_pose = pose
                self.last_stamp = msg.header.stamp
                self._last_odom_stamp = msg.header.stamp
                self.odom_initialized = True
            else:
                self.get_logger().info('Received first Odometry message')
                self.last_pose = pose

    def clicked_pose(self, msg):
        """RViz 2D Pose Estimate → init near pose; point click → global search."""
        if isinstance(msg, PointStamped):
            self._reset_particles()
        elif isinstance(msg, PoseWithCovarianceStamped):
            self._reset_particles(msg.pose.pose)

    # ── MCL timer callback (P-2) ──────────────────────────────────────────────

    def _mcl_timer_cb(self):
        """Drive MCL at fixed Hz; guard against missing data (F-3) and stale scan (F-4)."""
        if not (self._map_ready and self._laser_ready and self._pose_inited
                and self.odom_initialized):
            if not self._warned_not_ready:
                missing = [name for name, ready in [
                    ('map', self._map_ready),
                    ('laser', self._laser_ready),
                    ('pose_init', self._pose_inited),
                    ('odom', self.odom_initialized)] if not ready]
                self.get_logger().warn(f'MCL not ready: waiting for {missing}')
                self._warned_not_ready = True
            return
        self._warned_not_ready = False

        # F-4: skip cycle if scan is stale
        if self._last_scan_stamp is not None:
            age = (self.get_clock().now()
                   - rclpy.time.Time.from_msg(self._last_scan_stamp)).nanoseconds * 1e-9
            if age > self.SCAN_MAX_AGE_SEC:
                self.get_logger().warn(
                    f'Stale scan ({age:.2f}s > {self.SCAN_MAX_AGE_SEC}s); skipping MCL')
                return

        # skip cycle if odom is stale (publisher died after init)
        if self._last_odom_stamp is not None:
            age = (self.get_clock().now()
                   - rclpy.time.Time.from_msg(self._last_odom_stamp)).nanoseconds * 1e-9
            if age > self.ODOM_MAX_AGE_SEC:
                self.get_logger().warn(
                    f'Stale odom ({age:.2f}s > {self.ODOM_MAX_AGE_SEC}s); skipping MCL')
                return

        self.update()

    # ── update loop ───────────────────────────────────────────────────────────

    def update(self):
        """Run one MCL cycle; guards are in _mcl_timer_cb."""
        if self.state_lock.locked():
            self.get_logger().warn('MCL update skipped: state lock held')
            return

        with self.state_lock:
            observation = np.copy(self.downsampled_ranges).astype(np.float32)
            with self._odom_lock:
                action = np.copy(self.odometry_data)
                # Motion gate: only consume the accumulated delta (and run MCL) once
                # the robot has moved past update_min_d / update_min_a. Otherwise keep
                # accumulating so no motion is lost, and skip resample+noise this tick.
                # Force the first cycle so inferred_pose / TF get established at startup.
                moved = (abs(action[0]) > self.UPDATE_MIN_D
                         or abs(action[1]) > self.UPDATE_MIN_D
                         or abs(action[2]) > self.UPDATE_MIN_A
                         or self.inferred_pose is None)
                if moved:
                    self.odometry_data[:] = 0.0
                last_stamp = self.last_stamp

            if moved:
                self.timer.tick()
                self.iters += 1
                t1 = time.time()
                self.MCL(action, observation)
                self.inferred_pose = self.expected_pose()
                t2 = time.time()

        # Publish TF at the odom rate, not the MCL timer rate: broadcast whenever the
        # odom stamp advances (so a 40 Hz timer over a 30 Hz odom does not re-send a
        # stale stamp → TF_REPEATED_DATA). A heartbeat re-stamps the last transform if
        # too long passes with no fresh odom, keeping map→odom from expiring while odom
        # is slow but alive (the odom-staleness guard in the timer cb still stops
        # publishing once odom dies). The odom topic is published only on a real fix.
        if self.inferred_pose is not None:
            now_time = self.get_clock().now()
            stamp_changed = (last_stamp is not None
                             and last_stamp != self._last_published_stamp)
            heartbeat_due = (
                self.TF_HEARTBEAT_PERIOD > 0.0
                and (self._last_tf_pub_time is None
                     or (now_time - self._last_tf_pub_time).nanoseconds * 1e-9
                     >= self.TF_HEARTBEAT_PERIOD))
            if stamp_changed or heartbeat_due:
                # On a pure heartbeat (no new odom) re-stamp with the current time so
                # the forward-dated transform keeps advancing; otherwise use the odom
                # stamp the pose is associated with.
                pub_stamp = last_stamp if stamp_changed else now_time.to_msg()
                self.publish_tf(self.inferred_pose, pub_stamp,
                                publish_odom_topic=moved)
                self._last_published_stamp = last_stamp
                self._last_tf_pub_time = now_time

        if moved:
            ips = 1.0 / (t2 - t1)
            self.smoothing.append(ips)
            if self.iters % 10 == 0:
                self.get_logger().info(
                    f'MCL iters/s: {int(self.timer.fps())} '
                    f'(possible: {int(self.smoothing.mean())})')

        self.visualize()

    # ── TF / pose publishing ──────────────────────────────────────────────────

    def _anchor_to_odom_origin(self, T_map_base):
        """
        Express a map→base_link pose relative to the odom origin (comparison aid).

        Freezes the map→odom correction C = T_map_base · inv(T_odom_base) at the
        first valid fix, then returns inv(C) · T_map_base. At t0 this equals
        T_odom_base exactly, so the published pose starts coincident with raw
        odometry and afterwards diverges only by the drift the PF corrects.
        C is held constant after the first call (a static rigid alignment, not a
        feedback loop). Returns the input unchanged if odom is not available yet.
        """
        if self._odom_anchor_inv is None:
            if self.odom_pose is None:
                return T_map_base  # odom not ready; publish unanchored this once
            T_odom_base0 = pose_msg_to_matrix(self.odom_pose)
            # Analytical SE(3) inverse of the frozen T_map_base (R^T | -R^T·t);
            # inv(C) = T_odom_base0 · inv(T_map_base0) = T_odom_map at t0.
            _R = T_map_base[:3, :3]
            inv_T_map_base0 = np.eye(4)
            inv_T_map_base0[:3, :3] = _R.T
            inv_T_map_base0[:3, 3] = -(_R.T @ T_map_base[:3, 3])
            self._odom_anchor_inv = T_odom_base0 @ inv_T_map_base0
            self.get_logger().info(
                'anchor_to_odom_origin: froze map→odom correction; pose now '
                'expressed relative to the odom origin')
        return self._odom_anchor_inv @ T_map_base

    def publish_tf(self, pose, stamp=None, publish_odom_topic=True):
        """
        Publish the localisation result as a TF transform and (optionally) Odometry.

        Three modes determined by parameters:
          ODOM_FRAME_ID set  → map → odom   (relies on odom→base_link from VESC/DBW)
          ODOM_FRAME_ID ''   → map → base_link  (BASE_FRAME_ID set)
          both ''            → map → laser  (fallback; non-standard, breaks most nav stacks)

        For the map→odom mode, T_odom_scan is built from the latest odometry message pose
        (T_odom_base) composed with the cached static T_base_scan, avoiding a dynamic
        TF lookup on every cycle.
        """
        if stamp is None:
            stamp = self.get_clock().now()
        else:
            stamp = rclpy.time.Time.from_msg(stamp)

        # Fix 2: skip np.cov when nobody is subscribed to the odom topic, or when this
        # call is not publishing the odom topic (heartbeat / non-fix republish)
        if (publish_odom_topic and self.PUBLISH_ODOM
                and self.odom_pub.get_subscription_count() > 0):
            self.cov_3x3 = np.cov(self.particles, rowvar=False, ddof=0, aweights=self.weights)

        q_ms = quaternion_from_euler(0.0, 0.0, pose[2])
        self.T_map_to_scan = quat_trans_to_matrix(
            q_ms, [pose[0], pose[1], 0.0],
            dtype=np.float64, homogenous_matrix=self.T_map_to_scan)

        has_base = bool(self.BASE_FRAME_ID and self.BASE_FRAME_ID != self.LASER_FRAME_ID)
        has_odom = bool(self.ODOM_FRAME_ID and self.ODOM_FRAME_ID != self.GLOBAL_FRAME_ID)

        if has_odom:
            # ── map → odom mode ───────────────────────────────────────────────
            if self.odom_pose is None:
                self.get_logger().warn('No odometry message yet; skipping TF publish')
                return

            T_odom_base = pose_msg_to_matrix(self.odom_pose)

            if has_base:
                if self.base_frame_to_scan_tf is None:
                    self.get_logger().warn('base→scan TF not cached yet; skipping TF publish')
                    return
                T_odom_scan = T_odom_base @ self.base_frame_to_scan_tf
            else:
                T_odom_scan = T_odom_base

            # Fix 3: guard for non-finite values (NaN/Inf from bad odom/scan data);
            # np.linalg.inv propagates NaN silently rather than raising LinAlgError,
            # so an explicit isfinite check is the correct guard for both cases.
            if not np.isfinite(T_odom_scan).all():
                self.get_logger().error('T_odom_scan contains non-finite values; skipping TF publish')
                return
            # Analytical SE(3) inverse: T=[R|t] → T_inv=[R^T | -R^T@t].
            # Valid for any finite rigid-body transform; no LAPACK call needed.
            _R = T_odom_scan[:3, :3]
            self._T_odom_scan_inv[:3, :3] = _R.T
            self._T_odom_scan_inv[:3, 3] = -(_R.T @ T_odom_scan[:3, 3])
            T_publish = self.T_map_to_scan @ self._T_odom_scan_inv
            publish_child = self.ODOM_FRAME_ID

        elif has_base:
            # ── map → base_link mode ──────────────────────────────────────────
            if self.laser_to_base_frame_tf is None:
                self.get_logger().warn('laser→base TF not cached yet; skipping TF publish')
                return
            T_publish = self.T_map_to_scan @ self.laser_to_base_frame_tf
            publish_child = self.BASE_FRAME_ID
            if self.ANCHOR_TO_ODOM_ORIGIN:
                T_publish = self._anchor_to_odom_origin(T_publish)

        else:
            # ── map → laser fallback ──────────────────────────────────────────
            T_publish = self.T_map_to_scan
            publish_child = self.LASER_FRAME_ID

        if self.tf_broadcast:
            stamp_fwd = (stamp + rclpy.duration.Duration(
                seconds=self.TRANSFORM_TOLERANCE)).to_msg()
            self.pub_tf.sendTransform(
                matrix_to_transformstamped(
                    T_publish, self.GLOBAL_FRAME_ID, publish_child, stamp_fwd))

        if publish_odom_topic and self.PUBLISH_ODOM:
            quat = quaternion_from_matrix(T_publish[:3, :3])
            odom = Odometry()
            odom.header.stamp = stamp.to_msg()
            odom.header.frame_id = self.GLOBAL_FRAME_ID
            odom.child_frame_id = publish_child
            odom.pose.pose.position.x = float(T_publish[0, 3])
            odom.pose.pose.position.y = float(T_publish[1, 3])
            odom.pose.pose.position.z = float(T_publish[2, 3])
            odom.pose.pose.orientation = Quaternion(
                x=float(quat[0]), y=float(quat[1]),
                z=float(quat[2]), w=float(quat[3]))
            # Map 3×3 (x, y, θ) particle covariance → 6×6 ROS pose covariance
            # Row/col order: [x, y, z, roll, pitch, yaw]; θ lives at index 5.
            cov_6x6 = np.zeros((6, 6))
            c = self.cov_3x3
            cov_6x6[0, 0] = c[0, 0]; cov_6x6[0, 1] = c[0, 1]; cov_6x6[0, 5] = c[0, 2]
            cov_6x6[1, 0] = c[1, 0]; cov_6x6[1, 1] = c[1, 1]; cov_6x6[1, 5] = c[1, 2]
            cov_6x6[5, 0] = c[2, 0]; cov_6x6[5, 1] = c[2, 1]; cov_6x6[5, 5] = c[2, 2]
            odom.pose.covariance = cov_6x6.flatten().tolist()
            odom.twist.twist.linear.x = self.current_speed
            self.odom_pub.publish(odom)

    # ── Task 9: hybrid global localization ───────────────────────────────────

    def _build_candidate_poses(self):
        """G-1: sample permissible cells at coarse_res spacing × theta orientations."""
        step_px = max(1, int(self.GLOBAL_LOC_COARSE_RES / self.map_info.resolution))
        while True:
            ys, xs = np.where(self.permissible_region)
            mask = (ys % step_px == 0) & (xs % step_px == 0)
            cells = np.stack([xs[mask], ys[mask]], axis=1)
            n_theta = max(1, int(360.0 / self.GLOBAL_LOC_THETA_RES))
            thetas = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
            n_candidates = len(cells) * n_theta
            if n_candidates <= self.GLOBAL_LOC_MAX_CANDIDATES:
                break
            self.get_logger().warn(
                f'Global loc: {n_candidates} candidates exceed '
                f'global_loc_max_candidates={self.GLOBAL_LOC_MAX_CANDIDATES}; '
                f'doubling step_px to {step_px * 2}')
            step_px *= 2

        candidates = np.repeat(cells, n_theta, axis=0).astype(np.float64)
        candidates = np.column_stack([candidates, np.tile(thetas, len(cells))])
        Utils.map_to_world(candidates, self.map_info)
        return candidates

    def _score_candidates(self, candidates, scan_ranges):
        """G-2: ray-cast all candidates with variant-1 approach, sum sensor model scores."""
        num_beams = len(self.downsampled_angles)
        n = len(candidates)
        queries = np.zeros((n * num_beams, 3), dtype=np.float32)
        queries[:, 0] = np.repeat(candidates[:, 0], num_beams).astype(np.float32)
        queries[:, 1] = np.repeat(candidates[:, 1], num_beams).astype(np.float32)
        queries[:, 2] = (np.repeat(candidates[:, 2], num_beams).astype(np.float32)
                         + np.tile(self.downsampled_angles, n))
        buf = self._global_ranges_buf[:n * num_beams]
        self.range_method.calc_range_many(queries, buf)

        table_width = self.sensor_model_table.shape[0]
        obs_px = np.clip(
            (np.tile(scan_ranges, n) / self.map_info.resolution).astype(int),
            0, table_width - 1)
        rng_px = np.clip(buf.astype(int), 0, table_width - 1)
        scores = self.sensor_model_table[obs_px, rng_px]
        return scores.reshape(n, num_beams).sum(axis=1)

    def _pick_top_k(self, candidates, scores):
        """G-3: greedy NMS — accept highest-scoring candidates spaced > min_dist apart."""
        order = np.argsort(scores)[::-1]
        chosen = []
        for idx in order:
            pos = candidates[idx, :2]
            if all(np.linalg.norm(pos - candidates[c, :2]) > self.GLOBAL_LOC_MIN_DIST
                   for c in chosen):
                chosen.append(idx)
            if len(chosen) >= self.GLOBAL_LOC_TOP_K:
                break
        return candidates[chosen]

    def _hybrid_global_localize(self, scan_ranges):
        """G-4: coarse search → score → NMS → seed particles around top-K hypotheses."""
        self.get_logger().info('Global localization: coarse search starting')
        t0 = time.time()
        candidates = self._build_candidate_poses()
        scores = self._score_candidates(candidates, scan_ranges)
        top_k = self._pick_top_k(candidates, scores)
        self.get_logger().info(
            f'Global localization: {len(candidates)} candidates scored in '
            f'{time.time() - t0:.2f}s; seeding around {len(top_k)} hypothesis(es)')

        n_each = self.MAX_PARTICLES // len(top_k)
        theta_sigma = float(np.radians(self.GLOBAL_LOC_THETA_RES)) / 2.0
        xy_sigma = self.GLOBAL_LOC_COARSE_RES / 2.0
        with self.state_lock:
            parts = []
            for hyp in top_k:
                noise = np.random.randn(n_each, 3) * [xy_sigma, xy_sigma, theta_sigma]
                parts.append(hyp + noise)
            self.particles = np.vstack(parts)[:self.MAX_PARTICLES]
            self.weights[:] = 1.0 / self.MAX_PARTICLES
        self._pose_inited = True

    def _global_localize_srv_cb(self, _req, resp):
        """G-6: service handler — trigger hybrid search on demand."""
        if self._laser_ready:
            self._hybrid_global_localize(self.downsampled_ranges)
        else:
            self.get_logger().warn('Global localize called before first scan received')
        return resp

    def _startup_global_loc_cb(self):
        """G-7: one-shot timer — auto-trigger global search if no pose init yet."""
        self._startup_timer.cancel()
        if not self._pose_inited:
            self.get_logger().info(
                f'No initial pose received after {self.GLOBAL_LOC_TIMEOUT:.1f}s; '
                'triggering automatic global localization')
            self._reset_particles(pose=None)

    # ── visualisation ─────────────────────────────────────────────────────────

    def visualize(self, stamp=None):
        """Publish inferred pose, particle cloud, and simulated scan for RViz."""
        if not self.DO_VIZ:
            return
        # P-6: only publish every VIZ_THROTTLE MCL cycles (default ~10 Hz at 40 Hz MCL)
        self._viz_counter = (self._viz_counter + 1) % self.VIZ_THROTTLE
        if self._viz_counter != 0:
            return

        stamp = self._get_stamp(stamp)

        if (self.pose_pub.get_subscription_count() > 0
                and isinstance(self.inferred_pose, np.ndarray)):
            ps = PoseStamped()
            ps.header.stamp = stamp
            ps.header.frame_id = self.GLOBAL_FRAME_ID
            ps.pose.position.x = self.inferred_pose[0]
            ps.pose.position.y = self.inferred_pose[1]
            ps.pose.orientation = Utils.angle_to_quaternion(self.inferred_pose[2])
            self.pose_pub.publish(ps)

        if self.particle_pub.get_subscription_count() > 0:
            if self.MAX_PARTICLES > self.MAX_VIZ_PARTICLES:
                idx = np.random.choice(
                    self.particle_indices, self.MAX_VIZ_PARTICLES, p=self.weights)
                self._publish_particles(self.particles[idx], stamp)
            else:
                self._publish_particles(self.particles, stamp)

        if (self.pub_fake_scan.get_subscription_count() > 0
                and isinstance(self.ranges, np.ndarray)):
            self.viz_queries[:, 0] = self.inferred_pose[0]
            self.viz_queries[:, 1] = self.inferred_pose[1]
            self.viz_queries[:, 2] = self.downsampled_angles + self.inferred_pose[2]
            self.range_method.calc_range_many(self.viz_queries, self.viz_ranges)
            self._publish_scan(self.downsampled_angles, self.viz_ranges, stamp=self.last_stamp)

    def _publish_particles(self, particles, stamp=None):
        stamp = self._get_stamp(stamp)
        pa = PoseArray()
        pa.header.stamp = stamp
        pa.header.frame_id = self.GLOBAL_FRAME_ID
        pa.poses = Utils.particles_to_poses(particles)
        self.particle_pub.publish(pa)

    def _publish_scan(self, angles, ranges, stamp=None):
        stamp = self._get_stamp(stamp)
        ls = LaserScan()
        ls.header.stamp = stamp
        ls.header.frame_id = self.LASER_FRAME_ID
        ls.angle_min = float(np.min(angles))
        ls.angle_max = float(np.max(angles))
        ls.angle_increment = float(np.abs(angles[0] - angles[1]))
        ls.range_min = float(self._range_min)  # E-5: from actual scan message
        ls.range_max = float(np.max(ranges))
        ls.ranges = ranges.tolist()
        self.pub_fake_scan.publish(ls)


def main(args=None):
    rclpy.init(args=args)
    pf = ParticleFilter()
    # Fix 1: SingleThreadedExecutor sleeps at OS level (epoll) when idle;
    # MultiThreadedExecutor(4) with GIL has threads polling in tight loops consuming ~300% CPU.
    executor = SingleThreadedExecutor()
    executor.add_node(pf)
    executor.spin()


if __name__ == '__main__':
    main()