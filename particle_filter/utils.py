import time
import math
from typing import Optional

import numpy as np

try:
    import scipy
    from scipy.spatial.transform import Rotation as R
    SCIPY_INSTALLED = True
    SCIPY_VERSION = scipy.__version__
except ImportError:
    SCIPY_INSTALLED = False
    SCIPY_VERSION = '0.0.0'

# Fix 5: cache once at import time; avoids repeated string parsing in hot-path functions
# (quaternion_to_angle is called in odomCB at 100+ Hz).
# True  → scipy >= 1.14: use scalar_first=False kwarg
# False → scipy <  1.14: omit kwarg (not yet supported)
_SCIPY_SCALAR_FIRST = SCIPY_INSTALLED and int(SCIPY_VERSION.split('.')[1]) >= 14

import tf2_ros

try:
    import tf_transformations
    from tf_transformations import quaternion_matrix
    TF_TRANSFORMATIONS_INSTALLED = True
except ImportError:
    TF_TRANSFORMATIONS_INSTALLED = False

from std_msgs.msg import Header
from visualization_msgs.msg import Marker
from geometry_msgs.msg import (Point, Pose, PoseStamped, TransformStamped, PoseArray, Quaternion, PolygonStamped,
                               Polygon, Point32, PoseWithCovarianceStamped, PointStamped)


class CircularArray(object):
    """ Simple implementation of a circular array.
        You can append to it any number of times but only "size" items will be kept
    """
    def __init__(self, size):
        self.arr = np.zeros(size)
        self.ind = 0
        self.num_els = 0

    def append(self, value):
        if self.num_els < self.arr.shape[0]:
            self.num_els += 1
        self.arr[self.ind] = value
        self.ind = (self.ind + 1) % self.arr.shape[0]

    def mean(self):
        return np.mean(self.arr[:self.num_els])

    def median(self):
        return np.median(self.arr[:self.num_els])


class Timer:
    """ Simple helper class to compute the rate at which something is called.
        
        "smoothing" determines the size of the underlying circular array, which averages
        out variations in call rate over time.

        use timer.tick() to record an event
        use timer.fps() to report the average event rate.
    """
    def __init__(self, smoothing):
        self.arr = CircularArray(smoothing)
        self.last_time = time.time()

    def tick(self):
        t = time.time()
        self.arr.append(1.0 / (t - self.last_time))
        self.last_time = t

    def fps(self):
        return self.arr.mean()

def quaternion_from_euler(roll, pitch, yaw):
    if SCIPY_INSTALLED:
        if _SCIPY_SCALAR_FIRST:
            return R.from_euler('xyz', [roll, pitch, yaw]).as_quat(scalar_first=False)
        else:
            return R.from_euler('xyz', [roll, pitch, yaw]).as_quat()  # x, y, z, w
    return tf_transformations.quaternion_from_euler(roll, pitch, yaw)

def quaternion_from_matrix(mat):
    if SCIPY_INSTALLED:
        if _SCIPY_SCALAR_FIRST:
            return R.from_matrix(mat).as_quat(scalar_first=False)
        else:
            return R.from_matrix(mat).as_quat()  # x, y, z, w
    if mat.shape != (4, 4):
        mat = np.eye(4, dtype=np.float64)  # dtype=mat.dtype
        mat[:3, :3] = mat
    return tf_transformations.quaternion_from_matrix(mat)

def angle_to_quaternion(angle):
    """Convert an angle in radians into a quaternion _message_."""
    q = quaternion_from_euler(0, 0, angle)
    q_out = Quaternion()
    q_out.x = q[0]
    q_out.y = q[1]
    q_out.z = q[2]
    q_out.w = q[3]
    return q_out

def quaternion_to_angle(q):
    """Convert a quaternion _message_ into an angle in radians (yaw / rotation about Z)."""
    x, y, z, w = q.x, q.y, q.z, q.w
    if SCIPY_INSTALLED:
        if _SCIPY_SCALAR_FIRST:
            return R.from_quat([x, y, z, w], scalar_first=False).as_euler('xyz')[2]
        else:
            return R.from_quat([x, y, z, w]).as_euler('xyz')[2]
    return tf_transformations.euler_from_quaternion((x, y, z, w))[2]

def rotation_matrix(theta):
    """Return a (2, 2) ndarray rotation matrix for the given angle in radians."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])

def transpose_laser_to_base_link_xy(pose, laser_base_link_offset):
    yaw = quaternion_to_angle(pose.orientation)
    pose.position.x = pose.position.x - laser_base_link_offset[0] * math.cos(yaw)
    pose.position.y = pose.position.y - laser_base_link_offset[0] * math.sin(yaw)
    return pose


def particle_to_pose(particle, laser_base_link_offset=None):
    """ Converts a particle in the form [x, y, theta] into a Pose object """
    pose = Pose()
    pose.position.x = particle[0]
    pose.position.y = particle[1]
    pose.orientation = angle_to_quaternion(particle[2])
    if laser_base_link_offset:
        pose = transpose_laser_to_base_link_xy(pose, laser_base_link_offset)
    return pose


def particles_to_poses(particles, laser_base_link_offset=None):
    """ Converts a two dimensional array of particles into an array of Poses.
        Particles can be a array like [[x0, y0, theta0], [x1, y1, theta1]...]
    """
    return list(map(particle_to_pose, particles))


def map_to_world_slow(x, y, t, map_info):
    """ Converts given (x,y,t) coordinates from the coordinate space of the map (pixels) into world coordinates (meters).
        Provide the MapMetaData object from a map message to specify the change in coordinates.
        *** Logical, but slow implementation, when you need a lot of coordinate conversions, use the map_to_world function
    """
    scale = map_info.resolution
    angle = quaternion_to_angle(map_info.origin.orientation)
    rot = rotation_matrix(angle)
    trans = np.array([[map_info.origin.position.x],
                      [map_info.origin.position.y]])

    map_c = np.array([[x],
                      [y]])
    world = (rot*map_c) * scale + trans

    return world[0, 0], world[1, 0], t+angle


def map_to_world(poses, map_info):
    """ Takes a two dimensional numpy array of poses:
            [[x0,y0,theta0],
             [x1,y1,theta1],
             [x2,y2,theta2],
                   ...     ]
        And converts them from map coordinate space (pixels) to world coordinate space (meters).
        - Conversion is done in place, so this function does not return anything.
        - Provide the MapMetaData object from a map message to specify the change in coordinates.
        - This implements the same computation as map_to_world_slow but vectorized and inlined
    """

    scale = map_info.resolution
    angle = quaternion_to_angle(map_info.origin.orientation)

    # rotation
    c, s = np.cos(angle), np.sin(angle)
    # we need to store the x coordinates since they will be overwritten
    temp = np.copy(poses[:, 0])
    poses[:, 0] = c*poses[:, 0] - s*poses[:, 1]
    poses[:, 1] = s*temp       + c*poses[:, 1]

    # scale
    poses[:, :2] *= float(scale)

    # translate
    poses[:, 0] += map_info.origin.position.x
    poses[:, 1] += map_info.origin.position.y
    poses[:, 2] += angle

def world_to_map(poses, map_info):
    """ Takes a two dimensional numpy array of poses:
            [[x0,y0,theta0],
             [x1,y1,theta1],
             [x2,y2,theta2],
                   ...     ]
        And converts them from world coordinate space (meters) to world coordinate space (pixels).
        - Conversion is done in place, so this function does not return anything.
        - Provide the MapMetaData object from a map message to specify the change in coordinates.
        - This implements the same computation as world_to_map_slow but vectorized and inlined
        - You may have to transpose the returned x and y coordinates to directly index a pixel array
    """
    scale = map_info.resolution
    angle = -quaternion_to_angle(map_info.origin.orientation)

    # translation
    poses[:, 0] -= map_info.origin.position.x
    poses[:, 1] -= map_info.origin.position.y

    # scale
    poses[:, :2] *= (1.0/float(scale))

    # rotation
    c, s = np.cos(angle), np.sin(angle)
    # we need to store the x coordinates since they will be overwritten
    temp = np.copy(poses[:, 0])
    poses[:, 0] = c*poses[:, 0] - s*poses[:, 1]
    poses[:, 1] = s*temp       + c*poses[:, 1]
    poses[:, 2] += angle


def world_to_map_slow(x, y, t, map_info):
    """ Converts given (x,y,t) coordinates from the coordinate space of the world (meters) into map coordinates (pixels).
        Provide the MapMetaData object from a map message to specify the change in coordinates.
        *** Logical, but slow implementation, when you need a lot of coordinate conversions, use the world_to_map function
    """
    scale = map_info.resolution
    angle = quaternion_to_angle(map_info.origin.orientation)
    rot = rotation_matrix(-angle)
    trans = np.array([[map_info.origin.position.x],
                      [map_info.origin.position.y]])

    world = np.array([[x],
                      [y]])
    map_c = rot*((world - trans) / float(scale))
    return map_c[0, 0], map_c[1, 0], t-angle

# ##########################
def normalize(z):
    """Normalizes an angle to between [-pi, pi]"""
    if -np.pi <= z <= np.pi:
        return z
    return np.arctan2(np.sin(z), np.cos(z))

def angle_diff(a, b):
    """Computes the shortest distance between two angles"""
    a = normalize(a)
    b = normalize(b)
    d1 = a - b
    d2 = 2 * np.pi - np.abs(d1)
    if(d1 > 0):
        d2 *= -1.0
    if(np.abs(d1) < np.abs(d2)):
        return d1
    else:
        return d2


def quat_trans_to_matrix(quat: list | np.ndarray, trans: list | np.ndarray, dtype=np.float64,
                         homogenous_matrix: Optional[np.ndarray] = None):
    """
    Converts a quaternion (x,y,z,w) and translation (x,y,z) into a 4x4 numpy matrix.
    If scipy is installed, it will use scipy.spatial.transform.Rotation to create the rotation matrix.
    Otherwise, it will use tf_transformations to create the rotation matrix.

    Args:
        quat (np.ndarray): Quaternion (x,y,z,w)
        trans (np.ndarray): Translation (x,y,z)
    """
    if homogenous_matrix is None:
        homogenous_matrix = np.eye(4, dtype=dtype)

    if SCIPY_INSTALLED:
        if _SCIPY_SCALAR_FIRST:
            rotation_object = R.from_quat(quat, scalar_first=False)
        else:
            rotation_object = R.from_quat(quat)  # x, y, z, w

        homogenous_matrix[:3, :3] = rotation_object.as_matrix()
        homogenous_matrix[:3, 3] = trans

    elif TF_TRANSFORMATIONS_INSTALLED:
        homogenous_matrix = quaternion_matrix(quat)
        homogenous_matrix[:3, 3] = trans

    else:
        raise ValueError("scipy or tf_transformations must be installed to use this function")

    # homogenous_matrix = tf_transformations.concatenate_matrices(
    #             tf_transformations.translation_matrix(trans),
    #             tf_transformations.quaternion_matrix(quat)
    #     )
    return homogenous_matrix

def pose_msg_to_matrix(pose_msg):
    """Converts a geometry_msgs/Pose into a 4x4 numpy matrix."""
    q = pose_msg.orientation
    t = pose_msg.position
    mat = quat_trans_to_matrix([q.x, q.y, q.z, q.w], [t.x, t.y, t.z])
    return mat

def transformstamped_to_matrix(tr: TransformStamped):
    """Converts a geometry_msgs/TransformStamped into a 4x4 numpy matrix. todo: remove since its a duplicate of pose_msg_to_matrix"""
    q = tr.transform.rotation
    t = tr.transform.translation
    mat = quat_trans_to_matrix([q.x, q.y, q.z, q.w], [t.x, t.y, t.z])
    return mat

def matrix_to_transformstamped(mat: np.ndarray, parent_frame: str, child_frame: str, stamp):
    """Converts a 4x4 numpy matrix into a geometry_msgs/TransformStamped"""
    tr = TransformStamped()
    tr.header.stamp = stamp
    tr.header.frame_id = parent_frame
    tr.child_frame_id = child_frame
    t = mat[:3, 3]
    rot_mat = mat[:3, :3]
    quat = quaternion_from_matrix(rot_mat)
    tr.transform.translation.x = float(t[0])
    tr.transform.translation.y = float(t[1])
    tr.transform.translation.z = float(t[2])
    tr.transform.rotation.x = float(quat[0])
    tr.transform.rotation.y = float(quat[1])
    tr.transform.rotation.z = float(quat[2])
    tr.transform.rotation.w = float(quat[3])
    return tr

def adjoint_from_matrix(T: np.ndarray):
    """Computes the adjoint of a 4x4 numpy matrix"""
    Rmat = T[:3, :3]
    t = T[:3, 3]
    # skew(t)
    t_skew = np.array([[0, -t[2], t[1]],
                       [t[2], 0, -t[0]],
                       [-t[1], t[0], 0]], dtype=np.float64)
    Ad = np.zeros((6, 6), dtype=np.float64)
    Ad[:3, :3] = Rmat
    Ad[3:, 3:] = Rmat
    Ad[:3, 3:] = (t_skew @ Rmat)
    return Ad

def cov_list_to_mat(cov_list):
    """Converts a list of 36 covariance values into a 6x6 numpy matrix"""
    cov = np.array(cov_list, dtype=np.float64).reshape((6, 6))
    return cov

def cov_mat_to_list(cov_mat):
    """Converts a 6x6 numpy matrix into a list of 36 covariance values"""
    return cov_mat.reshape(-1).astype(float).tolist()