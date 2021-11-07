#!/usr/bin/env python

import rospy
import numpy as np

from std_msgs.msg import Header
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, Pose, PoseStamped, PoseArray, Quaternion, PolygonStamped, Polygon, Point32, PoseWithCovarianceStamped, PointStamped
import tf.transformations
import tf
import matplotlib.pyplot as plt
import time


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


def angle_to_quaternion(angle):
  """Convert an angle in radians into a quaternion _message_."""
  return Quaternion(*tf.transformations.quaternion_from_euler(0, 0, angle))


def quaternion_to_angle(q):
  """Convert a quaternion _message_ into an angle in radians.
    The angle represents the yaw.
    This is not just the z component of the quaternion."""
  x, y, z, w = q.x, q.y, q.z, q.w
  roll, pitch, yaw = tf.transformations.euler_from_quaternion((x, y, z, w))
  return yaw


def rotation_matrix(theta):
  ''' Creates a rotation matrix for the given angle in radians '''
  c, s = np.cos(theta), np.sin(theta)
  return np.matrix([[c, -s], [s, c]])


def particle_to_pose(particle):
  ''' Converts a particle in the form [x, y, theta] into a Pose object '''
  pose = Pose()
  pose.position.x = particle[0]
  pose.position.y = particle[1]
  pose.orientation = angle_to_quaternion(particle[2])
  return pose


def particles_to_poses(particles):
  ''' Converts a two dimensional array of particles into an array of Poses. 
        Particles can be a array like [[x0, y0, theta0], [x1, y1, theta1]...]
    '''
  return list(map(particle_to_pose, particles))


def make_header(frame_id, stamp=None):
  ''' Creates a Header object for stamped ROS objects '''
  if stamp == None:
    stamp = rospy.Time.now()
  header = Header()
  header.stamp = stamp
  header.frame_id = frame_id
  return header


def map_to_world_slow(x, y, t, map_info):
  ''' Converts given (x,y,t) coordinates from the coordinate space of the map (pixels) into world coordinates (meters).
        Provide the MapMetaData object from a map message to specify the change in coordinates.
        *** Logical, but slow implementation, when you need a lot of coordinate conversions, use the map_to_world function
    '''
  scale = map_info.resolution
  angle = quaternion_to_angle(map_info.origin.orientation)
  rot = rotation_matrix(angle)
  trans = np.array([[map_info.origin.position.x], [map_info.origin.position.y]])

  map_c = np.array([[x], [y]])
  world = (rot * map_c) * scale + trans

  return world[0, 0], world[1, 0], t + angle


def map_to_world(poses, map_info):
  ''' Takes a two dimensional numpy array of poses:
            [[x0,y0,theta0],
             [x1,y1,theta1],
             [x2,y2,theta2],
                   ...     ]
        And converts them from map coordinate space (pixels) to world coordinate space (meters).
        - Conversion is done in place, so this function does not return anything.
        - Provide the MapMetaData object from a map message to specify the change in coordinates.
        - This implements the same computation as map_to_world_slow but vectorized and inlined
    '''

  scale = map_info.resolution
  angle = quaternion_to_angle(map_info.origin.orientation)

  # rotation
  c, s = np.cos(angle), np.sin(angle)
  # we need to store the x coordinates since they will be overwritten
  temp = np.copy(poses[:, 0])
  poses[:, 0] = c * poses[:, 0] - s * poses[:, 1]
  poses[:, 1] = s * temp + c * poses[:, 1]

  # scale
  poses[:, :2] *= float(scale)

  # translate
  poses[:, 0] += map_info.origin.position.x
  poses[:, 1] += map_info.origin.position.y
  poses[:, 2] += angle


def world_to_map(poses, map_info):
  ''' Takes a two dimensional numpy array of poses:
            [[x0,y0,theta0],
             [x1,y1,theta1],
             [x2,y2,theta2],
                   ...     ]
        And converts them from world coordinate space (meters) to world coordinate space (pixels).
        - Conversion is done in place, so this function does not return anything.
        - Provide the MapMetaData object from a map message to specify the change in coordinates.
        - This implements the same computation as world_to_map_slow but vectorized and inlined
        - You may have to transpose the returned x and y coordinates to directly index a pixel array
    '''
  scale = map_info.resolution
  angle = -quaternion_to_angle(map_info.origin.orientation)

  # translation
  poses[:, 0] -= map_info.origin.position.x
  poses[:, 1] -= map_info.origin.position.y

  # scale
  poses[:, :2] *= (1.0 / float(scale))

  # rotation
  c, s = np.cos(angle), np.sin(angle)
  # we need to store the x coordinates since they will be overwritten
  temp = np.copy(poses[:, 0])
  poses[:, 0] = c * poses[:, 0] - s * poses[:, 1]
  poses[:, 1] = s * temp + c * poses[:, 1]
  poses[:, 2] += angle


def world_to_map_slow(x, y, t, map_info):
  ''' Converts given (x,y,t) coordinates from the coordinate space of the world (meters) into map coordinates (pixels).
        Provide the MapMetaData object from a map message to specify the change in coordinates.
        *** Logical, but slow implementation, when you need a lot of coordinate conversions, use the world_to_map function
    '''
  scale = map_info.resolution
  angle = quaternion_to_angle(map_info.origin.orientation)
  rot = rotation_matrix(-angle)
  trans = np.array([[map_info.origin.position.x], [map_info.origin.position.y]])

  world = np.array([[x], [y]])
  map_c = rot * ((world - trans) / float(scale))
  return map_c[0, 0], map_c[1, 0], t - angle


def read_landmark_csv(filename):
  '''
      Read in the reference landmark csv file that contains N instances with:
      1. landmark type, e.g. door, window
      2. landmark center x position in the map frame, in meters
      3. landmark center y position in the map frame, in meters
      4. landmark orientation in the map frame, in rad
      5. landmark length, in meters.

      Return an numpy array of size (N, 4) of (x_c, y_c, theta, length)
      TODO(zhihao): complete this function.
  '''
  return None


def seeable(landmarks, poses, ranges, max_distance, FOV):
  '''
    Check if landmarks centered at (x_c(m), y_c(m), theta(rad)) can be seen
    from poses (x_p(m), y_p(m), theta_p(rad)) with a camera that has
    1) max seeable distance: max_distance(m); 2) field of view: FOV(deg)
    
    Input:
      landmarks: (N, 4), 2nd dimension is (x_c, y_c, theta, length)
      of line-shaped reference landmarks, e.g. doors. These reference landmarks
      should be obtained from the floormap.

      poses: (N, 3), particles.

      ranges: lidar scan range from the particle position to the landmark center.
      This decides whether the landmark is blocked by wall/obstacles.

      TODO(shumin): currently whether the landmark is blocked is determined by checking 
      the landmark center only. Improve it to check both ends.
  '''
  # Assert input shapes.
  if poses.shape[0] != landmarks.shape[0]:
    rospy.logwarn(
        "Poses should be of shape (N, 3) and landmarks should be of shape (N,4)!"
    )
    return

  # Calculate the left and right corners of the landmarks.
  Xc, Yc, Theta, L =  landmarks[:, 0].reshape(-1,1), \
                      landmarks[:, 1].reshape(-1,1), \
                      landmarks[:, 2].reshape(-1,1), \
                      landmarks[:, 3].reshape(-1,1)
  landmarks_l = np.concatenate(
      (Xc + 0.5 * L * np.sin(Theta), Yc - 0.5 * L * np.cos(Theta), Theta),
      axis=1)
  landmarks_r = np.concatenate(
      (Xc - 0.5 * L * np.sin(Theta), Yc + 0.5 * L * np.cos(Theta), Theta),
      axis=1)

  # Calculate the distances between each landmark and pose.
  distances_l = np.sqrt(
      np.sum(np.power(landmarks_l[:, :2] - poses[:, :2], 2), axis=1))
  distances_r = np.sqrt(
      np.sum(np.power(landmarks_r[:, :2] - poses[:, :2], 2), axis=1))

  # Calculate the angles between each landmark and pose.
  Xl, Yl = landmarks_l[:, 0], landmarks_l[:, 1]
  Xr, Yr = landmarks_r[:, 0], landmarks_r[:, 1]
  Xp, Yp, Thetap = poses[:, :0], poses[:, :1], poses[:, :2]
  angles_l = np.arctan2(Xl - Xp, Yp - Yl) - Thetap
  angles_r = np.arctan2(Xr - Xp, Yp - Yr) - Thetap

  # Check if distances and angles are within sight.
  seeable = (distances_l <= np.where(ranges < max_distance, ranges, max_distance)) & \
            (distances_r <= np.where(ranges < max_distance, ranges, max_distance)) & \
            (np.abs(angles_l) <= np.deg2rad(FOV/2)) & \
            (np.abs(angles_r) <= np.deg2rad(FOV/2))

  return seeable


def normalize(x, scales):
  '''
      Balance the array dimensions according to their scales. 
      x: (n_sample, n_dimension), array to be normalized.
      scales: (n_dimension, ), scales for each dimension
  '''
  base = scales[0]
  for dim, scale in enumerate(scales):
    x[:, dim] *= scale / base

  return x


def landmarklist_to_array(landmark_list):
  ''' Convert from cartographer_ros LandmarkList to numpy array.'''

  a = np.empty((len(landmark_list), 3))
  for i, landmark in enumerate(landmark_list):
    a[i] = np.array([
        landmark.tracking_from_landmark_transform.position.x,
        landmark.tracking_from_landmark_transform.position.y,
        landmark.tracking_from_landmark_transform.orientation.z
    ])
  return a