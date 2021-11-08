#!/usr/bin/env python3

# packages
import rospy
from geometry_msgs.msg import PoseStamped
import utils as Utils

import numpy as np

if __name__ == "__main__":
  rospy.init_node("initialpose_publisher")
  initialpose_pub = rospy.Publisher("/initial_pose", PoseStamped, queue_size=1)

  while not rospy.is_shutdown():
    pose = PoseStamped()
    pose.header.stamp = rospy.get_rostime()
    pose.header.frame_id = "map"
    pose.pose.position.x = -5.0
    pose.pose.position.y = -.5
    pose.pose.orientation = Utils.angle_to_quaternion(-np.pi/2)
    initialpose_pub.publish(pose)

  rospy.spin()
