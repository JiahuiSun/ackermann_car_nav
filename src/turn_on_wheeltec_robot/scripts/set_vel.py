#!/usr/bin/env python

import time
import rospy
import numpy as np
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Vector3Stamped
from webots_ros.srv import set_float
import math


x = 0.0
z = 0.0
min_r = 0.7
wheel_base = 0.324
linear_velocity_ratio = 0.6944

pub_freq = 25.0
rospy.init_node('cmd_vel_listener', anonymous=True)
rate = rospy.Rate(pub_freq)

def callback(msg):

    x = msg.linear.x
    if not x >= -5 or not x <= 5:
        x = 0
    z = msg.angular.z
    if z != 0:
        r = x/z
    else: r = min_r
    theta = math.atan(wheel_base/r)

    set_cruising_speed_service = rospy.ServiceProxy('/vehicle/automobile/set_cruising_speed', set_float)
    try:
        set_cruising_speed_service(x / linear_velocity_ratio)
        # rospy.loginfo("Set cruising speed to %f", x)
    except rospy.ServiceException as e:
        rospy.logerr("Failed to set velocity: %s", str(e))

    set_steering_angle_service = rospy.ServiceProxy('/vehicle/automobile/set_steering_angle', set_float)
    try:
        set_steering_angle_service(z)
        # rospy.loginfo("Set steering angle to %f", z)
    except rospy.ServiceException as e:
        rospy.logerr("Failed to set velocity: %s", str(e))



def listener_and_pub():
    subscriber = rospy.Subscriber("/cmd_vel", Twist, callback) #/cmd_vel
    rospy.spin()

if __name__ == '__main__':
    try:
        listener_and_pub()
    except rospy.ROSInterruptException:
        pass
