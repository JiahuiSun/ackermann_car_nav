#!/usr/bin/env python

import time
import rospy
import numpy as np
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Vector3Stamped
from webots_ros.srv import set_float
import select
import sys
import termios
import tty
from std_msgs.msg import String


x = 0.0
z = 0.0
linear_velocity_ratio = 0.6944

pub_freq = 25.0
rospy.init_node('cmd_vel_listener', anonymous=True)
rate = rospy.Rate(pub_freq)

def listenKey():
    ## sys.stdin表示标准化输入
    ## termios.tcgetattr(fd)返回一个包含文件描述符fd的tty属性的列表
    property_list = termios.tcgetattr(sys.stdin)
    ## tty.setraw(fd, when=termios.TCSAFLUSH)将文件描述符fd的模式更改为raw。如果when被省略，则默认为termios.TCSAFLUSH，并传递给termios.tcsetattr()
    tty.setraw(sys.stdin.fileno())
    ## 第一个参数是需要监听可读的套接字, 第二个是需要监听可写的套接字, 第三个是需要监听异常的套接字, 第四个是时间限制设置
    ## 如果监听的套接字满足了可读可写条件, 那么所返回的can_read 或 can_write就会有值, 然后就可以利用这些返回值进行后续操作
    can_read, _, _ = select.select([sys.stdin], [], [], 0.1)
    if can_read:
        keyValue = sys.stdin.read(1)
    else:
        keyValue = ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, property_list)
    return keyValue


def set_cruising_speed(speed):
    set_cruising_speed_service = rospy.ServiceProxy('/vehicle/automobile/set_cruising_speed', set_float)
    try:
        set_cruising_speed_service(speed)
        rospy.loginfo("Set cruising speed to %f", speed)
    except rospy.ServiceException as e:
        rospy.logerr("Failed to set velocity: %s", str(e))

def set_steering_angle(angle):
    set_steering_angle_service = rospy.ServiceProxy('/vehicle/automobile/set_steering_angle', set_float)
    try:
        set_steering_angle_service(angle)
        rospy.loginfo("Set steering angle to %f", angle)
    except rospy.ServiceException as e:
        rospy.logerr("Failed to set velocity: %s", str(e))

def use_key(keyValue):
    if keyValue == 'q':
        set_cruising_speed(0)
        set_steering_angle(0)
        return False
    if keyValue == 'w':
        set_cruising_speed(0.5)
        set_steering_angle(0)
    if keyValue == 's':
        set_cruising_speed(0)
    if keyValue == 'a':
        set_steering_angle(-0.3)
    if keyValue == 'd':
        set_steering_angle(0.3)

    return True



def listener_and_pub():
    while(use_key(listenKey())):
        pass

if __name__ == '__main__':
    try:
        listener_and_pub()
    except rospy.ROSInterruptException:
        pass
