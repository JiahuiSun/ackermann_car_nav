import rospy
import sys
from actionlib_msgs.msg import *
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from std_msgs.msg import Float64MultiArray
import actionlib
import train.globalvar as gl
from tf import transformations
import numpy as np

success = True

def move_to_goal(msg):
    global success
    # 创建一个actionlib客户端，连接move_base服务器
    client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
    # 等待服务器启动
    client.wait_for_server()
    if success:
        x = msg.data[0]
        y = msg.data[1]
        theta = msg.data[2]
        quat = transformations.quaternion_from_euler(0, 0, theta)
        # 创建一个MoveBaseGoal对象
        goal = MoveBaseGoal()
        # 设置目标点的坐标
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = -y
        goal.target_pose.pose.orientation.x = quat[0]
        goal.target_pose.pose.orientation.y = quat[1]
        goal.target_pose.pose.orientation.z = quat[2]
        goal.target_pose.pose.orientation.w = quat[3]
        # print(goal)
        # 发送目标点
        client.send_goal(goal)
        # 等待机器人到达目标点
        # client.wait_for_result()
    state = client.get_state()
    if state == GoalStatus.SUCCEEDED:
        print("Goal succeeded!")
        success = True
    else:
        success = False


if __name__ == '__main__':
    # 初始化节点
    rospy.init_node('move_base_client')
    rospy.Subscriber('goal_list', Float64MultiArray, move_to_goal)
    rospy.spin()