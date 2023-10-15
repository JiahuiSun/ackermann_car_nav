import rospy
import sys
import message_filters
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from std_msgs.msg import Float64MultiArray
from sensor_msgs.point_cloud2 import read_points
from sensor_msgs.msg import PointCloud2
import actionlib
import threading
from webots_ros.srv import set_float, set_int, set_bool
from webots_ros.msg import BoolStamped
from multiprocessing import Process, Pool
import globalvar as gl

def thread_wait_for_server(client):
    client.wait_for_server()

def set_goal(x, y, quat):
    # 初始化节点
    # rospy.init_node('move_base_client')
    # 创建一个actionlib客户端，连接move_base服务器
    client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
    # 等待服务器启动
    wait_thread = threading.Thread(target=thread_wait_for_server, args=(client,))
    wait_thread.start()
    # client.wait_for_server()
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

    # 发送目标点
    client.send_goal(goal)
    # 等待机器人到达目标点
    client.wait_for_result()

def rosservice_call(func, arg = None):
    if arg == None:
        p = Process(target=func)
    else:
        p = Process(target=func, args=arg)
    p.start()
    p.join()

def rosservice_set_cruising_speed(speed):
    rospy.wait_for_service('/vehicle/automobile/set_cruising_speed')
    set_cruising_speed_service = rospy.ServiceProxy('/vehicle/automobile/set_cruising_speed', set_float)
    try:
        set_cruising_speed_service(speed)
        # rospy.loginfo("Set cruising speed to %f", speed)
    except rospy.ServiceException as e:
        rospy.logerr("Failed to set velocity: %s", str(e))

def rosservice_set_steering_angle(angle):
    rospy.wait_for_service('/vehicle/automobile/set_steering_angle')
    set_steering_angle_service = rospy.ServiceProxy('/vehicle/automobile/set_steering_angle', set_float)
    try:
        set_steering_angle_service(angle)
        # rospy.loginfo("Set steering angle to %f", angle)
    except rospy.ServiceException as e:
        rospy.logerr("Failed to set angle: %s", str(e))



def rosservice_touch_sensor_enable(timestep):
    rospy.wait_for_service('/vehicle/touch_sensor/enable')
    touch_sensor_enable_service = rospy.ServiceProxy('/vehicle/touch_sensor/enable', set_int)
    try:
        touch_sensor_enable_service(timestep)
        rospy.loginfo("Enable touch sensor")
    except rospy.ServiceException as e:
        rospy.logerr("Failed to enable touch sensor: %s", str(e))

def rosservice_lidar_enable(timestep):
    rospy.wait_for_service('/vehicle/lidar/enable')
    lidar_enable_service = rospy.ServiceProxy('/vehicle/lidar/enable', set_int)
    try:
        lidar_enable_service(timestep)
        rospy.loginfo("Enable lidar")
    except rospy.ServiceException as e:
        rospy.logerr("Failed to enable lidar: %s", str(e))


def rosservice_lidar_pointcloud_enable():
    rospy.wait_for_service('/vehicle/lidar/enable_point_cloud')
    lidar_pointcloud_enable_service = rospy.ServiceProxy('/vehicle/lidar/enable_point_cloud', set_bool)
    try:
        lidar_pointcloud_enable_service(True)
        rospy.loginfo("Enable lidar point cloud")
    except rospy.ServiceException as e:
        rospy.logerr("Failed to enable lidar point cloud: %s", str(e))

def thread_spin():
    rospy.spin()

def rostopic_lidar_pointcloud_get_callback(msg):
    pt = []
    for point in read_points(msg,skip_nans=True):
        ptx = point[0]
        pty = point[1]
        pt.append([ptx, pty])
    gl.set_value('point_cloud', pt)

def rostopic_lidar_pointcloud_get():
    # rospy.init_node('lidar_pointcloud_listener', anonymous=True)
    rospy.Subscriber("/vehicle/lidar/point_cloud", PointCloud2, rostopic_lidar_pointcloud_get_callback)
    # spin_thread = threading.Thread(target=thread_spin)
    # spin_thread.start()
    rospy.spin()

def rostopic_touch_sensor_get_callback(msg):
    gl.set_value('touch', msg.data)

def rostopic_touch_sensor_get():

    rospy.Subscriber("/vehicle/touch_sensor/value", BoolStamped, rostopic_touch_sensor_get_callback)
    # spin_thread = threading.Thread(target=thread_spin)
    # spin_thread.start()
    rospy.spin()
