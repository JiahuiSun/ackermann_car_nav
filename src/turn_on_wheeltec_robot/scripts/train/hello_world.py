from controller import *
from ros_utils import *
from multiprocessing import Process, Pool
import globalvar as gl
import numpy as np

supervisor = Supervisor()

timestep = 32*3

# pool = Pool(2)

# rospy.sleep(25)
# pool.apply_async(func=rosservice_lidar_enable, args=(32,))
# p1 = Process(target=rosservice_lidar_enable, args=(32,))
# p1.start()

# rospy.sleep(0.1)
# pool.apply_async(func=rosservice_lidar_pointcloud_enable)

# p2 = Process(target=rosservice_lidar_pointcloud_enable)
# p2.start()

# # rospy.sleep(0.1)
# rosservice_lidar_enable(96)
# rosservice_lidar_pointcloud_enable(True)
# # rospy.sleep(0.1)
# flag = 1


rospy.init_node('hello', anonymous=True)
# touch = threading.Thread(target=rostopic_touch_sensor_get)
# touch.start()

# pointcloud = threading.Thread(target=rostopic_lidar_pointcloud_get)
# pointcloud.start()

while True:
    print('hello ros!')
    # rostopic_touch_sensor_get()
    # print(gl.get_value('touch'))
    # print(gl.get_value('point_cloud'))
    robot_node = supervisor.getFromDef("base_link")
    robot_pos_field = robot_node.getField("translation")
    robot_rot_field = robot_node.getField("rotation")
    robot_pos_list = robot_pos_field.getSFVec3f()[:2]
    robot_rot_list = robot_rot_field.getSFRotation()
    x, y = robot_pos_list
    theta = robot_rot_list[3] if robot_rot_list[2] >= 0 else -robot_rot_list[3]
    
    success = False

    delta_r = np.random.uniform(0, 0.5)
    delta_theta = np.random.uniform(-np.pi/20, np.pi/20)
    theta_ = theta + delta_theta
    x_ = x + delta_r * np.cos(theta_)
    y_ = y + delta_r * np.sin(theta_)
    
    pub = rospy.Publisher('goal_list', Float64MultiArray, queue_size=10)
    array = Float64MultiArray()
    array.data = [x_, y_, theta_]
    loop_rate = rospy.Rate(50)
    pub.publish(array)
    loop_rate.sleep()


    supervisor.step(timestep)

    # print(rosservice_lidar_pointcloud_get())
    # pool.close()
    # pool.join()
    # p = Pool()
    # res = p.apply_async(func=rosservice_lidar_pointcloud_get, args=(0,))
    # print(res)
