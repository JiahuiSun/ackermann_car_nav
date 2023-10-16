import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import rosbag
from sensor_msgs import point_cloud2
from nlos_sensing import pc_filter


file_path = "/home/agent/Code/ackermann_car_nav/data/20231002/soft-3-C3_2023-10-01-21-30-52"
robot_range = [-2.1, 0, -3, 0]  # 切割小车
gt_range = [-8, 0, -0.3, 1.5]  # 切割人的点云
gt_range1 = [-8, 0, -0.3, 0.7]  # 切割人的点云
gt_range2 = [-8, 0, 0.8, 1.5]  # 切割人的点云

fig, ax = plt.subplots(figsize=(8, 8))
color_panel = ['ro', 'go', 'bo', 'co', 'yo', 'mo', 'ko']


def init_fig():
    ax.clear()
    ax.set_xlabel('x(m)')
    ax.set_ylabel('y(m)')
    ax.set_xlim([-8, 0])
    ax.set_ylim([-3, 2])
    ax.tick_params(direction='in')


def gen_data():
    for topic, msg, t in rosbag.Bag(
        f"{file_path}.bag", 'r'):
        if topic == '/laser_point_cloud2':
            points = point_cloud2.read_points_list(
                msg, field_names=['x', 'y']
            )
            x_pos = [p.x for p in points]
            y_pos = [p.y for p in points]
            point_cloud = np.array([x_pos, y_pos]).T

            # 用laser2提取小车位姿
            laser_pc_robot = pc_filter(point_cloud, *robot_range)

            # 标定激光雷达点云只保留人
            laser_pc_person1 = pc_filter(point_cloud, *gt_range1)
            laser_pc_person2 = pc_filter(point_cloud, *gt_range2)

            yield point_cloud, laser_pc_robot, laser_pc_person1, laser_pc_person2, msg.header.seq


def visualize(result):
    init_fig()
    point_cloud, laser_pc_robot, laser_pc_person1, laser_pc_person2, seq = result
    ax.set_title(f"frame id: {seq}")
    ax.plot(point_cloud[:, 0], point_cloud[:, 1], color_panel[-1], ms=2)
    ax.plot(laser_pc_robot[:, 0], laser_pc_robot[:, 1], color_panel[0], ms=2)
    ax.plot(laser_pc_person1[:, 0], laser_pc_person1[:, 1], color_panel[1], ms=2)
    ax.plot(laser_pc_person2[:, 0], laser_pc_person2[:, 1], color_panel[2], ms=2)


ani = animation.FuncAnimation(
    fig, visualize, gen_data, interval=100,
    init_func=init_fig, repeat=False, save_count=2000
)
ani.save(f"{file_path}-gt.gif", writer='pillow')
# plt.show()
