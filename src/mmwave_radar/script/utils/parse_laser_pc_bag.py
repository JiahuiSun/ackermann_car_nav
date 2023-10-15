import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import rosbag
from sensor_msgs import point_cloud2
from nlos_sensing import transform


fig, ax = plt.subplots(figsize=(8, 8))
line, = ax.plot([], [], 'ro', ms=2)
file_path = "/home/agent/Code/ackermann_car_nav/data/20231002/soft-3-F_2023-10-01-20-39-52"

# 这个脚本用来大致看一眼GT激光雷达点云，从而切割人和小车
def init_fig():
    ax.set_xlabel('x(m)')
    ax.set_ylabel('y(m)')
    ax.set_xlim([-7, 0])
    ax.set_ylim([-3, 2])
    return line

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
            # point_cloud = transform(point_cloud, 0.08, 0, 90)
            yield point_cloud, msg.header.seq

def visualize(result):
    adc_data, seq = result
    x_pos, y_pos = adc_data[:, 0], adc_data[:, 1]
    ax.set_title(f"frame id: {seq}")
    line.set_data(x_pos, y_pos)

ani = animation.FuncAnimation(
    fig, visualize, gen_data, interval=100,
    init_func=init_fig, repeat=False, save_count=2000
)
ani.save(f"{file_path}-gt.gif", writer='pillow')
# plt.show()
