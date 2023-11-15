import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import rosbag
from sensor_msgs import point_cloud2
import sys
import os

from nlos_sensing import transform, pc_filter
from corner_type import fit_lines


if len(sys.argv) > 1:
    file_path = sys.argv[1]
    out_path = sys.argv[2]
    mode = sys.argv[3]
else:
    file_path = "/home/agent/Code/ackermann_car_nav/data/20231115/bc_static2_2023-11-14-15-30-01.bag"
    out_path = "/home/agent/Code/ackermann_car_nav/data/tmp"
    mode = "train"
file_name = file_path.split('/')[-1].split('.')[0][:-20]
os.makedirs(f"{out_path}/images/{mode}", exist_ok=True)
os.makedirs(f"{out_path}/labels/{mode}", exist_ok=True)
os.makedirs(f"{out_path}/gifs", exist_ok=True)
save_gif = True
is_onboard = True
onboard_wall_pc = [-1, 5, -3, 3]  # 切割小车周围点云
gt_wall_pc = [-4, 2, -3, 3]  # 切割gt周围点云

fig, ax = plt.subplots(figsize=(8, 8))
color_panel = ['ro', 'go', 'bo', 'co', 'yo', 'wo', 'mo', 'ko']


def init_fig():
    ax.clear()
    ax.set_xlabel('x(m)')
    ax.set_ylabel('y(m)')
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.tick_params(direction='in')

def gen_data():
    for topic, msg, t in rosbag.Bag(file_path, 'r'):
        if not is_onboard:
            if topic == '/laser_point_cloud2':
                points = point_cloud2.read_points_list(
                    msg, field_names=['x', 'y']
                )
                x_pos = [p.x for p in points]
                y_pos = [p.y for p in points]
                point_cloud = np.array([x_pos, y_pos]).T
                point_cloud = pc_filter(point_cloud, *gt_wall_pc)
                yield point_cloud, msg.header.seq
        else:
            if topic == '/laser_point_cloud':
                points = point_cloud2.read_points_list(
                    msg, field_names=['x', 'y']
                )
                x_pos = [p.x for p in points]
                y_pos = [p.y for p in points]
                point_cloud = np.array([x_pos, y_pos]).T
                point_cloud = transform(point_cloud, 0.08, 0, 180)
                point_cloud = pc_filter(point_cloud, *onboard_wall_pc)
                yield point_cloud, msg.header.seq

def visualize(result):
    init_fig()
    point_cloud, seq = result
    fitted_lines, remaining_points = fit_lines(point_cloud, 2)
    ax.plot(remaining_points[:, 0], remaining_points[:, 1], color_panel[-1], ms=2)
    for i, (coef, pc) in enumerate(fitted_lines):
        ax.plot(pc[:, 0], pc[:, 1], color_panel[i], ms=2)
    ax.set_title(f"frame id: {seq}")

ani = animation.FuncAnimation(
    fig, visualize, gen_data, interval=100,
    init_func=init_fig, repeat=False, save_count=1000
)
if save_gif:
    if is_onboard:
        ani.save(f"{out_path}/gifs/{file_name}-onboard.gif", writer='pillow')
    else:
        ani.save(f"{out_path}/gifs/{file_name}-gt.gif", writer='pillow')
else:
    plt.show()
