import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import rosbag
from sensor_msgs import point_cloud2
from sklearn.cluster import DBSCAN
import sys
import os

from nlos_sensing import transform, fit_line_ransac, pc_filter, get_span


if len(sys.argv) > 1:
    file_path = sys.argv[1]
    out_path = sys.argv[2]
    mode = sys.argv[3]
else:
    file_path = "/home/agent/Code/ackermann_car_nav/data/20230626/exp01_2023-06-26-19-21-33.bag"
    out_path = "/home/agent/Code/ackermann_car_nav/data/20231023"
    mode = "train"
file_name = file_path.split('/')[-1].split('.')[0][:-20]
os.makedirs(f"{out_path}/images/{mode}", exist_ok=True)
os.makedirs(f"{out_path}/labels/{mode}", exist_ok=True)
os.makedirs(f"{out_path}/gifs", exist_ok=True)
save_gif = True
cluster = DBSCAN(eps=0.1, min_samples=5)
filter = DBSCAN(eps=1, min_samples=10)
min_points_inline = 20
min_length_inline = 0.6
local_sensing_range = [-1, 5, -3, 3]  # 切割小车周围点云

fig, ax = plt.subplots(figsize=(16, 16))
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
        if topic == '/laser_point_cloud':
            points = point_cloud2.read_points_list(
                msg, field_names=['x', 'y']
            )
            x_pos = [p.x for p in points]
            y_pos = [p.y for p in points]
            point_cloud = np.array([x_pos, y_pos]).T
            # 从激光雷达坐标系到小车坐标系
            point_cloud = transform(point_cloud, 0.08, 0, 180)
            # 过滤，去掉以距离小车中心3米以外的点
            point_cloud = pc_filter(point_cloud, *local_sensing_range)
            yield point_cloud, msg.header.seq

def visualize_cluster(result):
    init_fig()
    all_point_cloud, seq = result
    # 聚类，对每一类进行ransac拟合直线
    db = cluster.fit(all_point_cloud)
    labels = db.labels_
    unique_labels = sorted(set(labels[labels >= 0]))
    for label in unique_labels:
        print(label, unique_labels)
        point_cloud = all_point_cloud[label == labels]
        if len(point_cloud) < 10 or label > 7:
            continue
        coef, inlier_mask = fit_line_ransac(point_cloud)
        inlier_points = point_cloud[inlier_mask]
        ax.plot(inlier_points[:, 0], inlier_points[:, 1], color_panel[label], ms=2)
    ax.set_title(f"frame id: {seq}")

def visualize(result):
    init_fig()
    point_cloud, seq = result
    fitted_lines = []
    for i in range(2):
        if len(point_cloud) < min_points_inline:
            break
        coef, inlier_mask = fit_line_ransac(point_cloud)
        db = filter.fit(point_cloud[inlier_mask])
        cluster_mask = np.zeros_like(inlier_mask) > 0
        cluster_mask[inlier_mask] = db.labels_ >= 0
        inlier_mask = np.logical_and(inlier_mask, cluster_mask)
        inlier_points = point_cloud[inlier_mask]
        # 过滤非墙面的直线
        # 点数太少
        if len(inlier_points) < min_points_inline:
            continue
        # 跨度太小
        if get_span(inlier_points) < min_length_inline:
            continue
        ax.plot(inlier_points[:, 0], inlier_points[:, 1], color_panel[i], ms=2)
        fitted_lines.append([coef, inlier_mask])
        outlier_mask = np.logical_not(inlier_mask)
        point_cloud = point_cloud[outlier_mask]
    ax.plot(point_cloud[:, 0], point_cloud[:, 1], color_panel[-1], ms=2)
    ax.set_title(f"frame id: {seq}")

ani = animation.FuncAnimation(
    fig, visualize, gen_data, interval=100,
    init_func=init_fig, repeat=False, save_count=1000
)
if save_gif:
    ani.save(f"{out_path}/gifs/{file_name}.gif", writer='pillow')
else:
    plt.show()
