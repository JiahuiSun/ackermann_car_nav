import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import rosbag
from sensor_msgs import point_cloud2
from sklearn.cluster import DBSCAN

from ransac import fit_line_ransac
from nlos_sensing import transform


cluster = DBSCAN(eps=0.1, min_samples=5)

fig, ax = plt.subplots(figsize=(16, 16))
line0, = ax.plot([], [], 'ro', ms=2)
line1, = ax.plot([], [], 'go', ms=2)
line2, = ax.plot([], [], 'bo', ms=2)
line3, = ax.plot([], [], 'co', ms=2)
line4, = ax.plot([], [], 'yo', ms=2)
line5, = ax.plot([], [], 'wo', ms=2)
line6, = ax.plot([], [], 'mo', ms=2)
line7, = ax.plot([], [], 'ko', ms=1)
lines = [line0, line1, line2, line3, line4, line5, line6, line7]


def init_fig():
    ax.set_xlabel('x(m)')
    ax.set_ylabel('y(m)')
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    return lines

def gen_data():
    for topic, msg, t in rosbag.Bag(
        "/home/dingrong/Code/ackermann_car_nav/data/floor1_dynamic_2023-05-16-17-41-08.bag", 'r'):
        if topic == '/laser_point_cloud':
            points = point_cloud2.read_points_list(
                msg, field_names=['x', 'y']
            )
            x_pos = [p.x for p in points]
            y_pos = [p.y for p in points]
            point_cloud = np.array([x_pos, y_pos]).T
            # 从激光雷达坐标系到小车坐标系
            point_cloud = transform(point_cloud, 0.08, 0, 180)
            # 过滤，去掉以距离小车中心5米以外的点
            flag_x = np.logical_and(point_cloud[:, 0]>=-1, point_cloud[:, 0]<=5)
            flag_y = np.logical_and(point_cloud[:, 1]>=-5, point_cloud[:, 1]<=5)
            flag = np.logical_and(flag_x, flag_y) 
            point_cloud = point_cloud[flag]
            yield point_cloud, msg.header.seq

def visualize_cluster(result):
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
        lines[label].set_data(inlier_points[:, 0], inlier_points[:, 1])
    ax.set_title(f"frame id: {seq}")

def visualize(result):
    point_cloud, seq = result
    fitted_lines = []
    for i in range(4):
        if len(point_cloud) < 50:
            break
        coef, inlier_mask = fit_line_ransac(point_cloud)
        inlier_points = point_cloud[inlier_mask]
        if len(inlier_points) < 20:
            continue
        lines[i].set_data(inlier_points[:, 0], inlier_points[:, 1])
        fitted_lines.append([coef, inlier_mask])
        outlier_mask = np.logical_not(inlier_mask)
        point_cloud = point_cloud[outlier_mask]
    lines[-1].set_data(point_cloud[:, 0], point_cloud[:, 1])
    ax.set_title(f"frame id: {seq}")

ani = animation.FuncAnimation(
    fig, visualize, gen_data, interval=100,
    init_func=init_fig, repeat=False
)
# ani.save("plan1-2.gif", writer='imagemagick')
plt.show()
