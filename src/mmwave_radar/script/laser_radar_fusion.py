import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import pickle

from ransac import fit_line_ransac
from nlos_sensing import transform


cluster = DBSCAN(eps=0.1, min_samples=5)
filter = DBSCAN(eps=1, min_samples=10)

fig, ax = plt.subplots(figsize=(16, 16))
color_panel = ['ro', 'go', 'bo', 'co', 'yo', 'wo', 'mo', 'ko']


def init_fig():
    ax.clear()
    ax.set_xlabel('x(m)')
    ax.set_ylabel('y(m)')
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])

def gen_data():
    with open("/home/dingrong/Code/ackermann_car_nav/data/ransac_static_2023-05-21-20-29-07.pkl", 'rb') as f:
        all_point_cloud = pickle.load(f)
    for t, laser_pc, mmwave_pc in all_point_cloud:
        # 从激光雷达坐标系到小车坐标系
        laser_point_cloud = transform(laser_pc, 0.08, 0, 180)
        # 过滤，去掉以距离小车中心5米以外的点
        flag_x = np.logical_and(laser_point_cloud[:, 0]>=-1, laser_point_cloud[:, 0]<=5)
        flag_y = np.logical_and(laser_point_cloud[:, 1]>=-5, laser_point_cloud[:, 1]<=5)
        flag = np.logical_and(flag_x, flag_y) 
        laser_point_cloud = laser_point_cloud[flag]
        # 从毫米波雷达坐标系到小车坐标系
        mmwave_pc[:, :2] = transform(mmwave_pc[:, :2], 0.17, 0, 60)
        yield t, laser_point_cloud, mmwave_pc

def visualize_cluster(result):
    init_fig()
    t, laser_point_cloud, mmwave_point_cloud = result
    # 聚类，对每一类进行ransac拟合直线
    db = cluster.fit(laser_point_cloud)
    labels = db.labels_
    unique_labels = sorted(set(labels[labels >= 0]))
    for label in unique_labels:
        print(label, unique_labels)
        laser_point_cloud = laser_point_cloud[label == labels]
        if len(laser_point_cloud) < 10 or label > 7:
            continue
        coef, inlier_mask = fit_line_ransac(laser_point_cloud)
        inlier_points = laser_point_cloud[inlier_mask]
        ax.plot(inlier_points[:, 0], inlier_points[:, 1], color_panel[label], ms=2)
    ax.set_title(f"Timestamp: {t:.2f}s")

"""
给我前墙、侧墙、拐点
- 如果只有2条边：
    - 如果两条边平行，用侧墙来映射
    - 如果两条边垂直，用前墙来映射——前墙的平均x值最大
- 如果有3条边：
"""
def visualize(result):
    init_fig()
    t, laser_point_cloud, mmwave_point_cloud = result
    fitted_lines = []
    for i in range(3):
        if len(laser_point_cloud) < 100:
            break
        coef, inlier_mask = fit_line_ransac(laser_point_cloud)
        # 过滤墙面的直线上但不在线段上的点
        db = filter.fit(laser_point_cloud[inlier_mask])
        cluster_mask = np.zeros_like(inlier_mask) > 0
        cluster_mask[inlier_mask] = db.labels_ >= 0
        inlier_mask = np.logical_and(inlier_mask, cluster_mask)
        inlier_points = laser_point_cloud[inlier_mask]
        if len(inlier_points) < 50:
            continue
        ax.plot(inlier_points[:, 0], inlier_points[:, 1], color_panel[i], ms=2)
        outlier_mask = np.logical_not(inlier_mask)
        laser_point_cloud = laser_point_cloud[outlier_mask]

        center = np.mean(inlier_points, axis=0)
        fitted_lines.append([coef, inlier_points, center])
    ax.plot(laser_point_cloud[:, 0], laser_point_cloud[:, 1], color_panel[-1], ms=2)
    ax.plot(mmwave_point_cloud[:, 0], mmwave_point_cloud[:, 1], color_panel[-2], ms=2)
    ax.set_title(f"Timestamp: {t:.2f}s")

    # 判断
    if len(fitted_lines) == 2:
        corner_args = {}
        coef1, inlier_points1, center1 = fitted_lines[0]
        coef2, inlier_points2, center2 = fitted_lines[1]
        diff = np.abs(coef1[0]-coef2[0])
        print(f"diff: {diff}")
        if diff > 1:  # 两条线垂直，一个前墙，一个侧墙
            if center1[0] > center2[0]:  # 判断哪个是前墙
                corner_args['front_wall'] = coef1
                corner_args['side_wall'] = coef2
                corner_args['inner_corner'] = find_end_point(inlier_points2, 0)[1] # x值最大的
            else:
                corner_args['front_wall'] = coef2
                corner_args['side_wall'] = coef1
                corner_args['inner_corner'] = find_end_point(inlier_points1, 0)[1] # x值最大的
            ax.plot(*corner_args['inner_corner'], color_panel[-2], ms=8)
            ax.plot(0.08, 0, color_panel[-2], ms=8)
            # 现在墙面信息已经有了，可以开始过滤和映射了
            

def find_end_point(points, axis=0):
    max_v, min_v = -np.inf, np.inf
    max_p, min_p = None, None
    for p in points:
        if p[axis] > max_v:
            max_v = p[axis]
            max_p = p
        if p[axis] < min_v:
            min_v = p[axis]
            min_p = p
    return min_p, max_p


ani = animation.FuncAnimation(
    fig, visualize, gen_data, interval=200,
    init_func=init_fig, repeat=False
)
# ani.save("plan1-2.gif", writer='imagemagick')
plt.show()
