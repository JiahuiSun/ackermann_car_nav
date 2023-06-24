import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import pickle

from nlos_sensing import transform, nlos_filter_and_mapping, line_symmetry_point
from nlos_sensing import get_span, find_end_point, fit_line_ransac


local_sensing_range = [-0.5, 5, -3, 3]
min_points_inline = 20
min_length_inline = 0.6
ransac_sigma = 0.02
ransac_iter = 200
filter = DBSCAN(eps=1, min_samples=20)

fig, ax = plt.subplots(figsize=(16, 16))
color_panel = ['ro', 'go', 'bo', 'co', 'wo', 'yo', 'mo', 'ko']


def init_fig():
    ax.clear()
    ax.set_xlabel('x(m)')
    ax.set_ylabel('y(m)')
    ax.set_xlim([-5, 10])
    ax.set_ylim([-5, 5])


def gen_data():
    with open("/home/dingrong/Code/ackermann_car_nav/data/ransac_static_2023-05-21-20-29-07.pkl", 'rb') as f:
        all_point_cloud = pickle.load(f)
    for t, laser_pc, mmwave_pc in all_point_cloud:
        # 从激光雷达坐标系到小车坐标系
        laser_point_cloud = transform(laser_pc, 0.08, 0, 180)
        # 过滤激光雷达点云，去掉距离小车中心5米以外的点
        flag_x = np.logical_and(laser_point_cloud[:, 0]>=local_sensing_range[0], laser_point_cloud[:, 0]<=local_sensing_range[1])
        flag_y = np.logical_and(laser_point_cloud[:, 1]>=local_sensing_range[2], laser_point_cloud[:, 1]<=local_sensing_range[3])
        flag = np.logical_and(flag_x, flag_y) 
        laser_point_cloud = laser_point_cloud[flag]
        # 从毫米波雷达坐标系到小车坐标系
        mmwave_pc[:, :2] = transform(mmwave_pc[:, :2], 0.17, 0, 60)
        yield t, laser_point_cloud, mmwave_pc


def visualize(result):
    init_fig()
    t, laser_point_cloud, mmwave_point_cloud = result

    # 提取墙面
    fitted_lines = []
    for i in range(3):
        # 不用3条线就可以覆盖所有点
        if len(laser_point_cloud) < min_points_inline:
            break
        coef, inlier_mask = fit_line_ransac(laser_point_cloud, max_iter=ransac_iter, sigma=ransac_sigma)
        # 过滤在墙面的直线上但明显是噪声的点
        db = filter.fit(laser_point_cloud[inlier_mask])
        cluster_mask = np.zeros_like(inlier_mask) > 0
        cluster_mask[inlier_mask] = db.labels_ >= 0  # 即使前墙是2段的也保留
        inlier_mask = np.logical_and(inlier_mask, cluster_mask)
        inlier_points = laser_point_cloud[inlier_mask]
        # 过滤非墙面的直线
        # 点数太少
        if len(inlier_points) < min_points_inline:
            continue
        # 跨度太小
        if get_span(inlier_points) < min_length_inline:
            continue
        ax.plot(inlier_points[:, 0], inlier_points[:, 1], color_panel[i], ms=2)
        outlier_mask = np.logical_not(inlier_mask)
        laser_point_cloud = laser_point_cloud[outlier_mask]
        fitted_lines.append([coef, inlier_points])
    ax.plot(laser_point_cloud[:, 0], laser_point_cloud[:, 1], color_panel[-1], ms=2)
    ax.plot(mmwave_point_cloud[:, 0], mmwave_point_cloud[:, 1], color_panel[-2], ms=2)
    ax.set_title(f"Timestamp: {t:.2f}s")

    # 区分墙面
    if len(fitted_lines) == 2:
        corner_args = {}
        coef1, inlier_points1 = fitted_lines[0]
        center1 = np.mean(inlier_points1, axis=0)
        coef2, inlier_points2 = fitted_lines[1]
        center2 = np.mean(inlier_points2, axis=0)
        diff = np.abs(coef1[0]-coef2[0])
        if diff > 1:  # 两条线垂直，一个前墙，一个侧墙
            if center1[0] > center2[0]:  # 判断哪个是前墙
                corner_args['far_wall'] = coef1
                corner_args['barrier_wall'] = coef2
                barrier_corner = find_end_point(inlier_points2, 0)[1]
                corner_args['barrier_corner'] = np.array(barrier_corner) # x值最大的
            else:
                corner_args['far_wall'] = coef2
                corner_args['barrier_wall'] = coef1
                barrier_corner = find_end_point(inlier_points1, 0)[1]
                corner_args['barrier_corner'] = np.array(barrier_corner) # x值最大的
            ax.plot(*barrier_corner, color_panel[-3], ms=8)
            ax.plot(0.17, 0, color_panel[-3], ms=8)
            # 过滤和映射
            far_map_corner = line_symmetry_point(corner_args['far_wall'], corner_args['barrier_corner'])
            far_map_radar = line_symmetry_point(corner_args['far_wall'], np.array([0.17, 0]))
            ax.plot(*far_map_corner, color_panel[-3], ms=8)
            ax.plot(*far_map_radar, color_panel[-3], ms=8)
            point_cloud_nlos = nlos_filter_and_mapping(mmwave_point_cloud, np.array([0.17, 0]), corner_args)
            ax.plot(point_cloud_nlos[:, 0], point_cloud_nlos[:, 1], color_panel[-3], ms=2)


ani = animation.FuncAnimation(
    fig, visualize, gen_data, interval=100,
    init_func=init_fig, repeat=True, save_count=200
)
writergif = animation.PillowWriter(fps=10)
# ani.save("test2.gif", writer=writergif)
plt.show()
