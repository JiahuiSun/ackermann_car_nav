import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
from sklearn.cluster import DBSCAN
import pickle
import os
import cv2
import sys

from radar_fft_music_RA import *
from nlos_sensing import transform, bounding_box2, intersection_of_2line, isin_triangle
from nlos_sensing import get_span, find_end_point, fit_line_ransac, line_by_coef_p
from nlos_sensing import transform_inverse, line_symmetry_point, line_by_2p, pc_filter


if len(sys.argv) > 1:
    file_path = sys.argv[1]
    out_path = sys.argv[2]
    mode = sys.argv[3]
else:
    file_path = "/home/agent/Code/ackermann_car_nav/data/20231002/soft-3-A_2023-10-01-20-21-03.pkl"
    out_path = "/home/agent/Code/ackermann_car_nav/data/20231002"
    mode = "train"
file_name = file_path.split('/')[-1].split('.')[0][:-20]
os.makedirs(f"{out_path}/images/{mode}", exist_ok=True)
os.makedirs(f"{out_path}/labels/{mode}", exist_ok=True)
os.makedirs(f"{out_path}/gifs", exist_ok=True)
save_gif = True
save_data = True
plot_radar_pc = True
img_fmt = True
cnt = 0
local_sensing_range = [-0.5, 5, -3, 3]  # 切割小车周围点云
min_onboard_laser_point_num = 20
min_GT_laser_point_num = 4
min_points_inline = 20
min_length_inline = 0.6
ransac_sigma = 0.02
ransac_iter = 200
filter = DBSCAN(eps=1, min_samples=20)
stamp, accuracy = [], []
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
color_panel = ['ro', 'go', 'bo', 'co', 'yo', 'mo', 'ko']


def init_fig():
    ax1.clear()
    ax1.set_xlabel('x(m)')
    ax1.set_ylabel('y(m)')
    ax1.set_xlim([-7, 5])
    ax1.set_ylim([-2, 8])
    ax1.tick_params(direction='in')
    ax2.clear()
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')


def gen_data():
    with open(file_path, 'rb') as f:
        all_point_cloud = pickle.load(f)
    for t, laser_pc_onboard, laser_pc_person, mmwave_pc, mmwave_raw_data, trans in all_point_cloud:
        inter, theta = trans
        theta = theta * 180 / np.pi
        
        # 小车激光雷达只保留小车附近的点
        # 激光雷达->小车
        laser_pc_onboard = transform(laser_pc_onboard, 0.08, 0, 180)
        laser_pc_onboard = pc_filter(laser_pc_onboard, *local_sensing_range)
        if len(laser_pc_onboard) < min_onboard_laser_point_num:
            continue
        # 小车->毫米波雷达
        laser_pc_onboard = transform_inverse(laser_pc_onboard, 0.17, 0, 90)

        # 生成毫米波RA tensor和点云
        result = gen_point_cloud_plan3(mmwave_raw_data)
        if result is None:
            continue
        RA_cart, mmwave_point_cloud = result

        if len(laser_pc_person) < min_GT_laser_point_num:
            continue
        # 标定激光雷达->小车坐标系->毫米波雷达坐标系
        laser_pc_person = transform_inverse(
            transform_inverse(laser_pc_person, inter[0], inter[1], 360-theta), 
            0.17, 0, 90
        )
        yield t, laser_pc_person, laser_pc_onboard, RA_cart, mmwave_point_cloud


def visualize(result):
    global cnt
    init_fig()
    t, laser_pc_person, laser_pc_onboard, RA_cart, mmwave_point_cloud = result

    # 提取墙面
    fitted_lines = []
    for i in range(2):
        # 不用3条线就可以覆盖所有点
        if len(laser_pc_onboard) < min_points_inline:
            break
        coef, inlier_mask = fit_line_ransac(laser_pc_onboard, max_iter=ransac_iter, sigma=ransac_sigma)
        # 过滤在墙面的直线上但明显是噪声的点
        db = filter.fit(laser_pc_onboard[inlier_mask])
        cluster_mask = np.zeros_like(inlier_mask) > 0
        cluster_mask[inlier_mask] = db.labels_ >= 0  # 即使前墙是2段的也保留
        inlier_mask = np.logical_and(inlier_mask, cluster_mask)
        inlier_points = laser_pc_onboard[inlier_mask]
        # 过滤非墙面的直线
        # 点数太少
        if len(inlier_points) < min_points_inline:
            continue
        # 跨度太小
        if get_span(inlier_points) < min_length_inline:
            continue
        outlier_mask = np.logical_not(inlier_mask)
        laser_pc_onboard = laser_pc_onboard[outlier_mask]
        fitted_lines.append([coef, inlier_points])

    # 区分墙面，目前就针对L开放型转角做
    coef1, inlier_points1 = fitted_lines[0]
    center1 = np.mean(inlier_points1, axis=0)
    coef2, inlier_points2 = fitted_lines[1]
    center2 = np.mean(inlier_points2, axis=0)
    assert np.abs(coef1[0]-coef2[0]) > 1, "parallel walls?"
    if center1[1] > center2[1]:  # 判断哪个是前墙
        far_wall = coef1
        barrier_wall = coef2
        barrier_corner = find_end_point(inlier_points2, 1)[1]
        barrier_corner = np.array(barrier_corner)
    else:
        far_wall = coef2
        barrier_wall = coef1
        barrier_corner = find_end_point(inlier_points1, 1)[1]
        barrier_corner = np.array(barrier_corner)
    symmtric_corner = line_symmetry_point(far_wall, barrier_corner)
    line_by_radar_and_corner = line_by_2p(np.array([0, 0]), barrier_corner)
    line_by_far_wall_and_symmtric_corner = line_by_coef_p(far_wall, symmtric_corner)
    inter1 = intersection_of_2line(line_by_radar_and_corner, far_wall)
    inter2 = intersection_of_2line(line_by_radar_and_corner, line_by_far_wall_and_symmtric_corner)
    inter3 = line_symmetry_point(far_wall, inter2)
    ax1.plot(*inter1, color_panel[-2], ms=5)
    ax1.plot(*inter2, color_panel[-2], ms=5)
    ax1.plot(*inter3, color_panel[-2], ms=5)
    ax1.plot(*barrier_corner, color_panel[-2], ms=5)
    ax1.plot(*symmtric_corner, color_panel[-2], ms=5)

    # 原来是用激光点云平均值，这个是不准确的，应该是中心点
    key_points, _ = bounding_box2(laser_pc_person, delta_x=0.1, delta_y=0.1)
    gt_center = key_points[0]
    # 当人位于NLOS区域内，开始保存数据
    if isin_triangle(barrier_corner, inter1, inter3, gt_center):
        # 毫米波点云过滤
        flag = isin_triangle(symmtric_corner, inter2, inter1, mmwave_point_cloud[:, :2])
        point_cloud_nlos = mmwave_point_cloud[flag]

        # 激光点云映射
        laser_pc_person = line_symmetry_point(far_wall, laser_pc_person)
        # bounding box ground truth
        key_points, box_hw = bounding_box2(laser_pc_person, delta_x=0.1, delta_y=0.1)
        center, top_right, bottom_right, bottom_left, top_left = key_points
        # 激光点云纠偏，直接将人的中心移动到毫米波点云中心
        delta = [0, 0]
        if len(point_cloud_nlos):
            delta = np.mean(point_cloud_nlos[:, :2], axis=0) - center
            center += delta
        # 防止center出界
        center[0] = np.clip(center[0], -(H-1)*range_res, (H-1)*range_res)
        center[1] = np.clip(center[1], 0, (H-1)*range_res)
        x = np.array([top_right[0], bottom_right[0], bottom_left[0], top_left[0], top_right[0]]) + delta[0]
        y = np.array([top_right[1], bottom_right[1], bottom_left[1], top_left[1], top_right[1]]) + delta[1]
        ax1.plot(x, y, 'k-', lw=1)
        ax1.plot(*center, color_panel[-1], ms=2)

        # 保存你想要的
        if save_data:
            if img_fmt:
                RA_cart = (RA_cart - RA_cart.min()) / (RA_cart.max() - RA_cart.min())
                RA_cart = (RA_cart * 255).astype('uint8')
                image_path = f"{out_path}/images/{mode}/{file_name}_{cnt}.png"
                cv2.imwrite(image_path, RA_cart)
            else:
                image_path = f"{out_path}/images/{mode}/{file_name}_{cnt}.npy"
                np.save(image_path, RA_cart)
            txt_path = f"{out_path}/labels/{mode}/{file_name}_{cnt}.txt"
            fwrite = open(txt_path, 'w')
            cnt += 1
            x_norm = (center[0]+H*range_res) / (H*2*range_res)
            y_norm = center[1] / (H*range_res)
            width_norm = box_hw[1] / (H*2*range_res)
            height_norm = box_hw[0] / (H*range_res)
            fwrite.write(f"0 {x_norm} {y_norm} {width_norm} {height_norm}\n")
            fwrite.close()

    # 可视化所有点云
    ax1.set_title(f"Timestamp: {t:.2f}s")
    # 毫米波雷达
    if plot_radar_pc:
        static_idx = np.abs(mmwave_point_cloud[:, 2]) <= doppler_res
        dynamic_idx = np.abs(mmwave_point_cloud[:, 2]) > doppler_res
        ax1.plot(mmwave_point_cloud[static_idx, 0], mmwave_point_cloud[static_idx, 1], color_panel[2], ms=2)
        ax1.plot(mmwave_point_cloud[dynamic_idx, 0], mmwave_point_cloud[dynamic_idx, 1], color_panel[0], ms=2)
    ax2.imshow(RA_cart[..., 0])
    # 激光雷达
    ax1.plot(inlier_points1[:, 0], inlier_points1[:, 1], color_panel[1], ms=2)
    ax1.plot(inlier_points2[:, 0], inlier_points2[:, 1], color_panel[1], ms=2)
    ax1.plot(laser_pc_person[:, 0], laser_pc_person[:, 1], color_panel[-1], ms=2)


ani = animation.FuncAnimation(
    fig, visualize, gen_data, interval=100,
    init_func=init_fig, repeat=False, save_count=1000
)
if save_gif:
    ani.save(f"{out_path}/gifs/{file_name}.gif", writer='pillow')
else:
    plt.show()
