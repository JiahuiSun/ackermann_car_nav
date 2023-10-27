import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
import pickle
import os
import cv2
import sys

from radar_fft_music_RA import *
from nlos_sensing import transform, bounding_box2, isin_triangle, transform_inverse, line_symmetry_point, pc_filter
from corner_type import L_open_corner


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
    for t, laser_pc_onboard, laser_pc_person, mmwave_pc, mmwave_raw_data in all_point_cloud:
        # 小车激光雷达只保留小车附近的点
        # 激光雷达->小车
        laser_pc_onboard = transform(laser_pc_onboard, 0.08, 0, 180)
        laser_pc_onboard = pc_filter(laser_pc_onboard, *local_sensing_range)
        if len(laser_pc_onboard) < min_onboard_laser_point_num:
            continue
        # 小车->毫米波雷达
        laser_pc_onboard = transform_inverse(laser_pc_onboard, 0.17, 0, 360-90)

        # 生成毫米波RA tensor和点云
        result = gen_point_cloud_plan3(mmwave_raw_data)
        if result is None:
            continue
        RA_cart, mmwave_point_cloud = result

        if len(laser_pc_person) < min_GT_laser_point_num:
            continue
        # 标定激光雷达->小车坐标系->毫米波雷达坐标系
        laser_pc_person = transform_inverse(
            transform(laser_pc_person, 0.08, 0, 180), 
            0.17, 0, 360-90
        )
        yield t, laser_pc_person, laser_pc_onboard, RA_cart, mmwave_point_cloud


def visualize(result):
    global cnt
    init_fig()
    t, laser_pc_person, laser_pc_onboard, RA_cart, mmwave_point_cloud = result

    onboard_walls, onboard_points = L_open_corner(laser_pc_onboard)
    far_wall, far_wall_pc, barrier_wall_pc = onboard_walls['far_wall'], onboard_walls['far_wall_pc'], onboard_walls['barrier_wall_pc']
    barrier_corner, symmtric_corner = onboard_points['barrier_corner'], onboard_points['symmetric_barrier_corner']
    inter1, inter2, inter3 = onboard_points['inter1'], onboard_points['inter2'], onboard_points['inter3']
    ax1.plot(*inter1, color_panel[-2], ms=5)
    ax1.plot(*inter2, color_panel[-2], ms=5)
    ax1.plot(*inter3, color_panel[-2], ms=5)
    ax1.plot(*barrier_corner, color_panel[-2], ms=5)
    ax1.plot(*symmtric_corner, color_panel[-2], ms=5)

    # 原来是用激光点云平均值，这个是不准确的，应该是中心点
    key_points, _ = bounding_box2(laser_pc_person, delta_x=0.1, delta_y=0.1)
    gt_center = key_points[0]
    # 只要人进入NLOS就保存数据
    if np.cross(inter3-inter1, gt_center-inter1) > 0:
        # 生成NLOS区域：遍历所有的格子，把这个格子转化为坐标，判断这个坐标是否在三角形内
        pos = np.array([
            np.array([[i for i in range(-H, H)] for j in range(H)]).flatten(),
            np.array([[j for i in range(-H, H)] for j in range(H)]).flatten()
        ]).T * range_res
        mask = isin_triangle(symmtric_corner, inter2, inter1, pos).astype(np.float32).reshape(H, 2*H)

        # 激光点云映射
        if isin_triangle(barrier_corner, inter1, inter3, gt_center):
            laser_pc_person = line_symmetry_point(far_wall, laser_pc_person)
        # bounding box ground truth
        key_points, box_hw = bounding_box2(laser_pc_person, delta_x=0.1, delta_y=0.1)
        # 防止center出界
        key_points[0, 0] = np.clip(key_points[0, 0], -(H-1)*range_res, (H-1)*range_res)
        key_points[0, 1] = np.clip(key_points[0, 1], 0, (H-1)*range_res)
        ax1.plot(*key_points[0], color_panel[-1], ms=2)
        rect = patches.Rectangle(key_points[3], box_hw[1], box_hw[0], linewidth=1, edgecolor='k', facecolor='none')
        ax1.add_patch(rect)

        # 保存你想要的
        if save_data:
            if img_fmt:
                RA_cart = (RA_cart - RA_cart.min()) / (RA_cart.max() - RA_cart.min())
                RA_cart = (RA_cart * 255).astype('uint8')
                image_path = f"{out_path}/images/{mode}/{file_name}_{cnt}.png"
                cv2.imwrite(image_path, RA_cart)
            else:
                RA_cart = (RA_cart - RA_cart.min()) / (RA_cart.max() - RA_cart.min())
                res = np.concatenate([RA_cart, mask[..., np.newaxis]], axis=2)
                image_path = f"{out_path}/images/{mode}/{file_name}_{cnt}.npy"
                np.save(image_path, res)
            txt_path = f"{out_path}/labels/{mode}/{file_name}_{cnt}.txt"
            fwrite = open(txt_path, 'w')
            cnt += 1
            x_norm = (key_points[0, 0]+H*range_res) / (H*2*range_res)
            y_norm = key_points[0, 1] / (H*range_res)
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
    ax1.plot(far_wall_pc[:, 0], far_wall_pc[:, 1], color_panel[1], ms=2)
    ax1.plot(barrier_wall_pc[:, 0], barrier_wall_pc[:, 1], color_panel[1], ms=2)
    ax1.plot(laser_pc_person[:, 0], laser_pc_person[:, 1], color_panel[-1], ms=2)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        out_path = sys.argv[2]
        mode = sys.argv[3]
    else:
        file_path = "/home/agent/Code/ackermann_car_nav/data/20231002/soft-3-A_2023-10-01-20-21-03.pkl"
        out_path = "/home/agent/Code/ackermann_car_nav/data/tmp"
        mode = "train"
    file_name = file_path.split('/')[-1].split('.')[0][:-20]
    os.makedirs(f"{out_path}/images/{mode}", exist_ok=True)
    os.makedirs(f"{out_path}/labels/{mode}", exist_ok=True)
    os.makedirs(f"{out_path}/gifs", exist_ok=True)
    save_gif = True
    save_data = True
    plot_radar_pc = True
    img_fmt = False
    local_sensing_range = [-0.5, 5, -3, 3]  # 切割小车周围点云
    min_onboard_laser_point_num = 20
    min_GT_laser_point_num = 4
    color_panel = ['ro', 'go', 'bo', 'co', 'yo', 'mo', 'ko']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    cnt = 0
    ani = animation.FuncAnimation(
        fig, visualize, gen_data, interval=100,
        init_func=init_fig, repeat=False, save_count=1000
    )
    if save_gif:
        ani.save(f"{out_path}/gifs/{file_name}.gif", writer='pillow')
    else:
        plt.show()
