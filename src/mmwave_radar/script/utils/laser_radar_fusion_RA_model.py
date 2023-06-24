import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
from sklearn.cluster import DBSCAN
import mmwave.dsp as dsp
import pickle
from mmwave.dsp.utils import Window
from music import aoa_music_1D_mat
import pandas as pd
import os
import cv2
import torch
import sys
sys.path.append('/home/agent/Code/yolov5')
from utils.general import non_max_suppression, xyxy2xywh

from nlos_sensing import transform, bounding_box, intersection_of_2line, isin_triangle
from nlos_sensing import get_span, find_end_point, fit_line_ransac, line_by_coef_p, line_by_vertical_coef_p
from nlos_sensing import transform_inverse, parallel_line_distance, line_symmetry_point, line_by_2p


# 读取毫米波雷达参数
xwr_cfg = "/home/agent/Code/ackermann_car_nav/src/mmwave_radar/config/best_range_res.cfg"
for line in open(xwr_cfg):
    line = line.rstrip('\r\n')
    if line.startswith('profileCfg'):
        config = line.split(' ')
        start_freq = float(config[2])
        idle_time = float(config[3])
        ramp_end_time = float(config[5])
        freq_slop = float(config[8])
        num_samples = int(config[10])
        dig_out_sample_rate = float(config[11])
    elif line.startswith('frameCfg'):
        config = line.split(' ')
        num_chirps = int(config[3])
        frame_periodicity = float(config[5])
num_rx = 4
num_tx = 3
virt_ant_azimuth = 8
virt_ant_elevation = 2
angle_range_azimuth = 90
angle_range_elevation = 15
angle_res = 1
angle_bins_azimuth = (angle_range_azimuth * 2) // angle_res + 1
angle_bins_elevation = (angle_range_elevation * 2) // angle_res + 1
begin_range, end_range = 0, 159  # 1.4-5.6
W = end_range + 1
range_res, bandwidth = dsp.range_resolution(num_samples, dig_out_sample_rate, freq_slop)
doppler_res = dsp.doppler_resolution(bandwidth, start_freq, ramp_end_time, idle_time, num_chirps, num_tx)
frame_bytes = num_samples * num_chirps * num_tx * num_rx * 2 * 2
print("range resolution: ", range_res)
print("doppler resolution: ", doppler_res)
print("frame bytes: ", frame_bytes)
# 生成RA tensor和毫米波点云的代码
def gen_point_cloud_plan3(adc_data):
    # 2. 整理数据格式 Tx*num_chirps, num_rx, num_samples
    # adc_data 48 x 4 x 256
    ret = np.zeros(len(adc_data) // 2, dtype=complex)
    ret[0::2] = 1j * adc_data[0::4] + adc_data[2::4]
    ret[1::2] = 1j * adc_data[1::4] + adc_data[3::4]
    adc_data = ret.reshape((num_chirps*num_tx, num_rx, num_samples))

    # 3. range fft, 48 x 4 x 256
    radar_cube = dsp.range_processing(adc_data, window_type_1d=Window.BLACKMAN)
    # 4. Doppler processing, 256x16, 256x12x16
    det_matrix, aoa_input = dsp.doppler_processing(radar_cube, num_tx_antennas=num_tx, clutter_removal_enabled=True, window_type_2d=Window.HAMMING)
    # 5. MUSIC aoa
    # 100 x 16 x 8
    azimuthInput = aoa_input[begin_range:end_range+1, :8, :].transpose(0, 2, 1)
    _, steering_vec_azimuth = dsp.gen_steering_vec(angle_range_azimuth, angle_res, virt_ant_azimuth)
    # 100 x 16 x 181
    spectrum = aoa_music_1D_mat(steering_vec_azimuth, azimuthInput[..., np.newaxis])
    # 6. RA CFAR
    # 100 x 181
    RA = np.mean(spectrum, axis=1)
    RA_log = np.log2(RA)

    # --- cfar in azimuth direction
    first_pass, _ = np.apply_along_axis(func1d=dsp.ca_,
                                        axis=0,
                                        arr=RA_log.T,
                                        l_bound=1,
                                        guard_len=4,
                                        noise_len=16)

    # --- cfar in range direction
    second_pass, noise_floor = np.apply_along_axis(func1d=dsp.ca_,
                                                axis=0,
                                                arr=RA_log,
                                                l_bound=1,
                                                guard_len=4,
                                                noise_len=16)

    # --- classify peaks and caclulate snrs
    first_pass = (RA_log > first_pass.T)
    second_pass = (RA_log > second_pass)
    peaks = (first_pass & second_pass)
    pairs = np.argwhere(peaks)
    if len(pairs) < 1:
        return None
    ranges, azimuths = pairs[:, 0], pairs[:, 1]

    # RD转化到笛卡尔坐标系下可视化
    axis_range = np.arange(W).reshape(-1, 1) * range_res
    axis_azimuth = np.arange(angle_bins_azimuth).reshape(1, -1) * np.pi / 180
    xs_idx = axis_range * np.cos(axis_azimuth) // range_res
    ys_idx = axis_range * np.sin(axis_azimuth) // range_res
    df = pd.DataFrame({
        'x_idx': xs_idx.flatten().astype(np.int32),
        'y_idx': ys_idx.flatten().astype(np.int32),
        'rcs': RA.flatten()
    })
    df_group = df.groupby(['x_idx', 'y_idx'], as_index=False).mean()
    xs_idx2 = (df_group.x_idx + W).to_numpy()
    ys_idx2 = df_group.y_idx.to_numpy()
    rcs = df_group.rcs.to_numpy()
    bbox = np.zeros((W, W*2, 1))
    bbox[..., 0] = rcs.min()
    bbox[ys_idx2, xs_idx2, 0] = rcs

    # 7. doppler estimation
    dopplers = np.argmax(spectrum[ranges, :, azimuths], axis=1)

    # convert bins to units 
    azimuths = azimuths * angle_res * np.pi / 180
    ranges = (ranges + begin_range) * range_res
    dopplers = (dopplers - num_chirps // 2) * doppler_res

    x_pos = ranges * np.cos(azimuths)
    y_pos = ranges * np.sin(azimuths)
    point_cloud = np.array([x_pos, y_pos, dopplers]).T

    # 增加速度特征
    # xs_idx2 = (x_pos // range_res).astype(np.int32) + W
    # ys_idx2 = (y_pos // range_res).astype(np.int32)
    # bbox[ys_idx2, xs_idx2, 1] = dopplers
    return bbox, point_cloud


# Model
weights = "/home/agent/Code/yolov5/runs/train/exp/weights/best.pt"
device = "cuda:0"
model = torch.load(weights, map_location='cpu')
model = model['model'].to(device).float()
model.eval()
conf_thres = 0.25
iou_thres = 0.45
max_det = 1000

file_path = "/home/agent/Code/ackermann_car_nav/data/20230530/floor31_h1_120_L_120_angle_30_param1_2023-05-30-15-58-38.pkl"
file_name = file_path.split('/')[-1].split('.')[0][:-20]
out_path = "/home/agent/Code/ackermann_car_nav/data/data_20230530"
mode = "train"
if not os.path.exists(f"{out_path}/images/{mode}"):
    os.makedirs(f"{out_path}/images/{mode}")
if not os.path.exists(f"{out_path}/labels/{mode}"):
    os.makedirs(f"{out_path}/labels/{mode}")
if not os.path.exists(f"{out_path}/gifs"):
    os.makedirs(f"{out_path}/gifs")
save_gif = True
plot_radar_pc = True
cnt = 0
local_sensing_range = [-0.5, 5, -3, 3]
gt_range = [-8, 2, 0, 1.5]  # 切割人的点云
min_points_inline = 20
min_length_inline = 0.6
ransac_sigma = 0.02
ransac_iter = 200
filter = DBSCAN(eps=1, min_samples=20)
stamp, accuracy = [], []
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(16, 8))
color_panel = ['ro', 'go', 'bo', 'co', 'yo', 'mo', 'ko']


def init_fig():
    ax.clear()
    ax.set_xlabel('x(m)')
    ax.set_ylabel('y(m)')
    ax.set_xlim([-7, 5])
    ax.set_ylim([-2, 8])
    ax.tick_params(direction='in')
    ax2.clear()
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')


def gen_data():
    with open(file_path, 'rb') as f:
        all_point_cloud = pickle.load(f)
    for t, laser_pc, laser_pc2, mmwave_pc, mmwave_raw_data, trans in all_point_cloud:
        inter, theta = trans
        theta = theta * 180 / np.pi
        
        # 小车激光雷达只保留小车附近的点
        # 激光雷达->小车
        laser_point_cloud = transform(laser_pc, 0.08, 0, 180)
        # 过滤激光雷达点云，去掉距离小车中心5米以外的点
        flag_x = np.logical_and(laser_point_cloud[:, 0]>=local_sensing_range[0], laser_point_cloud[:, 0]<=local_sensing_range[1])
        flag_y = np.logical_and(laser_point_cloud[:, 1]>=local_sensing_range[2], laser_point_cloud[:, 1]<=local_sensing_range[3])
        flag = np.logical_and(flag_x, flag_y) 
        laser_point_cloud = laser_point_cloud[flag]
        # 小车->毫米波雷达
        laser_point_cloud = transform_inverse(laser_point_cloud, 0.17, 0, 90)

        # 生成毫米波RA tensor和点云
        result = gen_point_cloud_plan3(mmwave_raw_data)
        if result is None:
            continue
        RA_cart, mmwave_point_cloud = result
        
        # 标定激光雷达点云只保留人
        flag_x = np.logical_and(laser_pc2[:, 0]>=gt_range[0], laser_pc2[:, 0]<=gt_range[1])
        flag_y = np.logical_and(laser_pc2[:, 1]>=gt_range[2], laser_pc2[:, 1]<=gt_range[3])
        flag = np.logical_and(flag_x, flag_y) 
        laser_point_cloud2 = laser_pc2[flag]
        # 标定激光雷达->小车坐标系->毫米波雷达坐标系
        laser_point_cloud2 = transform_inverse(laser_point_cloud2, inter[0], inter[1], 360-theta)
        laser_point_cloud2 = transform_inverse(laser_point_cloud2, 0.17, 0, 90)
        yield t, laser_point_cloud2, laser_point_cloud, RA_cart, mmwave_point_cloud, mmwave_pc


def visualize(result):
    global cnt
    init_fig()
    t, laser_point_cloud2, laser_point_cloud, RA_cart, mmwave_point_cloud, mmwave_pc = result

    # 提取墙面
    fitted_lines = []
    for i in range(2):
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
        outlier_mask = np.logical_not(inlier_mask)
        laser_point_cloud = laser_point_cloud[outlier_mask]
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
    gt_center = np.mean(laser_point_cloud2, axis=0)
    # 当人位于边界右边，开始预测
    if np.cross(inter3-inter1, gt_center-inter1) > 0:
        # 毫米波点云过滤和映射
        flag = isin_triangle(symmtric_corner, inter2, inter1, mmwave_point_cloud[:, :2])
        point_cloud_nlos = mmwave_point_cloud[flag]
        point_cloud_nlos[:, :2] = line_symmetry_point(far_wall, point_cloud_nlos[:, :2])
        # 激光点云映射
        flag = isin_triangle(barrier_corner, inter1, inter3, gt_center)
        laser_point_cloud2 = line_symmetry_point(far_wall, laser_point_cloud2) if flag else laser_point_cloud2
        # bounding box ground truth
        key_points, box_length, box_width = bounding_box(laser_point_cloud2, far_wall, delta_x=0, delta_y=0)
        center, top_right, bottom_right, bottom_left, top_left = key_points
        x = [top_right[0], bottom_right[0], bottom_left[0], top_left[0], top_right[0]]
        y = [top_right[1], bottom_right[1], bottom_left[1], top_left[1], top_right[1]]
        ax.plot(x, y, 'k-', lw=1)
        ax.plot(*center, color_panel[-1], ms=2)

        # Image
        RA_cart = (RA_cart - RA_cart.min()) / (RA_cart.max() - RA_cart.min())
        RA_cart = (RA_cart * 255).astype('uint8')
        img = np.concatenate([RA_cart, RA_cart, RA_cart], axis=-1).transpose(2, 0, 1)
        img = torch.from_numpy(img).to(device)
        img = img.float() / 255
        img = img[None]
        # Inference
        pred, loss = model(img, augment=False)
        pred = non_max_suppression(pred, conf_thres, iou_thres, max_det=max_det)
        for det in pred:
            if len(det):
                xyxy, conf, cls = det[0, :4], det[0, 4], det[0, 5]
                xywh = xyxy2xywh(xyxy).tolist()
                pred_center = np.array([
                    (xywh[0]-W)*range_res, xywh[1]*range_res
                ])
                # 距离中心点一半宽度的两条直线
                half_h, half_w = xywh[3]*range_res / 2, xywh[2]*range_res / 2
                pred_center_line = line_by_coef_p(far_wall, pred_center)
                pred_center_line2 = line_by_vertical_coef_p(far_wall, pred_center)
                pred_parallel_wall1, pred_parallel_wall2 = parallel_line_distance(pred_center_line, half_h)
                pred_vertical_wall1, pred_vertical_wall2 = parallel_line_distance(pred_center_line2, half_w)
                p1 = intersection_of_2line(pred_parallel_wall1, pred_vertical_wall1)
                p2 = intersection_of_2line(pred_parallel_wall2, pred_vertical_wall1)
                p3 = intersection_of_2line(pred_parallel_wall1, pred_vertical_wall2)
                p4 = intersection_of_2line(pred_parallel_wall2, pred_vertical_wall2)
                x = [p1[0], p2[0], p4[0], p3[0], p1[0]]
                y = [p1[1], p2[1], p4[1], p3[1], p1[1]]
                ax.plot(x, y, 'g-', lw=1)
                ax.plot(*pred_center, color_panel[1], ms=2)

        # 保存你想要的
        if not save_gif:
            image_path = f"{out_path}/images/{mode}/{file_name}_{cnt}.png"
            cv2.imwrite(image_path, RA_cart)
            txt_path = f"{out_path}/labels/{mode}/{file_name}_{cnt}.txt"
            fwrite = open(txt_path, 'w')
            cnt += 1
            x_norm = (center[0]+W*range_res) / (W*2*range_res)
            y_norm = center[1] / (W*range_res)
            length_norm = box_length / (W*2*range_res)
            width_norm = box_width / (W*range_res)
            fwrite.write(f"0 {x_norm} {y_norm} {length_norm} {width_norm}\n")
            fwrite.close()

    # 可视化所有结果
    ax.set_title(f"Timestamp: {t:.2f}s")
    # 毫米波雷达
    if plot_radar_pc:
        ax.plot(point_cloud_nlos[:, 0], point_cloud_nlos[:, 1], color_panel[3], ms=2)
        static_idx = np.abs(mmwave_point_cloud[:, 2]) <= doppler_res
        dynamic_idx = np.abs(mmwave_point_cloud[:, 2]) > doppler_res
        ax.plot(mmwave_point_cloud[static_idx, 0], mmwave_point_cloud[static_idx, 1], color_panel[2], ms=2)
        ax.plot(mmwave_point_cloud[dynamic_idx, 0], mmwave_point_cloud[dynamic_idx, 1], color_panel[0], ms=2)
        ax.plot(mmwave_pc[:, 0], mmwave_pc[:, 1], color_panel[4], ms=2)
    ax2.imshow(RA_cart[..., 0])
    # 激光雷达
    ax.plot(inlier_points1[:, 0], inlier_points1[:, 1], color_panel[1], ms=2)
    ax.plot(inlier_points2[:, 0], inlier_points2[:, 1], color_panel[1], ms=2)
    ax.plot(laser_point_cloud2[:, 0], laser_point_cloud2[:, 1], color_panel[-1], ms=2)


ani = animation.FuncAnimation(
    fig, visualize, gen_data, interval=200,
    init_func=init_fig, repeat=False, save_count=500
)
writergif = animation.PillowWriter(fps=10)
if save_gif:
    gif_path = f"{out_path}/gifs/{file_name}.gif"
    ani.save(gif_path, writer=writergif)
plt.show()
