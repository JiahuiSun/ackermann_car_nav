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
from music import aoa_music_1D, aoa_music_1D_mat
import pandas as pd

from nlos_sensing import transform, nlos_filter_and_mapping, bounding_box
from nlos_sensing import get_span, find_end_point, fit_line_ransac, transform_inverse


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
begin_range, end_range = 0, 191  # 1.4-5.6
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
    det_matrix, aoa_input = dsp.doppler_processing(radar_cube, num_tx_antennas=num_tx, clutter_removal_enabled=False, window_type_2d=Window.HAMMING)
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
    W = end_range + 1
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
    bbox = np.zeros((W, W*2, 2)) 
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
    xs_idx2 = (x_pos // range_res).astype(np.int32) + W
    ys_idx2 = (y_pos // range_res).astype(np.int32)
    bbox[ys_idx2, xs_idx2, 1] = dopplers
    return bbox, point_cloud


gt_range = [-8, 2, 0, 1.5]  # 切割人的点云
fig, ax = plt.subplots(figsize=(8, 8))
color_panel = ['ro', 'go', 'bo', 'co', 'wo', 'yo', 'mo', 'ko']

def init_fig():
    ax.clear()
    ax.set_xlabel('x(m)')
    ax.set_ylabel('y(m)')
    ax.set_xlim([-5, 5])
    ax.set_ylim([0, 10])
    ax.tick_params(direction='in')

file_path = "/home/agent/Code/ackermann_car_nav/data/20230530/floor31_h1_120_L_120_angle_30_param1_2023-05-30-15-58-38"
save_gif = False
def gen_data():
    with open(f"{file_path}.pkl", 'rb') as f:
        all_point_cloud = pickle.load(f)
    for t, laser_pc, laser_pc2, mmwave_pc, mmwave_raw_data, trans in all_point_cloud:
        inter, theta = trans
        theta = theta * 180 / np.pi
        
        # 生成毫米波RA tensor
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
        yield t, laser_point_cloud2, RA_cart, mmwave_point_cloud, mmwave_pc


def visualize(result):
    init_fig()
    t, laser_point_cloud2, RA_cart, mmwave_point_cloud, mmwave_pc = result

    ax.set_title(f"Timestamp: {t:.2f}s")
    ax.plot(laser_point_cloud2[:, 0], laser_point_cloud2[:, 1], color_panel[-3], ms=2)

    static_idx = np.abs(mmwave_point_cloud[:, 2]) <= doppler_res
    dynamic_idx = np.abs(mmwave_point_cloud[:, 2]) > doppler_res
    ax.plot(mmwave_point_cloud[static_idx, 0], mmwave_point_cloud[static_idx, 1], 'ob', ms=2)
    ax.plot(mmwave_point_cloud[dynamic_idx, 0], mmwave_point_cloud[dynamic_idx, 1], 'or', ms=2)
    
    ax.plot(mmwave_pc[:, 0], mmwave_pc[:, 1], color_panel[-2], ms=2)


ani = animation.FuncAnimation(
    fig, visualize, gen_data, interval=200,
    init_func=init_fig, repeat=True, save_count=500
)
writergif = animation.PillowWriter(fps=10)
if save_gif:
    ani.save(f"{file_path}.gif", writer=writergif)
plt.show()
