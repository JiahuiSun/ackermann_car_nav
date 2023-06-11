import numpy as np
import mmwave.dsp as dsp
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from mmwave.dsp.utils import Window
from music import aoa_music_1D, aoa_music_1D_mat
from nlos_sensing import transform
import time
import struct
import rosbag


is_bag = 1
bag_file = "/home/dingrong/Code/ackermann_car_nav/data/20230530/floor31_h1_120_L_120_angle_30_param1_2023-05-30-15-58-38.bag"
xwr_cfg = "/home/dingrong/Code/ackermann_car_nav/src/mmwave_radar/config/best_range_res.cfg"
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
angle_range_azimuth = 60
angle_range_elevation = 15
angle_res = 1
angle_bins_azimuth = (angle_range_azimuth * 2) // angle_res + 1
angle_bins_elevation = (angle_range_elevation * 2) // angle_res + 1
begin_range, end_range = 32, 128  # 1.4-5.6
range_res, bandwidth = dsp.range_resolution(num_samples, dig_out_sample_rate, freq_slop)
doppler_res = dsp.doppler_resolution(bandwidth, start_freq, ramp_end_time, idle_time, num_chirps, num_tx)
frame_bytes = num_samples * num_chirps * num_tx * num_rx * 2 * 2
print("range resolution: ", range_res)
print("doppler resolution: ", doppler_res)
print("frame bytes: ", frame_bytes)

fig, ax = plt.subplots(figsize=(8, 8))
line0, = ax.plot([], [], 'ob', ms=2)
line1, = ax.plot([], [], 'or', ms=2)
lines = [line0, line1]

def init_fig():
    ax.set_xlabel('x(m)')
    ax.set_ylabel('y(m)')
    ax.set_xlim([-5, 10])
    ax.set_ylim([-5, 10])
    return lines

def gen_data():
    # 读取原始数据
    if is_bag:
        for topic, msg, t in rosbag.Bag(bag_file, 'r'):
            if topic == '/mmwave_radar_raw_data':
                adc_pack = struct.pack(f">{frame_bytes}b", *msg.data)
                adc_unpack = np.frombuffer(adc_pack, dtype=np.int16)
                yield adc_unpack, msg.header.seq
    else:
        # 1. 订阅数据，读取参数
        for cnt in range(100):
        # 1. 订阅数据，读取参数
            data_path = f"/home/dingrong/Code/ackermann_car_nav/data/person_walk/test_{cnt+1}.bin"
            adc_data = np.fromfile(data_path, dtype=np.int16)
            yield adc_data, cnt

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
    det_matrix, aoa_input = dsp.doppler_processing(radar_cube, num_tx_antennas=3, clutter_removal_enabled=False, window_type_2d=Window.HAMMING)

    # 5. MUSIC aoa
    # 100 x 16 x 8
    azimuthInput = aoa_input[begin_range:end_range+1, :8, :].transpose(0, 2, 1)
    _, steering_vec_azimuth = dsp.gen_steering_vec(angle_range_azimuth, angle_res, virt_ant_azimuth)
    # 100 x 16 x 181
    spectrum = aoa_music_1D_mat(steering_vec_azimuth, azimuthInput[..., np.newaxis])
    
    # 6. RA CFAR
    # 100 x 181
    RA = np.mean(spectrum, axis=1)
    heatmap_log = np.log2(RA)

    # --- cfar in azimuth direction
    first_pass, _ = np.apply_along_axis(func1d=dsp.ca_,
                                        axis=0,
                                        arr=heatmap_log.T,
                                        l_bound=1,
                                        guard_len=4,
                                        noise_len=16)

    # --- cfar in range direction
    second_pass, noise_floor = np.apply_along_axis(func1d=dsp.ca_,
                                                axis=0,
                                                arr=heatmap_log,
                                                l_bound=1,
                                                guard_len=4,
                                                noise_len=16)

    # --- classify peaks and caclulate snrs
    first_pass = (heatmap_log > first_pass.T)
    second_pass = (heatmap_log > second_pass)
    peaks = (first_pass & second_pass)
    pairs = np.argwhere(peaks)
    if len(pairs) < 1:
        return None
    ranges, azimuths = pairs[:, 0], pairs[:, 1]
    snrs = heatmap_log[ranges, azimuths] - noise_floor[ranges, azimuths]

    # doppler estimation
    dopplers = np.argmax(spectrum[ranges, :, azimuths], axis=1)

    # convert bins to units 
    azimuths = (azimuths - (angle_bins_azimuth // 2)) * (np.pi / 180)
    ranges = (ranges + begin_range) * range_res
    dopplers[dopplers >= num_chirps/2] -= num_chirps
    dopplers = dopplers * doppler_res

    x_pos = -ranges * np.sin(azimuths)
    y_pos = ranges * np.cos(azimuths)
    z_pos = np.zeros_like(ranges)
    return x_pos, y_pos, z_pos, dopplers, snrs


def visualize(result):
    adc_data, seq = result
    st = time.time()
    point_cloud = gen_point_cloud_plan3(adc_data)
    end = time.time()
    print(f"{end-st:.3f}s")
    if point_cloud is not None:
        x_pos, y_pos, z_pos, dopplers, snrs = point_cloud
        point_cloud = np.array([x_pos, y_pos]).T
        point_cloud = transform(point_cloud, 0.17, 0, 60)
        static_idx = dopplers == 0
        dynamic_idx = dopplers != 0
        ax.set_title(f"frame id: {seq}")
        lines[0].set_data(point_cloud[static_idx, 0], point_cloud[static_idx, 1])
        lines[1].set_data(point_cloud[dynamic_idx, 0], point_cloud[dynamic_idx, 1])


ani = animation.FuncAnimation(
    fig, visualize, gen_data, interval=100,
    init_func=init_fig, repeat=True, save_count=100
)
# ani.save("plan1.gif", writer='imagemagick')
plt.show()
