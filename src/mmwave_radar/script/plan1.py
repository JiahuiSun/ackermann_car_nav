import numpy as np
import mmwave.dsp as dsp
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from mmwave.dsp.utils import Window
from music import aoa_music_1D


num_chirps = 16
num_samples = 256
num_rx = 4
num_tx = 3
virt_ant_azimuth = 8
virt_ant_elevation = 2

angle_range_azimuth = 90
angle_range_elevation = 15
angle_res = 1
angle_bins_azimuth = (angle_range_azimuth * 2) // angle_res + 1
angle_bins_elevation = (angle_range_elevation * 2) // angle_res + 1
range_res = 0.044
doppler_res = 0.13

fig, ax = plt.subplots(figsize=(10, 4))
line0, = ax.plot([], [], 'ob', ms=2)
line1, = ax.plot([], [], 'or', ms=2)
lines = [line0, line1]


def init_fig():
    ax.set_xlabel('x(m)')
    ax.set_ylabel('y(m)')
    ax.set_xlim([-5, 5])
    ax.set_ylim([0, 10])
    ax.plot([-5, 5], [3, 3], 'k')
    return lines

def gen_data():
    for cnt in range(100):
        # 1. 订阅数据，读取参数
        data_path = f"/home/dingrong/Code/ackermann_car_nav/data/person_walk/test_{cnt+1}.bin"
        adc_data = np.fromfile(data_path, dtype=np.int16)
        yield adc_data

def visualize(adc_data):
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

    # 5. Object detection
    fft2d_sum = det_matrix.astype(np.int64)
    # 16x256
    thresholdDoppler, noiseFloorDoppler = np.apply_along_axis(func1d=dsp.ca_,
                                                                axis=0,
                                                                arr=fft2d_sum.T,
                                                                l_bound=20,
                                                                guard_len=2,
                                                                noise_len=4)
    # 256x16
    thresholdRange, noiseFloorRange = np.apply_along_axis(func1d=dsp.ca_,
                                                            axis=0,
                                                            arr=fft2d_sum,
                                                            l_bound=30,
                                                            guard_len=4,
                                                            noise_len=16)

    thresholdDoppler, noiseFloorDoppler = thresholdDoppler.T, noiseFloorDoppler.T
    # 256x16
    det_doppler_mask = (det_matrix > thresholdDoppler)
    det_range_mask = (det_matrix > thresholdRange)
    # Get indices of detected peaks
    full_mask = (det_doppler_mask & det_range_mask)
    det_peaks_indices = np.argwhere(full_mask == True)
    # peakVals and SNR calculation
    peakVals = fft2d_sum[det_peaks_indices[:, 0], det_peaks_indices[:, 1]]
    snr = peakVals - noiseFloorRange[det_peaks_indices[:, 0], det_peaks_indices[:, 1]]
    
    # 聚类
    dtype_location = '(' + str(num_tx) + ',)<f4'
    dtype_detObj2D = np.dtype({'names': ['rangeIdx', 'dopplerIdx', 'peakVal', 'location', 'SNR'],
                                'formats': ['<i4', '<i4', '<f4', dtype_location, '<f4']})
    detObj2DRaw = np.zeros((det_peaks_indices.shape[0],), dtype=dtype_detObj2D)
    detObj2DRaw['rangeIdx'] = det_peaks_indices[:, 0].squeeze()
    detObj2DRaw['dopplerIdx'] = det_peaks_indices[:, 1].squeeze()
    detObj2DRaw['peakVal'] = peakVals.flatten()
    detObj2DRaw['SNR'] = snr.flatten()

    # Further peak pruning. This increases the point cloud density but helps avoid having too many detections around one object.
    # detObj2D = detObj2DRaw
    detObj2D = dsp.prune_to_peaks(detObj2DRaw, det_matrix, num_chirps, reserve_neighbor=False)
    # --- Peak Grouping
    # detObj2D = dsp.peak_grouping_along_doppler(detObj2D, det_matrix, num_chirps)
    # SNRThresholds2 = np.array([[2, 23], [10, 11.5], [35, 16.0]])
    # peakValThresholds2 = np.array([[4, 275], [1, 400], [500, 0]])
    # detObj2D = dsp.range_based_pruning(detObj2D, SNRThresholds2, peakValThresholds2, num_samples, 0.5, range_res)

    # 6. MUSIC aoa
    # num_obj x 8
    azimuthInput = aoa_input[detObj2D['rangeIdx'], :8, detObj2D['dopplerIdx']].T
    elevationInput = np.concatenate([
        aoa_input[detObj2D['rangeIdx'], 2:3, detObj2D['dopplerIdx']],
        aoa_input[detObj2D['rangeIdx'], 8:9, detObj2D['dopplerIdx']]
    ], axis=1).T
    azimuths = np.zeros(detObj2D.shape[0])
    elevations = np.zeros(detObj2D.shape[0])
    
    _, steering_vec_azimuth = dsp.gen_steering_vec(angle_range_azimuth, angle_res, virt_ant_azimuth)
    for i in range(detObj2D.shape[0]):
        spectrum = aoa_music_1D(steering_vec_azimuth, azimuthInput[:, i:i+1], 1)
        azimuths[i] = np.argmax(spectrum)
    _, steering_vec_elevation = dsp.gen_steering_vec(angle_range_elevation, angle_res, virt_ant_elevation)
    for i in range(detObj2D.shape[0]):
        spectrum = aoa_music_1D(steering_vec_elevation, elevationInput[:, i:i+1], 1)
        elevations[i] = np.argmax(spectrum)
    
    # convert bins to units 
    azimuths = (azimuths - (angle_bins_azimuth // 2)) * (np.pi / 180)
    elevations = (elevations - (angle_bins_elevation // 2)) * (np.pi / 180)
    ranges = detObj2D['rangeIdx'] * range_res
    detObj2D['dopplerIdx'][detObj2D['dopplerIdx'] >= num_chirps/2] -= num_chirps
    dopplers = detObj2D['dopplerIdx'] * doppler_res

    x_pos = -ranges * np.sin(azimuths) * np.cos(elevations)
    y_pos = ranges * np.cos(azimuths) * np.cos(elevations)
    z_pos = ranges * np.sin(elevations)
    static_idx = dopplers == 0
    dynamic_idx = dopplers != 0
    lines[0].set_data(x_pos[static_idx], y_pos[static_idx])
    lines[1].set_data(x_pos[dynamic_idx], y_pos[dynamic_idx])


ani = animation.FuncAnimation(
    fig, visualize, gen_data, interval=100,
    init_func=init_fig, repeat=False, save_count=100
)
# ani.save("plan1.gif", writer='imagemagick')
plt.show()
