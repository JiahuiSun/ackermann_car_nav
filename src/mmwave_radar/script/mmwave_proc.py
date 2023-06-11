#!/usr/bin/env python3
import rospy
from mmwave_radar.msg import adcData
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField
import numpy as np
import struct
import mmwave.dsp as dsp
import time
from utils.music import aoa_music_1D
from mmwave.dsp.utils import Window
import sys


# 参数设置
xwr_cfg = sys.argv[1]
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
bins_processed = max(128, num_samples)
skip_size = 4
range_res, bandwidth = dsp.range_resolution(num_samples, dig_out_sample_rate, freq_slop)
doppler_res = dsp.doppler_resolution(bandwidth, start_freq, ramp_end_time, idle_time, num_chirps, num_tx)
frame_bytes = num_samples * num_chirps * num_tx * num_rx * 2 * 2
print("range resolution: ", range_res)
print("doppler resolution: ", doppler_res)
print("frame bytes: ", frame_bytes)
pub = rospy.Publisher("mmwave_radar_point_cloud", PointCloud2, queue_size=10)


def gen_point_cloud_plan2(adc_data):
    # 2. 整理数据格式 num_chirps, num_rx, num_samples
    ret = np.zeros(len(adc_data) // 2, dtype=complex)
    ret[0::2] = 1j * adc_data[0::4] + adc_data[2::4]
    ret[1::2] = 1j * adc_data[1::4] + adc_data[3::4]
    adc_data = ret.reshape((num_chirps*num_tx, num_rx, num_samples))

    # 3. range fft
    radar_cube = dsp.range_processing(adc_data, window_type_1d=Window.BLACKMAN)

    # 4. angle estimation
    range_azimuth = np.zeros((angle_bins_azimuth, bins_processed))
    beamWeights = np.zeros((virt_ant_azimuth, bins_processed), dtype=np.complex_)
    _, steering_vec_azimuth = dsp.gen_steering_vec(angle_range_azimuth, angle_res, virt_ant_azimuth)
    # static clutter removal, only detect moving objects
    # radar_cube = radar_cube - radar_cube.mean(0)
    radar_cube_azimuth = np.concatenate((radar_cube[0::3, ...], radar_cube[1::3, ...]), axis=1)
    for i in range(bins_processed):
        range_azimuth[:, i], beamWeights[:, i] = dsp.aoa_capon(radar_cube_azimuth[:, :, i].T, steering_vec_azimuth, magnitude=True)

    # 5. object detection
    heatmap_log = np.log2(range_azimuth)
        
    # --- cfar in azimuth direction
    first_pass, _ = np.apply_along_axis(func1d=dsp.ca_,
                                        axis=0,
                                        arr=heatmap_log,
                                        l_bound=2.5,
                                        guard_len=4,
                                        noise_len=16)

    # --- cfar in range direction
    second_pass, noise_floor = np.apply_along_axis(func1d=dsp.ca_,
                                                axis=0,
                                                arr=heatmap_log.T,
                                                l_bound=3.5,
                                                guard_len=4,
                                                noise_len=16)

    # --- classify peaks and caclulate snrs
    noise_floor = noise_floor.T
    first_pass = (heatmap_log > first_pass)
    second_pass = (heatmap_log > second_pass.T)
    peaks = (first_pass & second_pass)
    peaks[:skip_size, :] = 0
    peaks[-skip_size:, :] = 0
    peaks[:, :skip_size] = 0
    peaks[:, -skip_size:] = 0
    pairs = np.argwhere(peaks)
    azimuths, ranges = pairs.T
    snrs = heatmap_log[pairs[:,0], pairs[:,1]] - noise_floor[pairs[:,0], pairs[:,1]]

    # 6. doppler estimation
    # --- get peak indices
    # beamWeights should be selected based on the range indices from CFAR.
    dopplerFFTInput = radar_cube_azimuth[:, :, ranges]
    beamWeights  = beamWeights[:, ranges]

    # --- estimate doppler values
    # For each detected object and for each chirp combine the signals from 4 Rx, i.e.
    # For each detected object, matmul (numChirpsPerFrame, numRxAnt) with (numRxAnt) to (numChirpsPerFrame)
    dopplerFFTInput = np.einsum('ijk,jk->ik', dopplerFFTInput, beamWeights)
    dopplerEst = np.fft.fft(dopplerFFTInput, axis=0)
    dopplerEst = np.argmax(dopplerEst, axis=0)
    dopplerEst[dopplerEst[:]>=num_chirps/2] -= num_chirps

    # 7. convert bins to units 
    ranges = ranges * range_res
    azimuths = (azimuths - (angle_bins_azimuth // 2)) * (np.pi / 180)
    dopplers = dopplerEst * doppler_res
    snrs = snrs

    # 8. generate point cloud
    x_pos = -ranges * np.sin(azimuths)
    y_pos = ranges * np.cos(azimuths)
    z_pos = np.zeros_like(ranges)
    return x_pos, y_pos, z_pos, dopplers, snrs


def gen_point_cloud_plan1(adc_data):
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
    snrs = detObj2D['SNR']

    x_pos = -ranges * np.sin(azimuths) * np.cos(elevations)
    y_pos = ranges * np.cos(azimuths) * np.cos(elevations)
    z_pos = ranges * np.sin(elevations)
    return x_pos, y_pos, z_pos, dopplers, snrs


def pub_point_cloud(adcData):
    global pub
    adc_pack = struct.pack(f">{frame_bytes}b", *adcData.data)
    adc_unpack = np.frombuffer(adc_pack, dtype=np.int16)
    x_pos, y_pos, z_pos, velocity, snr = gen_point_cloud_plan1(adc_unpack)
    points = np.array([x_pos, y_pos, z_pos, velocity, snr]).T
    msg = PointCloud2()
    msg.header = adcData.header
    msg.height = 1
    msg.width = points.shape[0]
    msg.fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('vel', 12, PointField.FLOAT32, 1),
        PointField('snr', 16, PointField.FLOAT32, 1)
    ]
    msg.is_bigendian = False
    msg.point_step = 20
    msg.row_step = msg.point_step * points.shape[0]
    msg.is_dense = True
    msg.data = np.asarray(points, np.float32).tostring()
    pub.publish(msg)


rospy.init_node("mmwave_radar_subscriber")
sub = rospy.Subscriber("mmwave_radar_raw_data", adcData, pub_point_cloud, queue_size=10)
rospy.spin()
