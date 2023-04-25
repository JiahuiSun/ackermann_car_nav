#!/usr/bin/env python3
import rospy
from mmwave_radar.msg import adcData
from mmwave_radar.msg import mmwavePointCloud
import numpy as np
import struct
import mmwave.dsp as dsp
import time


num_chirps = 16
num_samples = 256
num_rx = 4
num_tx = 3
virt_ant_azimuth = 8
virt_ant_elevation = 2

angle_range_azimuth = 90
angle_range_elevation = 15
angle_res = 1
angle_bins = (angle_range_azimuth * 2) // angle_res + 1
bins_processed = 112
skip_size = 4
range_res = 0.044
doppler_res = 0.13

pub = rospy.Publisher("mmwave_radar_point_cloud", mmwavePointCloud, queue_size=10)

def gen_point_cloud(adc_data):
    # 2. 整理数据格式 num_chirps, num_rx, num_samples
    ret = np.zeros(len(adc_data) // 2, dtype=complex)
    ret[0::2] = 1j * adc_data[0::4] + adc_data[2::4]
    ret[1::2] = 1j * adc_data[1::4] + adc_data[3::4]
    adc_data = ret.reshape((num_chirps*num_tx, num_rx, num_samples))

    # 3. range fft
    radar_cube = dsp.range_processing(adc_data)

    # 4. angle estimation
    range_azimuth = np.zeros((angle_bins, bins_processed))
    beamWeights = np.zeros((virt_ant_azimuth, bins_processed), dtype=np.complex_)
    _, steering_vec_azimuth = dsp.gen_steering_vec(angle_range_azimuth, angle_res, virt_ant_azimuth)
    # static clutter removal, only detect moving objects
    radar_cube = radar_cube - radar_cube.mean(0)
    radar_cube_azimuth = np.concatenate((radar_cube[0::3, ...], radar_cube[1::3, ...]), axis=1)
    for i in range(bins_processed):
        range_azimuth[:, i], beamWeights[:, i] = dsp.aoa_capon(radar_cube_azimuth[:, :, i].T, steering_vec_azimuth, magnitude=True)

    # 5. object detection
    heatmap_log = np.log2(range_azimuth)
        
    # --- cfar in azimuth direction
    first_pass, _ = np.apply_along_axis(func1d=dsp.ca_,
                                        axis=0,
                                        arr=heatmap_log,
                                        l_bound=1.5,
                                        guard_len=4,
                                        noise_len=16)

    # --- cfar in range direction
    second_pass, noise_floor = np.apply_along_axis(func1d=dsp.ca_,
                                                axis=0,
                                                arr=heatmap_log.T,
                                                l_bound=2.5,
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
    azimuths = (azimuths - (angle_bins // 2)) * (np.pi / 180)
    dopplers = dopplerEst * doppler_res
    snrs = snrs

    # 8. generate point cloud
    x_pos = -ranges * np.sin(azimuths)
    y_pos = ranges * np.cos(azimuths)
    return x_pos, y_pos, dopplers


def pub_point_cloud(adcData):
    st1 = time.time()
    global pub
    adc_pack = struct.pack(">196608b", *adcData.data)
    adc_unpack = np.frombuffer(adc_pack, dtype=np.int16)
    st2 = time.time()
    x_pos, y_pos, velocity = gen_point_cloud(adc_unpack)
    # 在python下组织消息格式，还从未操作过
    point_cloud = mmwavePointCloud()
    point_cloud.header = adcData.header
    point_cloud.size = adcData.size
    point_cloud.x_pos = x_pos
    point_cloud.y_pos = y_pos
    point_cloud.velocity = velocity
    pub.publish(point_cloud)
    st3 = time.time()
    print(f"parse data cost: {st2-st1}s")
    print(f"gen point cloud cost: {st3-st2}s")
    print(f"total cost: {st3-st1}s")


if __name__ == "__main__":
    rospy.init_node("mmwave_radar_subscriber")
    sub = rospy.Subscriber("mmwave_radar_raw_data", adcData, pub_point_cloud, queue_size=10)
    rospy.spin()
