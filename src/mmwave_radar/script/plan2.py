import numpy as np
import mmwave.dsp as dsp
import matplotlib.animation as animation
import matplotlib.pyplot as plt
# from mmwave.tracking import EKF
from mmwave.dsp.utils import Window


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
bins_processed = 128
skip_size = 4
range_res = 0.044
doppler_res = 0.13

fig, ax = plt.subplots(figsize=(10, 4))
line0, = ax.plot([], [], 'ob', ms=2)
line1, = ax.plot([], [], 'or', ms=2)
lines = [line0, line1]

# tracker = EKF()


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
    # global tracker
    # 2. 整理数据格式 num_chirps, num_rx, num_samples
    ret = np.zeros(len(adc_data) // 2, dtype=complex)
    ret[0::2] = 1j * adc_data[0::4] + adc_data[2::4]
    ret[1::2] = 1j * adc_data[1::4] + adc_data[3::4]
    adc_data = ret.reshape((num_chirps*num_tx, num_rx, num_samples))

    # 3. range fft
    radar_cube = dsp.range_processing(adc_data, window_type_1d=Window.BLACKMAN)

    # 4. angle estimation
    range_azimuth = np.zeros((angle_bins, bins_processed))
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
    azimuths = (azimuths - (angle_bins // 2)) * (np.pi / 180)
    dopplers = dopplerEst * doppler_res
    snrs = snrs

    # --- put into EKF
    # tracker.update_point_cloud(ranges, azimuths, dopplers, snrs)
    # targetDescr, tNum = tracker.step()
    # x_pos_list, y_pos_list = [], []
    # for target in targetDescr:
    #     x_pos, y_pos, x_vel, y_vel = target.S[:4]
    #     vec_mag = np.sqrt(x_vel ** 2 + y_vel ** 2)
    #     if vec_mag < 0.2:
    #         continue
    #     x_pos_list.append(-x_pos)
    #     y_pos_list.append(y_pos)
    # lines[0].set_data(x_pos_list, y_pos_list)
    # lines[1].set_data(x_pos, y_pos)

    x_pos = -ranges * np.sin(azimuths)
    y_pos = ranges * np.cos(azimuths)
    static_idx = dopplers == 0
    dynamic_idx = dopplers != 0
    lines[0].set_data(x_pos[static_idx], y_pos[static_idx])
    lines[1].set_data(x_pos[dynamic_idx], y_pos[dynamic_idx])


ani = animation.FuncAnimation(
    fig, visualize, gen_data, interval=100,
    init_func=init_fig, repeat=False, save_count=100
)
ani.save("plan2.gif", writer='imagemagick')
plt.show()
