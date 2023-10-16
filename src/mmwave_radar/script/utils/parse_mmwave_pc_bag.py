import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import rosbag
from sensor_msgs import point_cloud2
import struct

from radar_fft_music_RA import *


file_path = "/home/agent/Code/ackermann_car_nav/data/20231002/soft-3-C3_2023-10-01-21-30-52"
fig, ax = plt.subplots(figsize=(8, 8))
color_panel = ['ro', 'go', 'bo', 'co', 'yo', 'mo', 'ko']


def init_fig():
    ax.clear()
    ax.set_xlabel('x(m)')
    ax.set_ylabel('y(m)')
    ax.set_xlim([-5, 5])
    ax.set_ylim([0, 10])
    ax.tick_params(direction='in')


def gen_data():
    for topic, msg, t in rosbag.Bag(
        f"{file_path}.bag", 'r'):
        if topic == '/mmwave_radar_raw_data':
            adc_pack = struct.pack(f">{frame_bytes}b", *msg.data)
            adc_unpack = np.frombuffer(adc_pack, dtype=np.int16)
            result = gen_point_cloud_plan3(adc_unpack)
            if result is None:
                continue
            RA_cart, mmwave_point_cloud = result
            yield mmwave_point_cloud, msg.header.seq


def gen_data_old():
    for topic, msg, t in rosbag.Bag(
        f"{file_path}.bag", 'r'):
        if topic == '/mmwave_radar_point_cloud':
            points = point_cloud2.read_points_list(
                msg, field_names=['x', 'y', 'z', 'vel']
            )
            x_pos = [p.x for p in points]
            y_pos = [p.y for p in points]
            z_pos = [p.z for p in points]
            vel = [p.vel for p in points]
            point_cloud = np.array([x_pos, y_pos, z_pos, vel]).T
            yield point_cloud, msg.header.seq


def visualize(result):
    init_fig()
    mmwave_pc, seq = result
    x_pos, y_pos, dopplers = mmwave_pc[:, 0], mmwave_pc[:, 1], mmwave_pc[:, 2]
    static_idx = dopplers <= range_res
    dynamic_idx = dopplers > range_res
    ax.set_title(f"frame id: {seq}")
    ax.plot(x_pos[static_idx], y_pos[static_idx], color_panel[2], ms=2)
    ax.plot(x_pos[dynamic_idx], y_pos[dynamic_idx], color_panel[0], ms=2)


ani = animation.FuncAnimation(
    fig, visualize, gen_data, interval=100,
    init_func=init_fig, repeat=False, save_count=1000
)
ani.save(f"{file_path}-mmwave_pc.gif", writer='pillow')
# plt.show()
