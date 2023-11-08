import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import os
import pickle


file_path = "/home/agent/Code/ackermann_car_nav/data/sync_test"
out_path = "/home/agent/Code/ackermann_car_nav/data/tmp"
file_name = 'sync'
os.makedirs(f"{out_path}/gifs", exist_ok=True)

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
    for cnt in range(1887):
        with open(os.path.join(file_path, f"frame-{cnt}.pkl"), 'rb') as f:
            tmp = pickle.load(f)
        (radar_stamp, radar_pc), (onboard_stamp, onboard_pc), (gt_stamp, gt_pc) = tmp
        yield cnt, radar_stamp, radar_pc, onboard_stamp, onboard_pc, gt_stamp, gt_pc


def visualize(result):
    init_fig()
    cnt, radar_stamp, radar_pc, onboard_stamp, onboard_pc, gt_stamp, gt_pc = result
    ax.set_title(f"frame id: {cnt}")
    ax.plot(radar_pc[:, 0], radar_pc[:, 1], color_panel[2], ms=2)
    ax.plot(onboard_pc[:, 0], onboard_pc[:, 1], color_panel[0], ms=2)
    ax.plot(gt_pc[:, 0], gt_pc[:, 1], color_panel[1], ms=2)
    print(radar_stamp-onboard_stamp, radar_stamp-gt_stamp, onboard_stamp-gt_stamp)


ani = animation.FuncAnimation(
    fig, visualize, gen_data, interval=100,
    init_func=init_fig, repeat=False, save_count=1000
)
ani.save(f"{out_path}/gifs/{file_name}.gif", writer='pillow')
# plt.show()
