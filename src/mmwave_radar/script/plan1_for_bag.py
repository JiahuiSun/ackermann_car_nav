import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from nlos_sensing import transform
import rosbag
from sensor_msgs import point_cloud2


fig, ax = plt.subplots(figsize=(10, 4))
line0, = ax.plot([], [], 'ob', ms=2)
line1, = ax.plot([], [], 'or', ms=2)
lines = [line0, line1]


def init_fig():
    ax.set_xlabel('x(m)')
    ax.set_ylabel('y(m)')
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 10])
    ax.plot([-5, 5], [0, 0], 'k')
    return lines

def gen_data():
    for topic, msg, t in rosbag.Bag(
        "/home/dingrong/Code/ackermann_car_nav/test_2023-05-11-19-45-10.bag", 'r'):
        if topic == '/mmwave_radar_point_cloud':
            points = point_cloud2.read_points_list(
                msg, field_names=['x', 'y', 'z', 'vel']
            )
            x_pos = [p.x for p in points]
            y_pos = [p.y for p in points]
            z_pos = [p.z for p in points]
            vel = [p.vel for p in points]
            point_cloud = np.array([x_pos, y_pos, z_pos, vel]).T
            point_cloud[:, :2] = transform(point_cloud[:, :2], -0.043, 0.13, 330)
            yield point_cloud, msg.header.seq

def visualize(result):
    adc_data, seq = result
    x_pos, y_pos, dopplers = adc_data[:, 0], adc_data[:, 1], adc_data[:, -1]
    static_idx = dopplers == 0
    dynamic_idx = dopplers != 0
    ax.set_title(f"frame id: {seq}")
    lines[0].set_data(x_pos[static_idx], y_pos[static_idx])
    lines[1].set_data(x_pos[dynamic_idx], y_pos[dynamic_idx])


ani = animation.FuncAnimation(
    fig, visualize, gen_data, interval=100,
    init_func=init_fig, repeat=False, save_count=100
)
ani.save("plan1-2.gif", writer='imagemagick')
plt.show()
