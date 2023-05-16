import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import rosbag
from sensor_msgs import point_cloud2


fig, ax = plt.subplots(figsize=(8, 8))
line, = ax.plot([], [], 'ro', ms=2)


def init_fig():
    ax.set_xlabel('x(m)')
    ax.set_ylabel('y(m)')
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    return line

def gen_data():
    for topic, msg, t in rosbag.Bag(
        "/home/dingrong/Code/ackermann_car_nav/data/test_laser_2023-05-16-14-46-03.bag", 'r'):
        if topic == '/laser_point_cloud':
            points = point_cloud2.read_points_list(
                msg, field_names=['x', 'y']
            )
            x_pos = [p.x for p in points]
            y_pos = [p.y for p in points]
            point_cloud = np.array([x_pos, y_pos]).T
            yield point_cloud, msg.header.seq

def visualize(result):
    adc_data, seq = result
    x_pos, y_pos = adc_data[:, 0], adc_data[:, 1]
    ax.set_title(f"frame id: {seq}")
    line.set_data(x_pos, y_pos)

ani = animation.FuncAnimation(
    fig, visualize, gen_data, interval=100,
    init_func=init_fig, repeat=False
)
# ani.save("plan1-2.gif", writer='imagemagick')
plt.show()
