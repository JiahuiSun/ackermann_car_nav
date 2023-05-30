import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import pickle
import rosbag
from sensor_msgs import point_cloud2

from ransac import fit_line_ransac
from nlos_sensing import transform, nlosFilterAndMapping, line_symmetry_point
from nlos_sensing import get_span, find_end_point


"""
需求分析：
- 把小车上的激光雷达点云、毫米波雷达点云同步到标定雷达下
- 这需要获得小车相对标定雷达的位姿
- 如果小车是静止的，可以手动从一帧点云中获取，可如果小车是运动的，怎么从每一帧点云中提取小车啊？
- 人也是运动的啊，你怎么提取人的位置？难道就真的要一帧一帧，手动提取人的位置？

- 打开标定激光雷达的一帧
    - 画图看看人和车的点云
    - 从点云中提取人的点云，人的位置是点云均值
    - 从点云中提取车的特征点，根据特征点换算出坐标系变换关系——这个是最关键的，如果这个可以做到，实际上我就可以
- 打开小车激光雷达的一帧
    - 把墙面挑出来，拟合直线
- 打开毫米波雷达的一帧
    - NLoS过滤和映射
    - 把人的位置换算到标定雷达坐标系下
- 计算定位误差
"""

local_sensing_range = [-5, 3, -5, 5]
min_points_inline = 20
min_length_inline = 0.6
ransac_sigma = 0.02
ransac_iter = 200
filter = DBSCAN(eps=1, min_samples=20)

fig, ax = plt.subplots(figsize=(16, 16))
color_panel = ['ro', 'go', 'bo', 'co', 'wo', 'yo', 'mo', 'ko']


def init_fig():
    ax.clear()
    ax.set_xlabel('x(m)')
    ax.set_ylabel('y(m)')
    ax.set_xlim([local_sensing_range[0], local_sensing_range[1]])
    ax.set_ylim([local_sensing_range[2], local_sensing_range[3]])


def gen_data():
    for topic, msg, t in rosbag.Bag(
        "/home/dingrong/Code/ackermann_car_nav/data/20230530/walk_param1_2023-05-30-16-58-37.bag", 'r'):
        if topic == '/laser_point_cloud2':
            points = point_cloud2.read_points_list(
                msg, field_names=['x', 'y']
            )
            x_pos = [p.x for p in points]
            y_pos = [p.y for p in points]
            point_cloud = np.array([x_pos, y_pos]).T
            flag_x = np.logical_and(point_cloud[:, 0]>=local_sensing_range[0], point_cloud[:, 0]<=local_sensing_range[1])
            flag_y = np.logical_and(point_cloud[:, 1]>=local_sensing_range[2], point_cloud[:, 1]<=local_sensing_range[3])
            flag = np.logical_and(flag_x, flag_y)
            point_cloud = point_cloud[flag]
            # point_cloud = transform(point_cloud, 0.08, 0, 90)
            yield t.to_sec(), point_cloud


def visualize(result):
    init_fig()
    t, laser_point_cloud = result
    ax.plot(laser_point_cloud[:, 0], laser_point_cloud[:, 1], color_panel[0], ms=2)
    # coef, inlier_mask = fit_line_ransac(laser_point_cloud, max_iter=ransac_iter, sigma=ransac_sigma)
    # print(np.sum(np.logical_not(inlier_mask)))
    ax.set_title(f"Timestamp: {t:.2f}s")


ani = animation.FuncAnimation(
    fig, visualize, gen_data, interval=100,
    init_func=init_fig, repeat=False, save_count=200
)
writergif = animation.PillowWriter(fps=10)
# ani.save("test2.gif", writer=writergif)
plt.show()
