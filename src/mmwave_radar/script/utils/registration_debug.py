import rosbag
from sensor_msgs import point_cloud2
import numpy as np
import pickle
import matplotlib.pyplot as plt
import struct
import sys
import matplotlib.animation as animation
import os

from nlos_sensing import registration, pc_filter, transform
from corner_type import L_open_corner_gt, L_open_corner_onboard

## 读取数据
if len(sys.argv) > 1:
    file_path = sys.argv[1]
    single = int(sys.argv[2])
else:
    file_path = "/home/agent/Code/ackermann_car_nav/data/20231115/bc_static2_2023-11-14-15-30-01"
    out_path = "/home/agent/Code/ackermann_car_nav/data/tmp"
    single = 1
file_name = file_path.split('/')[-1].split('.')[0][:-20]
os.makedirs(f"{out_path}/gifs", exist_ok=True)
bag = rosbag.Bag(f"{file_path}.bag")
bag_data = bag.read_messages(topics=['/laser_point_cloud', '/laser_point_cloud2', '/mmwave_radar_point_cloud', '/mmwave_radar_raw_data'])
frame_bytes = 196608
fwrite = open(f"{file_path}.log", 'w')

# 我原来想获取同一个时刻下的所有topic，才发现原来各个传感器时间是不一样的，没有任何消息是同时出现的，所以一定有先后时间顺序
robot_laser_list, gt_laser_list, mmwave_raw_list = [], [], []
for topic, msg, t in bag_data:
    if topic == '/laser_point_cloud':
        points = point_cloud2.read_points_list(
            msg, field_names=['x', 'y']
        )
        x_pos = [p.x for p in points]
        y_pos = [p.y for p in points]
        point_cloud = np.array([x_pos, y_pos]).T
        robot_laser_list.append((t.to_sec(), msg.header.stamp.to_sec(), point_cloud))
    if topic == '/laser_point_cloud2':
        points = point_cloud2.read_points_list(
            msg, field_names=['x', 'y']
        )
        x_pos = [p.x for p in points]
        y_pos = [p.y for p in points]
        point_cloud = np.array([x_pos, y_pos]).T
        gt_laser_list.append((t.to_sec(), msg.header.stamp.to_sec(), point_cloud))
    if topic == '/mmwave_radar_raw_data':
        adc_pack = struct.pack(f">{frame_bytes}b", *msg.data)
        adc_unpack = np.frombuffer(adc_pack, dtype=np.int16)
        mmwave_raw_list.append((t.to_sec(), msg.header.stamp.to_sec(), adc_unpack))
fwrite.write(f"mmwave_raw_len: {len(mmwave_raw_list)}, robot_laser_len: {len(robot_laser_list)}, gt_laser_len: {len(gt_laser_list)}\n")

mmwave_raw_t = np.array([t for t, stamp, raw in mmwave_raw_list])
mmwave_raw_stamp = np.array([stamp for t, stamp, raw in mmwave_raw_list])
laser_t = np.array([t for t, stamp, pc in robot_laser_list])
laser_stamp = np.array([stamp for t, stamp, pc in robot_laser_list])
laser2_t = np.array([t for t, stamp, pc in gt_laser_list])
laser2_stamp = np.array([stamp for t, stamp, pc in gt_laser_list])
fig, ax = plt.subplots(figsize=(20, 4))
ax.plot(mmwave_raw_t, np.ones(len(mmwave_raw_t))*1.1, 'ob', ms=0.5)
ax.plot(mmwave_raw_stamp, np.ones(len(mmwave_raw_stamp))*1.9, 'ob', ms=0.5)
ax.plot(laser_t, np.ones(len(laser_t))*1.2, 'og', ms=0.5)
ax.plot(laser_stamp, np.ones(len(laser_stamp))*1.8, 'og', ms=0.5)
ax.plot(laser2_t, np.ones(len(laser2_t))*1.3, 'ok', ms=0.5)
ax.plot(laser2_stamp, np.ones(len(laser2_stamp))*1.7, 'ok', ms=0.5)
fig.savefig(f"{file_path}-orignal-stamp.png", dpi=100)


## 时间对齐
def align(list1, list2, k=0, j=0):
    len_diff = min(len(list1), len(list2))
    minv, mini = np.inf, 0
    # 谁先曝光，谁就从头开始遍历，寻找距离对方开头时间最接近的
    if list1[0][k] < list2[0][j]:
        for i in range(len_diff):
            diff = np.abs(list1[i][k]-list2[0][j])
            if diff < minv:
                minv = diff
                mini = i
        list1 = list1[mini:]
    elif list1[0][k] > list2[0][j]:
        for i in range(len_diff):
            diff = np.abs(list2[i][k]-list1[0][j])
            if diff < minv:
                minv = diff
                mini = i
        list2 = list2[mini:]
    # 掐头去尾
    len_min = min(len(list1), len(list2))
    return list1[:len_min], list2[:len_min], minv, mini

data_list = [mmwave_raw_list, robot_laser_list, gt_laser_list]
while True:
    data_len = np.array([len(x) for x in data_list])
    stop = np.array([data_len[0] == x for x in data_len])
    if stop.all():
        break
    shortest_list = np.argmin(data_len)
    for i in range(len(data_list)):
        if i != shortest_list:
            data_list[i], data_list[shortest_list], _, _ = align(data_list[i], data_list[shortest_list], k=1, j=1)
mmwave_raw_list, robot_laser_list, gt_laser_list = data_list
fwrite.write(f"mmwave_raw_len: {len(mmwave_raw_list)}, robot_laser_len: {len(robot_laser_list)}, gt_laser_len: {len(gt_laser_list)}\n")

mmwave_raw_t = np.array([t for t, stamp, raw in mmwave_raw_list])
mmwave_raw_stamp = np.array([stamp for t, stamp, raw in mmwave_raw_list])
laser_t = np.array([t for t, stamp, pc in robot_laser_list])
laser_stamp = np.array([stamp for t, stamp, pc in robot_laser_list])
laser2_t = np.array([t for t, stamp, pc in gt_laser_list])
laser2_stamp = np.array([stamp for t, stamp, pc in gt_laser_list])
fig, ax = plt.subplots(figsize=(20, 4))
ax.plot(mmwave_raw_t, np.ones(len(mmwave_raw_t))*1.1, 'ob', ms=0.5)
ax.plot(mmwave_raw_stamp, np.ones(len(mmwave_raw_stamp))*1.9, 'ob', ms=0.5)
ax.plot(laser_t, np.ones(len(laser_t))*1.2, 'og', ms=0.5)
ax.plot(laser_stamp, np.ones(len(laser_stamp))*1.8, 'og', ms=0.5)
ax.plot(laser2_t, np.ones(len(laser2_t))*1.3, 'ok', ms=0.5)
ax.plot(laser2_stamp, np.ones(len(laser2_stamp))*1.7, 'ok', ms=0.5)
fig.savefig(f"{file_path}-temporal-align.png", dpi=100)


## 空间对齐
gt_wall_pc_range = [-4, 2, -3, 3]  # 切割gt周围墙的点云
onboard_wall_pc_range = [-5, 1, -3, 3]  # 切割小车周围墙的点云

fig, ax = plt.subplots(figsize=(8, 8))
color_panel = ['ro', 'go', 'bo', 'co', 'yo', 'wo', 'mo', 'ko']
def init_fig():
    ax.clear()
    ax.set_xlabel('x(m)')
    ax.set_ylabel('y(m)')
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.tick_params(direction='in')

from sklearn.cluster import DBSCAN, KMeans
db = KMeans(n_clusters=2, n_init='auto')
# db = DBSCAN(eps=0.5, min_samples=10)

def gen_data():
    for i in range(len(gt_laser_list)):
        # 从GT激光雷达点云中提取墙面
        gt_laser_pc = pc_filter(gt_laser_list[i][2], *gt_wall_pc_range)
        gt_walls, gt_points = L_open_corner_gt(gt_laser_pc)
        # 从onboard激光雷达点云中提取墙面
        onboard_laser_pc = pc_filter(robot_laser_list[i][2], *onboard_wall_pc_range)
        onboard_walls, onboard_points = L_open_corner_onboard(onboard_laser_pc)

        # 按照x值大小排序，取最小的top20
        indices = np.argsort(-gt_walls['barrier_wall_pc'][:, 1])
        gt_barrier_wall_set = gt_walls['barrier_wall_pc'][indices][:20]
        ax.plot(gt_barrier_wall_set[:, 0], gt_barrier_wall_set[:, 1], color_panel[0], ms=2)

        gt_labels = db.fit(gt_walls['far_wall_pc']).labels_
        part1 = gt_walls['far_wall_pc'][gt_labels==0]
        part2 = gt_walls['far_wall_pc'][gt_labels==1]
        gt_part1, gt_part2 = (part1, part2) if np.mean(part1[:, 0]) < np.mean(part2[:, 0]) else (part2, part1)
        ax.plot(gt_part1[:, 0], gt_part2[:, 1], color_panel[1], ms=2)
        ax.plot(gt_part2[:, 0], gt_part2[:, 1], color_panel[2], ms=2)

        # 按照x值大小排序，取最小的top20
        indices = np.argsort(onboard_walls['barrier_wall_pc'][:, 0])
        onboard_barrier_wall_set = onboard_walls['barrier_wall_pc'][indices][:20]
        ax.plot(onboard_barrier_wall_set[:, 0], onboard_barrier_wall_set[:, 1], color_panel[3], ms=2)

        onboard_labels = db.fit(onboard_walls['far_wall_pc']).labels_
        part1 = onboard_walls['far_wall_pc'][onboard_labels==0]
        part2 = onboard_walls['far_wall_pc'][onboard_labels==1]
        onboard_part1, onboard_part2 = (part1, part2) if np.mean(part1[:, 1]) < np.mean(part2[:, 1]) else (part2, part1)
        ax.plot(onboard_part1[:, 0], onboard_part1[:, 1], color_panel[4], ms=2)
        ax.plot(onboard_part2[:, 0], onboard_part2[:, 1], color_panel[5], ms=2)

        # 把两个part2对齐
        min_len = min(len(gt_part2), len(onboard_part2))
        indices = np.argsort(gt_part2[:, 0])
        gt_part2 = gt_part2[indices][:min_len]
        indices = np.argsort(onboard_part2[:, 1])
        onboard_part2 = onboard_part2[indices][:min_len]
        # 从GT激光雷达到小车激光雷达
        R, T = registration(gt_part2.T, onboard_part2.T)

        src2tar_far_wall_pc = (R.dot(gt_walls['far_wall_pc'].T) + T).T
        src2tar_barrier_wall_pc = (R.dot(gt_walls['barrier_wall_pc'].T) + T).T
        src2tar_gt_pc = (R.dot(gt_laser_list[i][2].T) + T).T
        # 把两个点云都转化到小车坐标系下可视化
        src2tar_far_wall_pc = transform(src2tar_far_wall_pc, 0.08, 0, 180)
        src2tar_barrier_wall_pc = transform(src2tar_barrier_wall_pc, 0.08, 0, 180)
        src2tar_gt_pc = transform(src2tar_gt_pc, 0.08, 0, 180)
        onboard_walls['far_wall_pc'] = transform(onboard_walls['far_wall_pc'], 0.08, 0, 180)
        onboard_walls['barrier_wall_pc'] = transform(onboard_walls['barrier_wall_pc'], 0.08, 0, 180)
        onboard_laser_pc = transform(robot_laser_list[i][2], 0.08, 0, 180)

        yield src2tar_far_wall_pc, src2tar_barrier_wall_pc, src2tar_gt_pc, onboard_walls['far_wall_pc'], onboard_walls['barrier_wall_pc'], onboard_laser_pc

def visualize(result):
    init_fig()
    src2tar_far, src2tar_barrier, src2tar_gt_pc, far, barrier, onboard_pc = result
    # ax.plot(src2tar_gt_pc[:, 0], src2tar_gt_pc[:, 1], color_panel[4], ms=2)
    # ax.plot(onboard_pc[:, 0], onboard_pc[:, 1], color_panel[-1], ms=2)
    ax.plot(src2tar_far[:, 0], src2tar_far[:, 1], color_panel[0], ms=2)
    ax.plot(src2tar_barrier[:, 0], src2tar_barrier[:, 1], color_panel[1], ms=2)
    ax.plot(far[:, 0], far[:, 1], color_panel[2], ms=2)
    ax.plot(barrier[:, 0], barrier[:, 1], color_panel[3], ms=2)

ani = animation.FuncAnimation(
    fig, visualize, gen_data, interval=100,
    init_func=init_fig, repeat=False, save_count=1000
)
ani.save(f"{out_path}/gifs/{file_name}-regis4.gif", writer='pillow')

fwrite.close()
