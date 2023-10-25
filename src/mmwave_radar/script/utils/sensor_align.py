import rosbag
from sensor_msgs import point_cloud2
import numpy as np
import pickle
import matplotlib.pyplot as plt
import struct
import sys

from nlos_sensing import intersection_of_2line, line_by_vertical_coef_p, parallel_line_distance, point2line_distance, fit_line_ransac, pc_filter


## 读取数据
if len(sys.argv) > 1:
    file_path = sys.argv[1]
    single = int(sys.argv[2])
else:
    file_path = "/home/agent/Code/ackermann_car_nav/data/20231002/soft-3-A_2023-10-01-20-21-03"
    single = 1
bag = rosbag.Bag(f"{file_path}.bag")
bag_data = bag.read_messages(topics=['/laser_point_cloud', '/laser_point_cloud2', '/mmwave_radar_point_cloud', '/mmwave_radar_raw_data'])
frame_bytes = 196608
fwrite = open(f"{file_path}.log", 'w')

# 我原来想获取同一个时刻下的所有topic，才发现原来各个传感器时间是不一样的，没有任何消息是同时出现的，所以一定有先后时间顺序
robot_laser_list, gt_laser_list, mmwave_list, mmwave_raw_list = [], [], [], []
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
    if topic == '/mmwave_radar_point_cloud':
        points = point_cloud2.read_points_list(
            msg, field_names=['x', 'y', 'z', 'vel', 'snr']
        )
        x_pos = [p.x for p in points]
        y_pos = [p.y for p in points]
        z_pos = [p.z for p in points]
        vel = [p.vel for p in points]
        snr = [p.snr for p in points]
        point_cloud = np.array([x_pos, y_pos, z_pos, vel, snr]).T
        mmwave_list.append((t.to_sec(), msg.header.stamp.to_sec(), point_cloud))
    if topic == '/mmwave_radar_raw_data':
        adc_pack = struct.pack(f">{frame_bytes}b", *msg.data)
        adc_unpack = np.frombuffer(adc_pack, dtype=np.int16)
        mmwave_raw_list.append((t.to_sec(), msg.header.stamp.to_sec(), adc_unpack))
fwrite.write(f"mmwave_len: {len(mmwave_list)}, mmwave_raw_len: {len(mmwave_raw_list)}, robot_laser_len: {len(robot_laser_list)}, gt_laser_len: {len(gt_laser_list)}\n")

mmwave_t = np.array([t for t, stamp, pc in mmwave_list])
mmwave_stamp = np.array([stamp for t, stamp, pc in mmwave_list])
mmwave_raw_t = np.array([t for t, stamp, raw in mmwave_raw_list])
mmwave_raw_stamp = np.array([stamp for t, stamp, raw in mmwave_raw_list])
laser_t = np.array([t for t, stamp, pc in robot_laser_list])
laser_stamp = np.array([stamp for t, stamp, pc in robot_laser_list])
laser2_t = np.array([t for t, stamp, pc in gt_laser_list])
laser2_stamp = np.array([stamp for t, stamp, pc in gt_laser_list])
fig, ax = plt.subplots(figsize=(20, 4))
ax.plot(mmwave_t, np.ones(len(mmwave_t)), 'or', ms=0.5)
ax.plot(mmwave_stamp, np.ones(len(mmwave_stamp))*2, 'or', ms=0.5)
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

# 第一次的做法存在问题：3是最早开始的，0是最晚结束的，二者都长为1003，对齐后也为1003
# 用3对齐1和2；1也为1003，对齐后仍为1003；2为1001，对齐后3和2变成1000；所以最终长度为1003、1003、1000、1000
# 正确的逻辑：如果所有人长度一致，停止；先用最短的list对齐所有人；否则再找到最短的list，对齐所有人；
# 这也不对吧？都一样长不代表就对齐了，比如时间恰好错开几个stamp；不是有图吗？通过肉眼判断吧
data_list = [mmwave_list, mmwave_raw_list, robot_laser_list, gt_laser_list]
while True:
    data_len = np.array([len(x) for x in data_list])
    stop = np.array([data_len[0] == x for x in data_len])
    if stop.all():
        break
    shortest_list = np.argmin(data_len)
    for i in range(len(data_list)):
        if i != shortest_list:
            data_list[i], data_list[shortest_list], _, _ = align(data_list[i], data_list[shortest_list], k=1, j=1)
mmwave_list, mmwave_raw_list, robot_laser_list, gt_laser_list = data_list
fwrite.write(f"mmwave_len: {len(mmwave_list)}, mmwave_raw_len: {len(mmwave_raw_list)}, robot_laser_len: {len(robot_laser_list)}, gt_laser_len: {len(gt_laser_list)}\n")

mmwave_t = np.array([t for t, stamp, pc in mmwave_list])
mmwave_stamp = np.array([stamp for t, stamp, pc in mmwave_list])
mmwave_raw_t = np.array([t for t, stamp, raw in mmwave_raw_list])
mmwave_raw_stamp = np.array([stamp for t, stamp, raw in mmwave_raw_list])
laser_t = np.array([t for t, stamp, pc in robot_laser_list])
laser_stamp = np.array([stamp for t, stamp, pc in robot_laser_list])
laser2_t = np.array([t for t, stamp, pc in gt_laser_list])
laser2_stamp = np.array([stamp for t, stamp, pc in gt_laser_list])
fig, ax = plt.subplots(figsize=(20, 4))
ax.plot(mmwave_t, np.ones(len(mmwave_t)), 'or', ms=0.5)
ax.plot(mmwave_stamp, np.ones(len(mmwave_stamp))*2, 'or', ms=0.5)
ax.plot(mmwave_raw_t, np.ones(len(mmwave_raw_t))*1.1, 'ob', ms=0.5)
ax.plot(mmwave_raw_stamp, np.ones(len(mmwave_raw_stamp))*1.9, 'ob', ms=0.5)
ax.plot(laser_t, np.ones(len(laser_t))*1.2, 'og', ms=0.5)
ax.plot(laser_stamp, np.ones(len(laser_stamp))*1.8, 'og', ms=0.5)
ax.plot(laser2_t, np.ones(len(laser2_t))*1.3, 'ok', ms=0.5)
ax.plot(laser2_stamp, np.ones(len(laser2_stamp))*1.7, 'ok', ms=0.5)
fig.savefig(f"{file_path}-temporal-align.png", dpi=100)


## 空间对齐和人的点云提取
# 提取小车坐标系相对GT激光雷达坐标系的坐标变换，因为小车可能是运动的，所以每一帧提取一次相对位姿
robot_range = [-2.1, 0, -3, 0]  # 切割小车
gt_range = [-8, 0, -0.3, 1.5]  # 切割人的点云
gt_range1 = [-8, 0, -0.3, 0.7]  # 切割人的点云
gt_range2 = [-8, 0, 0.8, 1.5]  # 切割人的点云

all_point_cloud = []
for i in range(len(gt_laser_list)):
    # 用gt_laser提取小车位姿
    laser_frame = gt_laser_list[i][2]
    laser_part = pc_filter(laser_frame, *robot_range)

    # 提取小车的面的直线
    coef, inlier_mask = fit_line_ransac(laser_part, max_iter=200, sigma=0.02)
    laser_part = laser_part[inlier_mask]

    # 根据两条腿计算旋转角度
    theta = np.arctan(coef[0]) + np.pi if coef[0] < 0 else np.arctan(coef[0])
    # 计算小车中心点在标定激光雷达坐标系下的位置
    # 小车的右上角是A，右下角是B，AB中点是D，小车中心是C
    CD = 0.105
    OD = np.mean(laser_part, axis=1)
    coef1 = line_by_vertical_coef_p(coef, OD)
    coef2a, coef2b = parallel_line_distance(coef, CD)
    coef2 = coef2a if point2line_distance(coef2a, [0, 0]) > point2line_distance(coef2b, [0, 0]) else coef2b
    inter = intersection_of_2line(coef1, coef2)
    fwrite.write(f"laser points: {laser_part.shape} theta: {theta*180/np.pi}, inter: {inter}\n")

    transform = (inter, theta)
    # 保存结果：时间、小车激光雷达点云、人的点云、毫米波点云、毫米波原始数据、小车位姿
    # 标定激光雷达点云只保留人
    if single:
        laser_pc_person = pc_filter(laser_frame, *gt_range)
        tmp = (robot_laser_list[i][1], robot_laser_list[i][2], laser_pc_person, mmwave_list[i][2], mmwave_raw_list[i][2], transform)
    else:
        laser_pc_person1 = pc_filter(laser_frame, *gt_range1)
        laser_pc_person2 = pc_filter(laser_frame, *gt_range2)
        tmp = (robot_laser_list[i][1], robot_laser_list[i][2], laser_pc_person1, laser_pc_person2, mmwave_list[i][2], mmwave_raw_list[i][2], transform)
    all_point_cloud.append(tmp)
fwrite.close()

## 保存结果
with open(f"{file_path}.pkl", 'wb') as f:
    pickle.dump(all_point_cloud, f)
