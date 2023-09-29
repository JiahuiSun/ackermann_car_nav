import rosbag
from sensor_msgs import point_cloud2
import numpy as np
import pickle
import matplotlib.pyplot as plt
import struct
from nlos_sensing import intersection_of_2line, line_by_vertical_coef_p, parallel_line_distance, point2line_distance, fit_line_ransac


## 读取数据
file_path = "/home/dingrong/Desktop/exp2_mid_2023-09-19-22-42-47"
bag = rosbag.Bag(f"{file_path}.bag")
bag_data = bag.read_messages(topics=['/laser_point_cloud', '/laser_point_cloud2', '/mmwave_radar_point_cloud', '/mmwave_radar_raw_data'])
frame_bytes = 196608

# 我原来想获取同一个时刻下的所有topic，才发现原来各个传感器时间是不一样的，没有任何消息是同时出现的，所以一定有先后时间顺序
laser_list, laser_list2, mmwave_list, mmwave_raw_list = [], [], [], []
for topic, msg, t in bag_data:
    if topic == '/laser_point_cloud':
        points = point_cloud2.read_points_list(
            msg, field_names=['x', 'y']
        )
        x_pos = [p.x for p in points]
        y_pos = [p.y for p in points]
        point_cloud = np.array([x_pos, y_pos]).T
        laser_list.append((t.to_sec(), msg.header.stamp.to_sec(), point_cloud))
    if topic == '/laser_point_cloud2':
        points = point_cloud2.read_points_list(
            msg, field_names=['x', 'y']
        )
        x_pos = [p.x for p in points]
        y_pos = [p.y for p in points]
        point_cloud = np.array([x_pos, y_pos]).T
        laser_list2.append((t.to_sec(), msg.header.stamp.to_sec(), point_cloud))
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
print(len(mmwave_list), len(mmwave_raw_list), len(laser_list), len(laser_list2))

mmwave_t = np.array([t for t, stamp, pc in mmwave_list])
mmwave_stamp = np.array([stamp for t, stamp, pc in mmwave_list])
mmwave_raw_t = np.array([t for t, stamp, raw in mmwave_raw_list])
mmwave_raw_stamp = np.array([stamp for t, stamp, raw in mmwave_raw_list])
laser_t = np.array([t for t, stamp, pc in laser_list])
laser_stamp = np.array([stamp for t, stamp, pc in laser_list])
laser2_t = np.array([t for t, stamp, pc in laser_list2])
laser2_stamp = np.array([stamp for t, stamp, pc in laser_list2])
fig, ax = plt.subplots(figsize=(20, 4))
ax.plot(mmwave_t, np.ones(len(mmwave_t)), 'or', ms=0.5)
ax.plot(mmwave_stamp, np.ones(len(mmwave_stamp))*2, 'or', ms=0.5)
ax.plot(mmwave_raw_t, np.ones(len(mmwave_raw_t))*1.1, 'ob', ms=0.5)
ax.plot(mmwave_raw_stamp, np.ones(len(mmwave_raw_stamp))*1.9, 'ob', ms=0.5)
ax.plot(laser_t, np.ones(len(laser_t))*1.2, 'og', ms=0.5)
ax.plot(laser_stamp, np.ones(len(laser_stamp))*1.8, 'og', ms=0.5)
ax.plot(laser2_t, np.ones(len(laser2_t))*1.3, 'ok', ms=0.5)
ax.plot(laser2_stamp, np.ones(len(laser2_stamp))*1.7, 'ok', ms=0.5)
plt.show()

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

data_list = [mmwave_list, mmwave_raw_list, laser_list, laser_list2]
# 找到最早开始最晚结束的两个人进行align，这样一定是最短的
latest_begin = np.argmax([
    mmwave_stamp[0], mmwave_raw_stamp[0], laser_stamp[0], laser2_stamp[0]
])
earliest_end = np.argmin([
    mmwave_stamp[-1], mmwave_raw_stamp[-1], laser_stamp[-1], laser2_stamp[-1]
])
print(latest_begin, earliest_end)
# 如果是同一个人最晚开始最早结束
if latest_begin == earliest_end:
    pivot = latest_begin
    for i in range(len(data_list)):
        if i == pivot:
            continue
        data_list[i], data_list[pivot], _, _ = align(data_list[i], data_list[pivot], k=1, j=1)
else:
    data_list[latest_begin], data_list[earliest_end], _, _ = align(data_list[latest_begin], data_list[earliest_end], k=1, j=1)
    pivot = latest_begin
    for i in range(len(data_list)):
        if i in [latest_begin, earliest_end]:
            continue
        else:
            data_list[i], data_list[pivot], _, _ = align(data_list[i], data_list[pivot], k=1, j=1)
mmwave_list, mmwave_raw_list, laser_list, laser_list2 = data_list
print(len(mmwave_list), len(mmwave_raw_list), len(laser_list), len(laser_list2))

mmwave_t = np.array([t for t, stamp, pc in mmwave_list])
mmwave_stamp = np.array([stamp for t, stamp, pc in mmwave_list])
mmwave_raw_t = np.array([t for t, stamp, raw in mmwave_raw_list])
mmwave_raw_stamp = np.array([stamp for t, stamp, raw in mmwave_raw_list])
laser_t = np.array([t for t, stamp, pc in laser_list])
laser_stamp = np.array([stamp for t, stamp, pc in laser_list])
laser2_t = np.array([t for t, stamp, pc in laser_list2])
laser2_stamp = np.array([stamp for t, stamp, pc in laser_list2])
fig, ax = plt.subplots(figsize=(20, 4))
ax.plot(mmwave_t, np.ones(len(mmwave_t)), 'or', ms=0.5)
ax.plot(mmwave_stamp, np.ones(len(mmwave_stamp))*2, 'or', ms=0.5)
ax.plot(mmwave_raw_t, np.ones(len(mmwave_raw_t))*1.1, 'ob', ms=0.5)
ax.plot(mmwave_raw_stamp, np.ones(len(mmwave_raw_stamp))*1.9, 'ob', ms=0.5)
ax.plot(laser_t, np.ones(len(laser_t))*1.2, 'og', ms=0.5)
ax.plot(laser_stamp, np.ones(len(laser_stamp))*1.8, 'og', ms=0.5)
ax.plot(laser2_t, np.ones(len(laser2_t))*1.3, 'ok', ms=0.5)
ax.plot(laser2_stamp, np.ones(len(laser2_stamp))*1.7, 'ok', ms=0.5)
plt.show()


## 空间对齐
# 难点在于：原来用小车的4条腿来提取小车位姿，但腿太细导致点太少，只用1帧几乎无法提取小车位姿，小车不动还可以把多帧叠加，一旦运动起来就只能靠1帧的点云
# 因此我想通过面来检测，而不是腿，此时存在的问题是：在有些角度，面上的点也很少
local_sensing_range = [-3, -2, -3, -1]  # TODO: xxyy切割小车，只保留小车的点云；先用parse_laser_pc_bag.py看一下怎么切割

all_point_cloud = []
for i in range(len(laser_list2)):
    # 用laser2提取小车位姿
    laser_n_frame = laser_list2[i][2]
    flag_x = np.logical_and(laser_n_frame[:, 0]>=local_sensing_range[0], laser_n_frame[:, 0]<=local_sensing_range[1])
    flag_y = np.logical_and(laser_n_frame[:, 1]>=local_sensing_range[2], laser_n_frame[:, 1]<=local_sensing_range[3])
    flag = np.logical_and(flag_x, flag_y)
    laser_part = laser_n_frame[flag]

    # 提取小车的面的直线
    ransac_sigma = 0.02
    ransac_iter = 200
    coef, inlier_mask = fit_line_ransac(laser_part, max_iter=ransac_iter, sigma=ransac_sigma)
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
    print("laser points:", laser_part.shape, "theta:", theta*180/np.pi, "inter:", inter)

    # 保存结果：时间、laser、laser2、毫米波点云、毫米波原始数据、小车位姿
    transform = (inter, theta)
    tmp = (laser_list[i][1], laser_list[i][2], laser_list2[i][2], mmwave_list[i][2], mmwave_raw_list[i][2], transform)
    all_point_cloud.append(tmp)

## 保存结果
with open(f"{file_path}.pkl", 'wb') as f:
    pickle.dump(all_point_cloud, f)
