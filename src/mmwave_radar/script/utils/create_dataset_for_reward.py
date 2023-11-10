#!/usr/bin/env python3
import message_filters
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from geometry_msgs.msg import Twist
import pickle
from mmwave_radar.msg import adcData
import rospy
import torch
import struct
import time
import os

from radar_fft_music_RA import *
from corner_type import L_open_corner, L_open_corner_gt
from nlos_sensing import pc_filter, isin_triangle, line_symmetry_point, transform, transform_inverse, registration, bounding_box2
from postprocess import postprocess, nms_single_class
from ..model.model import Darknet
from bev import BEV


def perception(gt_laser_pc_msg, onboard_laser_pc_msg, radar_adc_data, cmd_vel):
    st1 = time.time()
    # 把毫米波雷达原始数据变成热力图和点云
    adc_pack = struct.pack(f">{frame_bytes}b", *radar_adc_data.data)
    adc_unpack = np.frombuffer(adc_pack, dtype=np.int16)
    result = gen_point_cloud_plan3(adc_unpack)
    if result is None:
        return
    RA_cart, mmwave_point_cloud = result

    st2 = time.time()
    # 小车激光雷达提取墙面和关键点
    points = point_cloud2.read_points_list(
        onboard_laser_pc_msg, field_names=['x', 'y']
    )
    x_pos = [p.x for p in points]
    y_pos = [p.y for p in points]
    onboard_laser_pc = np.array([x_pos, y_pos]).T
    onboard_laser_pc_2car = transform(onboard_laser_pc, 0.08, 0, 180)
    onboard_laser_pc_2car = pc_filter(onboard_laser_pc_2car, *local_sensing_range)
    onboard_laser_pc_2radar = transform_inverse(onboard_laser_pc_2car, 0.17, 0, 360-90)
    onboard_walls, onboard_points = L_open_corner(onboard_laser_pc_2radar)

    st3 = time.time()
    # GT激光雷达提取墙面和关键点
    points = point_cloud2.read_points_list(
        gt_laser_pc_msg, field_names=['x', 'y']
    )
    x_pos = [p.x for p in points]
    y_pos = [p.y for p in points]
    gt_laser_pc = np.array([x_pos, y_pos]).T
    gt_laser_pc_wall = pc_filter(gt_laser_pc, *gt_sensing_range)
    gt_walls, gt_points = L_open_corner_gt(gt_laser_pc_wall)
    src = gt_points['reference_points'].T
    tar = onboard_points['reference_points'].T
    R, T = registration(src, tar)

    # 提取state
    person_pc = pc_filter(gt_laser_pc, *person_range)
    person_pc_2radar = (R.dot(person_pc.T) + T).T
    person_pc_2car = transform(person_pc_2radar, 0.17, 0, 360-90)
    state = bev_map.mapping(onboard_laser_pc_2car, person_pc_2car)

    # 提取action
    linear_vel = cmd_vel.linear.x
    angular_vel = cmd_vel.angular.z
    R = linear_vel / angular_vel
    angle = np.arctan(wheel_base / R)
    speed = linear_vel / linear_velocity_ratio
    action = np.array([speed, angle])

    # 提取bbox GT
    key_points, box_hw = bounding_box2(person_pc_2radar, delta_x=0.1, delta_y=0.1)
    key_points[0, 0] = np.clip(key_points[0, 0], -(H-1)*range_res, (H-1)*range_res)
    key_points[0, 1] = np.clip(key_points[0, 1], 0, (H-1)*range_res)
    x_norm = (key_points[0, 0]+H*range_res) / (H*2*range_res)
    y_norm = key_points[0, 1] / (H*range_res)
    width_norm = box_hw[1] / (H*2*range_res)
    height_norm = box_hw[0] / (H*range_res)
    label = [x_norm, y_norm, width_norm, height_norm]

    st4 = time.time()
    # 目标检测
    RA_cart = (RA_cart - RA_cart.min()) / (RA_cart.max() - RA_cart.min())
    img = RA_cart.transpose(2, 0, 1)
    img = torch.from_numpy(img).float().to(device)
    img = img[None]
    with torch.no_grad():
        pred = model(img)
    pred_bbox = postprocess(pred, anchors, img_size)
    detections = nms_single_class(pred_bbox.cpu().numpy(), conf_thres, nms_thres)[0]
    # TODO: 获得可以计算ap的结果，本质就是原来的代码也做不到啊
    # 再用NLOS过滤一下结果
    final_det = []
    for det in detections:
        xyxy, conf = det[:4], det[4]
        pred_center = [(xyxy[0]+xyxy[2])/2, (xyxy[1]+xyxy[3])/2]
        pred_center = np.array([
            (pred_center[0]-H)*range_res, pred_center[1]*range_res
        ])
        # 一个目标是画图，把bbox在RVIZ上可视化出来；另一个目的是保存检测结果xywh，用来计算map
        if isin_triangle(onboard_points['symmetric_barrier_corner'], onboard_points['inter2'], \
                         onboard_points['inter1'], pred_center):
            final_det.append(xyxy)

            xyxy_real = np.copy(xyxy)
            xyxy_real[0] = (xyxy[0]-H)*range_res
            xyxy_real[1] = xyxy[1] * range_res
            xyxy_real[2] = (xyxy[2]-H)*range_res
            xyxy_real[3] = xyxy[3] * range_res
            xyxy_real[:2] = line_symmetry_point(onboard_walls['far_wall'], xyxy_real[:2])
            xyxy_real[2:] = line_symmetry_point(onboard_walls['far_wall'], xyxy_real[2:])        
    st5 = time.time()

    # 把整条轨迹的s、a、pred、label保存下来
    global cnt
    with open(os.path.join(save_dir, f"sample{cnt}.pkl"), 'wb') as f:
        pickle.dump([state, action, detections, label], f)
    cnt += 1
    end = time.time()
    print(f"total:{end-st1:.2f} RA:{st2-st1:.2f} onboard_lidar_proc:{st3-st2:.2f} gt_lidar_proc:{st4-st3:.2f} object detect:{st5-st4:.2f} save:{end-st5:.2f}")


if __name__ == '__main__':
    rospy.init_node("reward_model")
    gt_lidar_sub = message_filters.Subscriber('laser_point_cloud2', PointCloud2)
    onboard_lidar_sub = message_filters.Subscriber('laser_point_cloud', PointCloud2)
    radar_sub = message_filters.Subscriber('mmwave_radar_raw_data', adcData)
    vel_sub = message_filters.Subscriber('cmd_vel', Twist)

    model_path = "/home/dingrong/Downloads/model-99.pth"
    device = "cpu"
    anchors = torch.tensor([[10, 13], [16, 30], [33, 23]])
    img_size = [160, 320]
    conf_thres, nms_thres = 0.5, 0.4
    model = Darknet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    local_sensing_range = [-0.5, 5, -3, 3]  # 切割小车周围点云
    gt_sensing_range = [-4, 2, -4, 3]  # 切割gt周围点云
    person_range = [-8, 0, -0.3, 1.5]

    wheel_base = 0.324
    linear_velocity_ratio = 0.6944
    bev_map = BEV(height=6, width=8, n=256)

    # policy_my
        # traj_name
            # (s1, a1, pred1, label1)
            # ...
        # traj_name
            # (s1, a1, pred1, label1)
            # ...
    out_path = "/home/agent/Code/ackermann_car_nav/data/trajectories"
    mode = "policy_my"
    file_path = ""
    file_name = file_path.split('/')[-1].split('.')[0][:-20]
    save_dir = os.path.join(out_path, mode, file_name)
    os.makedirs(save_dir, exist_ok=True)
    cnt = 0

    ts = message_filters.ApproximateTimeSynchronizer([gt_lidar_sub, onboard_lidar_sub, radar_sub, vel_sub], 10, 0.05)
    ts.registerCallback(perception)
    rospy.spin()
