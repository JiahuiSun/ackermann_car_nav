#!/usr/bin/env python3
import message_filters
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from geometry_msgs.msg import TwistStamped
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
import sys
sys.path.append('/home/agent/Code/ackermann_car_nav/src/mmwave_radar/script/model')
from model import Darknet
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
    onboard_laser_pc_2car = pc_filter(onboard_laser_pc_2car, *onboard_wall_pc_range)
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
    gt_laser_pc_wall = pc_filter(gt_laser_pc, *gt_wall_pc_range)
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
    action = cmd_vel2action(cmd_vel.twist)

    # 提取bbox GT
    key_points, box_hw = bounding_box2(person_pc_2radar, delta_x=0.1, delta_y=0.1)
    # 只有当人位于NLOS右边才保存label和预测结果
    # FIXME: 这里有问题啊，NLOS区域是在雷达坐标系下构建的，随着移动的不同会变化，但不在NLOS区域内并不是小车不该检测到人的理由
    # 这就说明了，你模型还是没有想清楚
    inter1, inter3, gt_center = onboard_points['inter1'], onboard_points['inter3'], key_points[0]
    if np.cross(inter3-inter1, gt_center-inter1) <= 0:
        return
    # 如果label在NLOS内，映射过去
    if isin_triangle(onboard_points['barrier_corner'], inter1, inter3, gt_center):
        person_pc_2radar = line_symmetry_point(onboard_walls['far_wall'], person_pc_2radar)
        key_points, box_hw = bounding_box2(person_pc_2radar, delta_x=0.1, delta_y=0.1)
    key_points[0, 0] = np.clip(key_points[0, 0], -(H-1)*range_res, (H-1)*range_res)
    key_points[0, 1] = np.clip(key_points[0, 1], 0, (H-1)*range_res)
    label = np.array([
        [key_points[0, 0]+H*range_res, key_points[0, 1], box_hw[1], box_hw[0]]
    ])

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

    st5 = time.time()
    # 把整条轨迹的s、a、pred、label保存下来
    global cnt
    with open(os.path.join(save_dir, f"sample{cnt}.pkl"), 'wb') as f:
        pickle.dump([state, action, detections, label], f)
    cnt += 1
    end = time.time()
    print(f"total:{end-st1:.2f} RA:{st2-st1:.2f} onboard_lidar_proc:{st3-st2:.2f} gt_lidar_proc:{st4-st3:.2f} object detect:{st5-st4:.2f} save:{end-st5:.2f}")


def cmd_vel2action(cmd_vel):
    """
    映射规则：
    (0, 0): 0
    (0.25, 0): 1
    (0.25, 0.3): 2
    (0.25, -0.3): 3
    Args:
        cmd_vel: Twist
    Returns:
        action: id 0-3 
    """
    x, z = cmd_vel.linear.x, cmd_vel.angular.z
    vel2act = {
        (0.0, 0.0): 0,
        (0.25, 0.0): 1,
        (0.25, 0.3): 2,
        (0.25, -0.3): 3
    }
    return vel2act[(x, z)]


if __name__ == '__main__':
    rospy.init_node("reward_model")
    gt_lidar_sub = message_filters.Subscriber('laser_point_cloud2', PointCloud2)
    onboard_lidar_sub = message_filters.Subscriber('laser_point_cloud', PointCloud2)
    radar_sub = message_filters.Subscriber('mmwave_radar_raw_data', adcData)
    vel_sub = message_filters.Subscriber('cmd_vel2', TwistStamped)

    # 目标检测相关
    model_path = "/home/agent/Code/yolov3_my/output/20231031_144012/model/model-99.pth"
    device = "cuda:0"
    anchors = torch.tensor([[10, 13], [16, 30], [33, 23]])
    img_size = [160, 320]
    conf_thres, nms_thres = 0.5, 0.4
    model = Darknet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    onboard_wall_pc_range = [-1, 5, -3, 3]  # 切割小车周围墙面点云
    gt_wall_pc_range = [-4, 2, -3, 3]  # 切割gt周围墙面点云
    person_range = [-8, 0, -0.3, 1]  # 根据实际情况设定

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
    file_path = "/home/agent/Code/ackermann_car_nav/data/20231117/traj_loc1_2023-11-16-11-50-40.bag"
    file_name = file_path.split('/')[-1].split('.')[0]
    save_dir = os.path.join(out_path, mode, file_name)
    os.makedirs(save_dir, exist_ok=True)
    cnt = 0
    ts = message_filters.ApproximateTimeSynchronizer([gt_lidar_sub, onboard_lidar_sub, radar_sub, vel_sub], 10, 0.05)
    ts.registerCallback(perception)
    rospy.spin()
