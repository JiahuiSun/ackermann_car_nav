import message_filters
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointField
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from mmwave_radar.msg import adcData
import rospy
import torch
import struct

from utils.radar_fft_music_RA import *
from utils.corner_type import L_open_corner, L_open_corner_gt
from utils.nlos_sensing import pc_filter, isin_triangle, line_symmetry_point, transform, transform_inverse, registration
from utils.postprocess import postprocess, nms_single_class
from model.model import Darknet


def perception(gt_laser_pc_msg, onboard_laser_pc_msg, radar_adc_data):
    """
    1. 把毫米波雷达原始数据变成热力图和点云——毫米波雷达坐标系
    2. 小车激光雷达提取墙面和关键点——毫米波雷达坐标系
    3. 用关键点构建NLOS理论区域，获得mask——毫米波雷达坐标系，所以上一步必须是毫米波雷达坐标系
    4. GT激光雷达提取墙面和关键点——GT激光雷达坐标系
    5. 上两步的关键点进行点云配准和坐标变换——从GT到毫米波雷达坐标系
    6. 把热力图、mask输入模型，输出bbox，通过NMS过滤一波——毫米波雷达坐标系
    7. 过滤和映射NLOS的bbox——毫米波雷达坐标系
    8. 可视化所有结果，需要的话，把结果转换到小车坐标系下
    """
    # 把毫米波雷达原始数据变成热力图和点云
    adc_pack = struct.pack(f">{frame_bytes}b", *radar_adc_data.data)
    adc_unpack = np.frombuffer(adc_pack, dtype=np.int16)
    result = gen_point_cloud_plan3(adc_unpack)
    if result is None:
        return
    RA_cart, mmwave_point_cloud = result
    msg = PointCloud2()
    msg.header = radar_adc_data.header
    msg.height = 1
    msg.width = mmwave_point_cloud.shape[0]
    msg.fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('vel', 8, PointField.FLOAT32, 1),
        PointField('snr', 12, PointField.FLOAT32, 1)
    ]
    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step = msg.point_step * mmwave_point_cloud.shape[0]
    msg.is_dense = True
    msg.data = mmwave_point_cloud.astype(np.float32).tobytes()
    radar_pc_pub.publish(msg)

    # 小车激光雷达提取墙面和关键点
    points = point_cloud2.read_points_list(
        onboard_laser_pc_msg, field_names=['x', 'y']
    )
    x_pos = [p.x for p in points]
    y_pos = [p.y for p in points]
    onboard_laser_pc = np.array([x_pos, y_pos]).T
    # 激光雷达->小车
    onboard_laser_pc_trans = transform(onboard_laser_pc, 0.08, 0, 180)
    # 小车激光雷达只保留小车附近的点
    onboard_laser_pc_trans = pc_filter(onboard_laser_pc_trans, *local_sensing_range)
    # 小车->毫米波雷达
    onboard_laser_pc_trans = transform_inverse(onboard_laser_pc_trans, 0.17, 0, 360-90)
    # 提取墙面和NLOS关键点
    onboard_walls, onboard_points = L_open_corner(onboard_laser_pc_trans)

    # 生成NLOS区域：遍历所有的格子，把这个格子转化为坐标，判断这个坐标是否在三角形内
    pos = np.array([
        np.array([[i for i in range(-H, H)] for j in range(H)]).flatten(),
        np.array([[j for i in range(-H, H)] for j in range(H)]).flatten()
    ]).T * range_res
    mask = isin_triangle(onboard_points['symmetric_barrier_corner'], onboard_points['inter2'], \
                         onboard_points['inter1'], pos).astype(np.float32).reshape(H, 2*H)

    # GT激光雷达提取墙面和关键点
    points = point_cloud2.read_points_list(
        gt_laser_pc_msg, field_names=['x', 'y']
    )
    x_pos = [p.x for p in points]
    y_pos = [p.y for p in points]
    gt_laser_pc = np.array([x_pos, y_pos]).T
    gt_laser_pc = pc_filter(gt_laser_pc, *gt_sensing_range)
    gt_walls, gt_points = L_open_corner_gt(gt_laser_pc)
    src = gt_points['reference_points'].T
    tar = onboard_points['reference_points'].T
    # 从GT激光雷达到小车激光雷达的变换矩阵
    R, T = registration(src, tar)

    # 把GT激光雷达转化到毫米波雷达坐标系下，并发布消息，为了能rviz可视化
    gt_laser_pc_trans = (R.dot(gt_laser_pc.T) + T).T
    msg = PointCloud2()
    msg.header = gt_laser_pc_msg.header
    msg.header.frame_id = 'radar'
    msg.height = 1
    msg.width = gt_laser_pc_trans.shape[0]
    msg.fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1)
    ]
    msg.is_bigendian = False
    msg.point_step = 8
    msg.row_step = msg.point_step * gt_laser_pc_trans.shape[0]
    msg.is_dense = True
    msg.data = gt_laser_pc_trans.astype(np.float32).tobytes()
    gt_pc_pub.publish(msg)

    # 目标检测
    # 加载模型，把RA热力图输入模型，通过nms得到结果
    RA_cart = (RA_cart - RA_cart.min()) / (RA_cart.max() - RA_cart.min())
    img = RA_cart.transpose(2, 0, 1)
    img = torch.from_numpy(img).float().to(device)
    img = img[None]
    with torch.no_grad():
        pred = model(img)
    pred_bbox = postprocess(pred, anchors, img_size)
    detections = nms_single_class(pred_bbox.cpu().numpy(), conf_thres, nms_thres)[0]

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
            # 发布RVIZ
            marker = Marker()
            marker.header = radar_adc_data.header
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.02  # 设置线宽
            point1 = Point()
            point1.x = xyxy_real[0]
            point1.y = xyxy_real[1]
            point1.z = 0
            point2 = Point()
            point2.x = xyxy_real[2]
            point2.y = xyxy_real[1]
            point2.z = 0
            point3 = Point()
            point3.x = xyxy_real[2]
            point3.y = xyxy_real[3]
            point3.z = 0
            point4 = Point()
            point4.x = xyxy_real[0]
            point4.y = xyxy_real[3]
            point4.z = 0
            marker.points = [point1, point2, point3, point4, point1]
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker.color.a = 1.0
            pred_bbox_pub.publish(marker)


if __name__ == '__main__':
    rospy.init_node("perception")
    gt_lidar_sub = message_filters.Subscriber('laser_point_cloud2', PointCloud2)
    onboard_lidar_sub = message_filters.Subscriber('laser_point_cloud', PointCloud2)
    radar_sub = message_filters.Subscriber('mmwave_radar_raw_data', adcData)

    model_path = "/home/agent/Code/yolov3_my/output/20231031_144012/model/model-99.pth"
    device = "cuda:0"
    anchors = torch.tensor([[10, 13], [16, 30], [33, 23]])
    img_size = [160, 320]
    conf_thres, nms_thres = 0.5, 0.4
    model = Darknet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    local_sensing_range = [-0.5, 5, -3, 3]  # 切割小车周围点云
    gt_sensing_range = [-4, 2, -4, 3]  # 切割gt周围点云

    gt_pc_pub = rospy.Publisher("laser_point_cloud_gt", PointCloud2, queue_size=10)
    pred_bbox_pub = rospy.Publisher("pred_bbox", Marker, queue_size=10)
    radar_pc_pub = rospy.Publisher("mmwave_radar_point_cloud", PointCloud2, queue_size=10)
    ts = message_filters.ApproximateTimeSynchronizer([gt_lidar_sub, onboard_lidar_sub, radar_sub], 10, 0.05)
    ts.registerCallback(perception)
    rospy.spin()
