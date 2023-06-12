import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
from sklearn.cluster import DBSCAN
import pickle

from nlos_sensing import transform, nlosFilterAndMapping, line_by_coef_p, line_by_vertical_coef_p
from nlos_sensing import get_span, find_end_point, fit_line_ransac, transform_line


local_sensing_range = [-0.5, 5, -3, 3]
gt_range = [-8, 2, 0, 1.5]  # 切割人的点云
min_points_inline = 20
min_length_inline = 0.6
ransac_sigma = 0.02
ransac_iter = 200
filter = DBSCAN(eps=1, min_samples=20)

stamp, accuracy = [], []

fig, ax = plt.subplots(figsize=(8, 8))
color_panel = ['ro', 'go', 'bo', 'co', 'wo', 'yo', 'mo', 'ko']


def init_fig():
    ax.clear()
    ax.set_xlabel('x(m)')
    ax.set_ylabel('y(m)')
    ax.set_xlim([-7, 2])
    ax.set_ylim([-4, 4])
    ax.tick_params(direction='in')


file_path = "/home/dingrong/Code/ackermann_car_nav/data/20230530/floor31_h1_120_L_180_angle_30_param2_2023-05-30-16-45-05"
fwrite = open(f"{file_path}.txt", 'w')
save_gif = False
def gen_data():
    with open(f"{file_path}.pkl", 'rb') as f:
        all_point_cloud = pickle.load(f)
    for t, laser_pc, laser_pc2, mmwave_pc, trans in all_point_cloud:
        # 从激光雷达坐标系到小车坐标系
        laser_point_cloud = transform(laser_pc, 0.08, 0, 180)
        # 过滤激光雷达点云，去掉距离小车中心5米以外的点
        flag_x = np.logical_and(laser_point_cloud[:, 0]>=local_sensing_range[0], laser_point_cloud[:, 0]<=local_sensing_range[1])
        flag_y = np.logical_and(laser_point_cloud[:, 1]>=local_sensing_range[2], laser_point_cloud[:, 1]<=local_sensing_range[3])
        flag = np.logical_and(flag_x, flag_y) 
        laser_point_cloud = laser_point_cloud[flag]
        # 标定激光雷达点云只保留人
        flag_x = np.logical_and(laser_pc2[:, 0]>=gt_range[0], laser_pc2[:, 0]<=gt_range[1])
        flag_y = np.logical_and(laser_pc2[:, 1]>=gt_range[2], laser_pc2[:, 1]<=gt_range[3])
        flag = np.logical_and(flag_x, flag_y) 
        laser_point_cloud2 = laser_pc2[flag]
        # 从毫米波雷达坐标系到小车坐标系
        mmwave_pc[:, :2] = transform(mmwave_pc[:, :2], 0.17, 0, 90)
        yield t, laser_point_cloud, laser_point_cloud2, mmwave_pc, trans


def visualize(result):
    init_fig()
    t, laser_point_cloud, laser_point_cloud2, mmwave_point_cloud, trans = result
    inter, theta = trans
    theta = theta * 180 / np.pi

    # 提取墙面
    fitted_lines = []
    for i in range(2):
        # 不用3条线就可以覆盖所有点
        if len(laser_point_cloud) < min_points_inline:
            break
        coef, inlier_mask = fit_line_ransac(laser_point_cloud, max_iter=ransac_iter, sigma=ransac_sigma)
        # 过滤在墙面的直线上但明显是噪声的点
        db = filter.fit(laser_point_cloud[inlier_mask])
        cluster_mask = np.zeros_like(inlier_mask) > 0
        cluster_mask[inlier_mask] = db.labels_ >= 0  # 即使前墙是2段的也保留
        inlier_mask = np.logical_and(inlier_mask, cluster_mask)
        inlier_points = laser_point_cloud[inlier_mask]
        # 过滤非墙面的直线
        # 点数太少
        if len(inlier_points) < min_points_inline:
            continue
        # 跨度太小
        if get_span(inlier_points) < min_length_inline:
            continue
        outlier_mask = np.logical_not(inlier_mask)
        laser_point_cloud = laser_point_cloud[outlier_mask]
        fitted_lines.append([coef, inlier_points])
    ax.set_title(f"Timestamp: {t:.2f}s")

    # 区分墙面
    if len(fitted_lines) == 2:
        corner_args = {}
        coef1, inlier_points1 = fitted_lines[0]
        center1 = np.mean(inlier_points1, axis=0)
        coef2, inlier_points2 = fitted_lines[1]
        center2 = np.mean(inlier_points2, axis=0)
        diff = np.abs(coef1[0]-coef2[0])
        if diff > 1:  # 两条线垂直，一个前墙，一个侧墙
            if center1[0] > center2[0]:  # 判断哪个是前墙
                corner_args['far_wall'] = coef1
                corner_args['barrier_wall'] = coef2
                barrier_corner = find_end_point(inlier_points2, 0)[1]
                corner_args['barrier_corner'] = np.array(barrier_corner) # x值最大的
            else:
                corner_args['far_wall'] = coef2
                corner_args['barrier_wall'] = coef1
                barrier_corner = find_end_point(inlier_points1, 0)[1]
                corner_args['barrier_corner'] = np.array(barrier_corner) # x值最大的
            # 过滤和映射
            point_cloud_nlos = nlosFilterAndMapping(mmwave_point_cloud, np.array([0.17, 0]), corner_args)

            # 把毫米波雷达、激光雷达点云变换到标定坐标系下
            point_cloud_nlos[:, :2] = transform(point_cloud_nlos[:, :2], inter[0], inter[1], 360-theta)
            mmwave_point_cloud[:, :2] = transform(mmwave_point_cloud[:, :2], inter[0], inter[1], 360-theta)
            inlier_points1 = transform(inlier_points1, inter[0], inter[1], 360-theta)
            inlier_points2 = transform(inlier_points2, inter[0], inter[1], 360-theta)
            laser_point_cloud = transform(laser_point_cloud, inter[0], inter[1], 360-theta)

            ax.plot(point_cloud_nlos[:, 0], point_cloud_nlos[:, 1], color_panel[-3], ms=2)
            ax.plot(mmwave_point_cloud[:, 0], mmwave_point_cloud[:, 1], color_panel[-2], ms=2)
            ax.plot(inlier_points1[:, 0], inlier_points1[:, 1], color_panel[0], ms=2)
            ax.plot(inlier_points2[:, 0], inlier_points2[:, 1], color_panel[1], ms=2)
            # ax.plot(laser_point_cloud[:, 0], laser_point_cloud[:, 1], color_panel[-1], ms=2)
            ax.plot(laser_point_cloud2[:, 0], laser_point_cloud2[:, 1], color_panel[-1], ms=2)
            # 统计精度
            if len(point_cloud_nlos) > 1 and len(laser_point_cloud2) > 0:
                pred = np.mean(point_cloud_nlos[:, :2], axis=0)
                gt = np.mean(laser_point_cloud2, axis=0)
                mae = np.abs(pred - gt)
                if mae[0] < 1 and mae[1] < 0.5:
                    stamp.append(t)
                    accuracy.append(mae)
                    fwrite.write(f"{t} {-pred[0]} {-gt[0]} {mae}\n")
                    print(t, -pred[0], -gt[0], mae)

        # 打box
        if len(laser_point_cloud2) > 0:
            box_center = np.mean(laser_point_cloud2, axis=0)
            vecB = laser_point_cloud2 - box_center
            center_wall_coef = transform_line(corner_args['far_wall'], inter[0], inter[1], 360-theta)

            center_parallel_wall = line_by_coef_p(center_wall_coef, box_center)
            vecA = np.array([1, center_parallel_wall[0]*(box_center[0]+1)+center_parallel_wall[1]-box_center[1]])
            proj = vecB.dot(vecA) / np.linalg.norm(vecA)
            max_idx, min_idx = np.argmax(proj), np.argmin(proj)
            box_length = proj[max_idx] - proj[min_idx]
            max_x, min_x = laser_point_cloud2[max_idx, 0], laser_point_cloud2[min_idx, 0]

            center_vertical_wall = line_by_vertical_coef_p(center_wall_coef, box_center)
            vecA = np.array([(box_center[1]+1-center_vertical_wall[1])/center_vertical_wall[0]-box_center[0], 1])
            proj = vecB.dot(vecA) / np.linalg.norm(vecA)
            max_idx, min_idx = np.argmax(proj), np.argmin(proj)
            box_width = proj[max_idx] - proj[min_idx]
            max_y, min_y = laser_point_cloud2[max_idx, 1], laser_point_cloud2[min_idx, 1]
            
            top_right = np.array([max_x, max_y])
            bottom_left = np.array([min_x, min_y])
            box_center = (top_right + bottom_left) / 2
            ax.plot(*box_center, color_panel[0], ms=2)
            ax.plot(*top_right, color_panel[1], ms=2)
            ax.plot(*bottom_left, color_panel[2], ms=2)

            # 画出box
            rect = patches.Rectangle(bottom_left, box_length, box_width, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)


ani = animation.FuncAnimation(
    fig, visualize, gen_data, interval=200,
    init_func=init_fig, repeat=True, save_count=200
)
writergif = animation.PillowWriter(fps=10)
if save_gif:
    ani.save(f"{file_path}.gif", writer=writergif)
plt.show()
accuracy = np.stack(accuracy)
fwrite.write(f"avg acc: {np.mean(accuracy, axis=0)}")
fwrite.write(f"detected frames: {len(stamp)}")
fwrite.close()
print("Average acc: ", np.mean(accuracy, axis=0))
print("Detected frames: ", len(stamp))
