import numpy as np
from sklearn.cluster import DBSCAN

from nlos_sensing import fit_line_ransac, find_end_point, line_symmetry_point
from nlos_sensing import intersection_of_2line, line_by_coef_p, get_span, line_by_2p


def fit_lines(laser_pc, n_line=2):
    """Iteratively extract lines with the most points via Ransac.
    Args:
        laser_pc: (N, 2)
        n_line: number of lines to extract
    Returns:
        fitted_lines: [[coef, points]]
        laser_pc: remaining points
    """
    min_points_inline = 20
    min_length_inline = 0.6
    ransac_sigma = 0.02
    ransac_iter = 200
    cluster = DBSCAN(eps=1, min_samples=20)
    # 提取墙面
    fitted_lines = []
    for i in range(n_line):
        if len(laser_pc) < min_points_inline:
            break
        coef, inlier_mask = fit_line_ransac(laser_pc, max_iter=ransac_iter, sigma=ransac_sigma)
        # 过滤在墙面的直线上但明显是噪声的点
        db = cluster.fit(laser_pc[inlier_mask])
        cluster_mask = np.zeros_like(inlier_mask) > 0
        cluster_mask[inlier_mask] = db.labels_ >= 0  # 即使前墙是2段的也保留
        inlier_mask = np.logical_and(inlier_mask, cluster_mask)
        inlier_points = laser_pc[inlier_mask]
        # 过滤非墙面的直线
        # 点数太少
        if len(inlier_points) < min_points_inline:
            continue
        # 跨度太小
        if get_span(inlier_points) < min_length_inline:
            continue
        outlier_mask = np.logical_not(inlier_mask)
        laser_pc = laser_pc[outlier_mask]
        fitted_lines.append([coef, inlier_points])
    return fitted_lines, laser_pc


def L_open_corner(laser_pc_onboard):
    """输入毫米波坐标系下小车激光雷达点云，输出转角各个墙面的直线和关键点
    Args:
        laser_pc_onboard: (N, 2)
    Returns:
        walls: 各个墙面参数和点云
        key_points: 转角坐标，NLOS区域5个相关坐标
    """
    # 提取墙面
    fitted_lines, remaining_pc = fit_lines(laser_pc_onboard, 2)

    # 区分墙面
    coef1, inlier_points1 = fitted_lines[0]
    center1 = np.mean(inlier_points1, axis=0)
    coef2, inlier_points2 = fitted_lines[1]
    center2 = np.mean(inlier_points2, axis=0)
    if center1[1] > center2[1]:  # 判断哪个是前墙
        far_wall, far_wall_pc = coef1, inlier_points1
        barrier_wall, barrier_wall_pc = coef2, inlier_points2
    else:
        far_wall, far_wall_pc = coef2, inlier_points2
        barrier_wall, barrier_wall_pc = coef1, inlier_points1

    # 提取关键点
    barrier_corner = np.array(find_end_point(barrier_wall_pc, 1)[1])
    symmtric_barrier_corner = line_symmetry_point(far_wall, barrier_corner)
    line_by_radar_and_corner = line_by_2p(np.array([0, 0]), barrier_corner)
    line_by_far_wall_and_symmtric_corner = line_by_coef_p(far_wall, symmtric_barrier_corner)
    inter1 = intersection_of_2line(line_by_radar_and_corner, far_wall)
    inter2 = intersection_of_2line(line_by_radar_and_corner, line_by_far_wall_and_symmtric_corner)
    inter3 = line_symmetry_point(far_wall, inter2)
    walls = {
        'far_wall': far_wall,
        'far_wall_pc': far_wall_pc,
        'barrier_wall': barrier_wall,
        'barrier_wall_pc': barrier_wall_pc
    }
    key_points = {
        'barrier_corner': barrier_corner,
        'symmetric_barrier_corner': symmtric_barrier_corner,
        'inter1': inter1,
        'inter2': inter2,
        'inter3': inter3
    }
    return walls, key_points


def L_open_corner_gt(laser_pc_gt):
    """输入GT激光雷达坐标系下的点云，输出转角各个墙面的直线和关键点
    Args:
        laser_pc_gt: (N, 2)
    Returns:
        walls: 各个墙面参数和点云
        key_points: 转角坐标，2个相关坐标
    """
    # 提取墙面
    fitted_lines, remaining_pc = fit_lines(laser_pc_gt, 2)

    # 区分墙面
    coef1, inlier_points1 = fitted_lines[0]
    center1 = np.mean(inlier_points1, axis=0)
    coef2, inlier_points2 = fitted_lines[1]
    center2 = np.mean(inlier_points2, axis=0)
    if center1[1] > center2[1]:  # 判断哪个是前墙
        far_wall, far_wall_pc = coef1, inlier_points1
        barrier_wall, barrier_wall_pc = coef2, inlier_points2
    else:
        far_wall, far_wall_pc = coef2, inlier_points2
        barrier_wall, barrier_wall_pc = coef1, inlier_points1

    # 提取关键点
    barrier_corner = np.array(find_end_point(barrier_wall_pc, 1)[1])
    symmtric_barrier_corner = line_symmetry_point(far_wall, barrier_corner)
    walls = {
        'far_wall': far_wall,
        'far_wall_pc': far_wall_pc,
        'barrier_wall': barrier_wall,
        'barrier_wall_pc': barrier_wall_pc
    }
    points = {
        'barrier_corner': barrier_corner,
        'symmetric_barrier_corner': symmtric_barrier_corner
    }
    return walls, points
