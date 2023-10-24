import numpy as np
from sklearn.cluster import DBSCAN

from nlos_sensing import fit_line_ransac, find_end_point, line_symmetry_point
from nlos_sensing import intersection_of_2line, line_by_coef_p, get_span, line_by_2p


min_points_inline = 20
min_length_inline = 0.6
ransac_sigma = 0.02
ransac_iter = 200
cluster = DBSCAN(eps=1, min_samples=20)


def L_open_corner(laser_pc_onboard):
    # 提取墙面
    fitted_lines = []
    for i in range(2):
        # 不用3条线就可以覆盖所有点
        if len(laser_pc_onboard) < min_points_inline:
            break
        coef, inlier_mask = fit_line_ransac(laser_pc_onboard, max_iter=ransac_iter, sigma=ransac_sigma)
        # 过滤在墙面的直线上但明显是噪声的点
        db = cluster.fit(laser_pc_onboard[inlier_mask])
        cluster_mask = np.zeros_like(inlier_mask) > 0
        cluster_mask[inlier_mask] = db.labels_ >= 0  # 即使前墙是2段的也保留
        inlier_mask = np.logical_and(inlier_mask, cluster_mask)
        inlier_points = laser_pc_onboard[inlier_mask]
        # 过滤非墙面的直线
        # 点数太少
        if len(inlier_points) < min_points_inline:
            continue
        # 跨度太小
        if get_span(inlier_points) < min_length_inline:
            continue
        outlier_mask = np.logical_not(inlier_mask)
        laser_pc_onboard = laser_pc_onboard[outlier_mask]
        fitted_lines.append([coef, inlier_points])

    # 区分墙面，目前就针对L开放型转角做
    coef1, inlier_points1 = fitted_lines[0]
    center1 = np.mean(inlier_points1, axis=0)
    coef2, inlier_points2 = fitted_lines[1]
    center2 = np.mean(inlier_points2, axis=0)
    assert np.abs(coef1[0]-coef2[0]) > 1, "parallel walls?"
    if center1[1] > center2[1]:  # 判断哪个是前墙
        far_wall = coef1
        barrier_wall = coef2
        barrier_corner = find_end_point(inlier_points2, 1)[1]
        barrier_corner = np.array(barrier_corner)
    else:
        far_wall = coef2
        barrier_wall = coef1
        barrier_corner = find_end_point(inlier_points1, 1)[1]
        barrier_corner = np.array(barrier_corner)
    symmtric_corner = line_symmetry_point(far_wall, barrier_corner)
    line_by_radar_and_corner = line_by_2p(np.array([0, 0]), barrier_corner)
    line_by_far_wall_and_symmtric_corner = line_by_coef_p(far_wall, symmtric_corner)
    inter1 = intersection_of_2line(line_by_radar_and_corner, far_wall)
    inter2 = intersection_of_2line(line_by_radar_and_corner, line_by_far_wall_and_symmtric_corner)
    inter3 = line_symmetry_point(far_wall, inter2)

    return far_wall, inlier_points1, barrier_wall, inlier_points2, barrier_corner, symmtric_corner, inter1, inter2, inter3
