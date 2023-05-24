import numpy as np


def get_span(points):
    span = 0
    for ax in range(2):
        min_p, max_p = find_end_point(points, axis=ax)
        span = max(span, np.linalg.norm(max_p-min_p))
    return span


def find_end_point(points, axis=0):
    max_v, min_v = -np.inf, np.inf
    max_p, min_p = None, None
    for p in points:
        if p[axis] > max_v:
            max_v = p[axis]
            max_p = p
        if p[axis] < min_v:
            min_v = p[axis]
            min_p = p
    return min_p, max_p


def line_by_2p(p1, p2):
    """A ray from p1 to p2.
    """
    ABC = np.array([p2[1]-p1[1], p1[0]-p2[0], p2[0]*p1[1]-p1[0]*p2[1]])
    if p1[1] > p2[1]:
        ABC = -ABC
    return ABC


def line_symmetry_point(coef, point):
    """
    Args:
        coef: y = kx + b, coef = [k, b]
        point (ndarry): shape=(2,) or (N, 2)
    
    Returns:
        point_sym (ndarry): shape=(2,) or (N, 2)
    """
    if len(point.shape) < 2:
        x2 = point[0] - 2 * coef[0] * (coef[0] * point[0] - point[1] + coef[1]) / (coef[0]**2 + 1)
        y2 = point[1] + 2 * (coef[0] * point[0] - point[1] + coef[1]) / (coef[0]**2 + 1)
        return [x2, y2]
    else:
        x2 = point[:, 0] - 2 * coef[0] * (coef[0] * point[:, 0] - point[:, 1] + coef[1]) / (coef[0]**2 + 1)
        y2 = point[:, 1] + 2 * (coef[0] * point[:, 0] - point[:, 1] + coef[1]) / (coef[0]**2 + 1)
        return np.array([x2, y2]).T


def nlosFilterAndMapping(pointCloud, radar_pos, corner_args):
    """TODO: 现在只实现了1种转角
    """
    point_cloud_ext = np.concatenate([pointCloud[:, :2], np.ones((pointCloud.shape[0], 1))], axis=1)
    far_wall = corner_args['far_wall']
    barrier_wall = corner_args['barrier_wall']
    barrier_corner = corner_args['barrier_corner']

    # Filter
    far_map_corner = line_symmetry_point(far_wall, barrier_corner)
    far_map_radar = line_symmetry_point(far_wall, radar_pos)
    # 用前墙反射
    # 如果转角在雷达左边
    if barrier_corner[1] > 0:
        left_border = line_by_2p(radar_pos, barrier_corner)
        right_border = line_by_2p(far_map_radar, far_map_corner)
    # 如果转角在雷达右边
    elif barrier_corner[1] < 0:
        left_border = line_by_2p(far_map_radar, far_map_corner)
        right_border = line_by_2p(radar_pos, barrier_corner)
    else:
        raise Exception("Wrong input!")
    flag1 = point_cloud_ext.dot(left_border) > 0
    flag2 = point_cloud_ext.dot(right_border) > 0
    flag = np.logical_and(flag1, flag2)

    # Mapping
    point_cloud_filter = pointCloud[flag]
    point_cloud_filter[:, :2] = line_symmetry_point(far_wall, point_cloud_filter[:, :2])
    return point_cloud_filter


def transform(radar_xy, delta_x, delta_y, yaw):
    """Transform xy from radar coordinate to the world coordinate.
    Inputs:
        radar_xy: Nx2，雷达坐标系下的点云
        delta_x, delta_y: 雷达在世界坐标系中的坐标
        yaw: 雷达坐标系逆时针旋转yaw度与世界坐标系重合
    Return:
        world_xy: Nx2
    """
    yaw = yaw * np.pi / 180
    rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    translation_vector = np.array([[delta_x, delta_y]])
    world_xy = radar_xy.dot(rotation_matrix) + translation_vector
    return world_xy
