import numpy as np
from sklearn.linear_model import LinearRegression


def get_angle(points):
    """(1, 0)正方向和点中心的夹角
    """
    center = np.mean(points, axis=0)
    angle = np.arccos(center[0] / np.linalg.norm(center)) * 180 / np.pi
    return angle


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


def line_by_coef_p(coef, p):
    """
    Args:
        coef: [k, b]
        p: [x, y]
    
    Returns:
        coef_: [k_, b_], k_ = k, b_ = y - k * x
    """
    return np.array([coef[0], p[1]-coef[0]*p[0]])


def line_by_2p(p1, p2):
    """
    Args:
        p1: [x1, y1]
        p2: [x2, y2]
    
    Returns:
        coef: [k, b], y = kx + b
                k = (y2 - y1) / (x2 - x1)
                b = (x2*y1 - x1*y2) / (x2 - x1)
    """
    return np.array([(p2[1]-p1[1])/(p2[0]-p1[0]), (p2[0]*p1[1]-p1[0]*p2[1])/(p2[0]-p1[0])])


def intersection_of_2line(coef1, coef2):
    """
    Args:
        coef1: [k1, b1]
        coef2: [k2, b2]

    Returns:
        point: [x, y], x = (b2 - b1) / (k1 - k2), y = k1 * x + b1
    """
    x = (coef2[1] - coef1[1]) / (coef1[0] - coef2[0])
    y = coef1[0] * x + coef1[1]
    return np.array([x, y])


def isin_triangle(p1, p2, p3, points):
    """判断点集points是否在p1, p2, p3形成的三角形内。
    先让p1、p2、p3按逆时针排序，则三角形内部的点在p2-p1, p3-p2, p1-p3向量的左边，再通过叉乘判断。

    Args:
        p1: [x1, y1], shape=(2,)
        p2: [x2, y2]
        p3: [x3, y3]
        points (ndarry): shape=(N, 2)

    Returns:
        mask: 1 if p in triangle, 0, otherwise.
    """
    # 先让p1, p2, p3逆时针排序
    if np.cross(p2-p1, p3-p1) < 0:
        p3, p2 = p2, p3
    # 向量叉乘判断点在向量的左边
    mask1 = np.cross(p2-p1, points-p1) > 0
    mask2 = np.cross(p3-p2, points-p2) > 0
    mask3 = np.cross(p1-p3, points-p3) > 0
    mask = np.logical_and(mask1, np.logical_and(mask2, mask3))
    return mask


def line_symmetry_point(coef, point):
    """
    Args:
        coef: [k, b], y = kx + b
        point (ndarry): shape=(2,) or (N, 2)
    
    Returns:
        point_sym (ndarry): shape=(2,) or (N, 2)
                            x2 = x1 - 2 * k * (k * x1 - y1 + b) / (k^2 + 1)
                            y2 = y1 + 2 * (k * x1 - y1 + b) / (k^2 + 1)
    """
    if len(point.shape) < 2:
        x2 = point[0] - 2 * coef[0] * (coef[0] * point[0] - point[1] + coef[1]) / (coef[0]**2 + 1)
        y2 = point[1] + 2 * (coef[0] * point[0] - point[1] + coef[1]) / (coef[0]**2 + 1)
        return [x2, y2]
    else:
        x2 = point[:, 0] - 2 * coef[0] * (coef[0] * point[:, 0] - point[:, 1] + coef[1]) / (coef[0]**2 + 1)
        y2 = point[:, 1] + 2 * (coef[0] * point[:, 0] - point[:, 1] + coef[1]) / (coef[0]**2 + 1)
        return np.array([x2, y2]).T


def nlosFilterAndMapping(point_cloud, radar_pos, corner_args):
    """TODO: 现在只实现了1种转角
    """
    far_wall = corner_args['far_wall']
    barrier_wall = corner_args['barrier_wall']
    barrier_corner = corner_args['barrier_corner']

    # Filter
    far_map_corner = line_symmetry_point(far_wall, barrier_corner)
    radar_corner_line = line_by_2p(radar_pos, barrier_corner)
    inter1 = intersection_of_2line(far_wall, radar_corner_line)
    far_map_radar_corner_line = line_by_coef_p(far_wall, far_map_corner)
    inter2 = intersection_of_2line(far_map_radar_corner_line, radar_corner_line)
    flag = isin_triangle(far_map_corner, inter2, inter1, point_cloud[:, :2])

    # Mapping
    point_cloud_filter = point_cloud[flag]
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


def fit_line_ransac(points, max_iter=200, sigma=0.03):
    if len(points) < 2:
        return None, None
    best_score, best_inlier_mask = 0, []
    for _ in range(max_iter):
        p1_idx, p2_idx = np.random.choice(len(points), 2, replace=False)
        p1, p2 = points[p1_idx], points[p2_idx]
        # compute the distance from a point to the line by cross product
        line_dir = p1 - p2
        line_dir /= np.linalg.norm(line_dir)
        v = points - p2
        dis = np.abs(v.dot(
            np.array([[line_dir[1]], [-line_dir[0]]])
        )).squeeze()
        inlier_mask = dis <= sigma
        score = np.sum(inlier_mask)
        if best_score < score:
            best_score = score
            best_inlier_mask = inlier_mask
    inliers = points[best_inlier_mask]
    reg = LinearRegression().fit(inliers[:, :1], inliers[:, 1])
    coef = [reg.coef_[0], reg.intercept_]
    return coef, best_inlier_mask
