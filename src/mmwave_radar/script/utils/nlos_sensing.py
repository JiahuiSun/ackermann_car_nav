import numpy as np
from sklearn.linear_model import LinearRegression


def xyxy2xywh(xyxy):
    """
    Args:
        xyxy: (4,)
    Returns:
        xywh: (4,)
    """
    if len(xyxy.shape) == 1:
        xywh = np.array([
            (xyxy[0]+xyxy[2])/2, (xyxy[1]+xyxy[3])/2, np.abs(xyxy[0]-xyxy[2]), np.abs(xyxy[1]-xyxy[3])
        ])
    else:
        xywh = np.zeros_like(xyxy)
        xywh[:, 0] = (xyxy[:, 0] + xyxy[:, 2]) / 2
        xywh[:, 1] = (xyxy[:, 1] + xyxy[:, 3]) / 2
        xywh[:, 2] = np.abs(xyxy[:, 0] - xyxy[:, 2])
        xyxy[:, 3] = np.abs(xyxy[:, 1] - xyxy[:, 3])
    return xywh


def pc_filter(pc, x_min, x_max, y_min, y_max):
    """
    Arguments:
        pc: shape=(N, 2), [[x, y]]
        x_min, x_max, y_min, y_max: 保留的点云范围
    Returns:
        pc_left: shape=(M, 2)
    """
    flag_x = np.logical_and(pc[:, 0]>=x_min, pc[:, 0]<=x_max)
    flag_y = np.logical_and(pc[:, 1]>=y_min, pc[:, 1]<=y_max)
    flag = np.logical_and(flag_x, flag_y)
    return pc[flag]


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
    """
    Args:
        points: Nx2, [x, y]
        axis: 按哪个坐标轴比较大小
    
    Returns:
        min_p, max_p: 在axis轴上最小和最大的点
    """
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


def line_by_vertical_coef_p(coef, p):
    """
    Args:
        coef: [k, b]
        p: [x, y]
    
    Returns:
        coef_: [k_, b_], k_ = -1/k, b_ = y - k_ * x
    """
    return np.array([-1/coef[0], p[1]+1/coef[0]*p[0]])


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


def point2line_distance(coef, point):
    """点到直线的距离
    """
    return np.abs(coef[0]*point[0]-point[1]+coef[1]) / np.sqrt(1+coef[0]**2)


def parallel_line_distance(coef, dist):
    """平行于给定直线，并且距离为dist
    """
    t0 = coef[1] + dist * np.sqrt(1 + coef[0]**2)
    t1 = coef[1] - dist * np.sqrt(1 + coef[0]**2)
    return [coef[0], t0], [coef[0], t1]


def bounding_box2(laser_point_cloud, delta_x=0.1, delta_y=0.1, fixed=False):
    """
    Arguments:
        laser_point_cloud: shape=(N, 2), [[x, y]]
        delta_x, delta_y: 比点云范围宽一点
    Returns:
        key_points: shape=(5, 2), [box_center, top_right, bottom_right, bottom_left, top_left]
        box_hw: [box_height, box_width]
    """
    if fixed:
        max_x, min_x = np.max(laser_point_cloud[:, 0]), np.min(laser_point_cloud[:, 0])
        max_y, min_y = np.max(laser_point_cloud[:, 1]), np.min(laser_point_cloud[:, 1])
        box_center = np.array([(max_x+min_x)/2, (max_y+min_y)/2])
        box_length, box_width = delta_x, delta_y
        top_right = box_center + np.array([box_length/2, box_width/2])
        bottom_right = box_center + np.array([box_length/2, -box_width/2])
        bottom_left = box_center + np.array([-box_length/2, -box_width/2])
        top_left = box_center + np.array([-box_length/2, box_width/2])
    else:
        max_x, min_x = np.max(laser_point_cloud[:, 0]), np.min(laser_point_cloud[:, 0])
        max_y, min_y = np.max(laser_point_cloud[:, 1]), np.min(laser_point_cloud[:, 1])
        top_right = np.array([max_x+delta_x, max_y+delta_y])
        bottom_right = np.array([max_x+delta_x, min_y-delta_y])
        bottom_left = np.array([min_x-delta_x, min_y-delta_y])
        top_left = np.array([min_x-delta_x, max_y+delta_y])
        box_center = (top_right + bottom_left) / 2
        box_width = top_right[0] - top_left[0]
        box_height = top_right[1] - bottom_right[1]
    key_points = np.array([box_center, top_right, bottom_right, bottom_left, top_left])
    box_hw = np.array([box_height, box_width])
    return key_points, box_hw


def bounding_box(laser_point_cloud, wall_coef, inter=None, theta=None, delta_x=0.10, delta_y=0.05):
    """给定一堆点，一条表示方向的线，找平行于该线的最小外接矩形
    Args:
        laser_point_cloud: Nx2, 点云
        wall_coef: bounding box边的方向
        inter, theta: 可能要对wall_coef进行坐标系转换
        delta_x, delta_y: 生成box时留一些safe region
    
    Returns:
        key_points: 中心点和四个顶点的坐标
        box_length, box_width: bounding box边长
    """
    box_center = np.mean(laser_point_cloud, axis=0)
    vecB = laser_point_cloud - box_center
    center_wall_coef = transform_line(wall_coef, inter[0], inter[1], theta) if inter is not None else wall_coef
    center_parallel_wall = line_by_coef_p(center_wall_coef, box_center)
    center_vertical_wall = line_by_vertical_coef_p(center_wall_coef, box_center)

    vecA = np.array([1, center_parallel_wall[0]*(box_center[0]+1)+center_parallel_wall[1]-box_center[1]])
    proj = vecB.dot(vecA) / np.linalg.norm(vecA)
    max_idx, min_idx = np.argmax(proj), np.argmin(proj)
    box_length = proj[max_idx] - proj[min_idx]
    # 找到了两个x方向的极值点
    max_x_vertical_line = line_by_coef_p(center_vertical_wall, laser_point_cloud[max_idx])
    min_x_vertical_line = line_by_coef_p(center_vertical_wall, laser_point_cloud[min_idx])
    # 平行直线，并且距离直线15cm，距离中心更远的直线
    coef1, coef2 = parallel_line_distance(max_x_vertical_line, delta_x)
    if point2line_distance(coef1, box_center) > point2line_distance(coef2, box_center):
        max_x_vertical_line = coef1
    else:
        max_x_vertical_line = coef2
    coef1, coef2 = parallel_line_distance(min_x_vertical_line, delta_x)
    if point2line_distance(coef1, box_center) > point2line_distance(coef2, box_center):
        min_x_vertical_line = coef1
    else:
        min_x_vertical_line = coef2

    vecA = np.array([(box_center[1]+1-center_vertical_wall[1])/center_vertical_wall[0]-box_center[0], 1])
    proj = vecB.dot(vecA) / np.linalg.norm(vecA)
    max_idx, min_idx = np.argmax(proj), np.argmin(proj)
    box_width = proj[max_idx] - proj[min_idx]
    # 找到2个y方向的极值点
    max_y_parallel_line = line_by_coef_p(center_parallel_wall, laser_point_cloud[max_idx])
    min_y_parallel_line = line_by_coef_p(center_parallel_wall, laser_point_cloud[min_idx])
    # 平行直线，并且距离直线15cm，距离中心更远的直线
    coef1, coef2 = parallel_line_distance(max_y_parallel_line, delta_y)
    if point2line_distance(coef1, box_center) > point2line_distance(coef2, box_center):
        max_y_parallel_line = coef1
    else:
        max_y_parallel_line = coef2
    coef1, coef2 = parallel_line_distance(min_y_parallel_line, delta_y)
    if point2line_distance(coef1, box_center) > point2line_distance(coef2, box_center):
        min_y_parallel_line = coef1
    else:
        min_y_parallel_line = coef2

    top_right = intersection_of_2line(max_x_vertical_line, max_y_parallel_line)
    top_left = intersection_of_2line(min_x_vertical_line, max_y_parallel_line)
    bottom_left = intersection_of_2line(min_x_vertical_line, min_y_parallel_line)
    bottom_right = intersection_of_2line(max_x_vertical_line, min_y_parallel_line)
    box_center = (top_right + bottom_left) / 2

    return (box_center, top_right, bottom_right, bottom_left, top_left), box_length, box_width


def nlos_filter_and_mapping(point_cloud, radar_pos, corner_args):
    """TODO: 现在只实现了1种转角
    """
    far_wall = corner_args['far_wall']
    barrier_wall = corner_args['barrier_wall']
    barrier_corner = corner_args['barrier_corner']

    # Filter
    far_map_corner = line_symmetry_point(far_wall, barrier_corner)
    radar_corner_line = line_by_2p(radar_pos, barrier_corner)
    far_map_radar_corner_line = line_by_coef_p(far_wall, far_map_corner)
    inter1 = intersection_of_2line(far_wall, radar_corner_line)
    inter2 = intersection_of_2line(far_map_radar_corner_line, radar_corner_line)
    # 只保留三角形内的点
    flag = isin_triangle(far_map_corner, inter2, inter1, point_cloud[:, :2])
    point_cloud_filter = point_cloud[flag]

    # Mapping
    point_cloud_filter[:, :2] = line_symmetry_point(far_wall, point_cloud_filter[:, :2])
    return point_cloud_filter


def nlos_mapping(point_cloud, radar_pos, corner_args):
    far_wall = corner_args['far_wall']
    barrier_wall = corner_args['barrier_wall']
    barrier_corner = corner_args['barrier_corner']

    far_map_corner = line_symmetry_point(far_wall, barrier_corner)
    far_map_radar = line_symmetry_point(far_wall, radar_pos)
    far_map_radar_corner = line_by_2p(far_map_radar, far_map_corner)
    far_corner_line = line_by_coef_p(far_wall, barrier_corner)
    inter1 = intersection_of_2line(far_map_radar_corner, far_wall)
    inter2 = intersection_of_2line(far_map_radar_corner, far_corner_line)
    # 当激光雷达点云中心点在三角形内，就映射
    center = np.mean(point_cloud, axis=0)
    flag = isin_triangle(barrier_corner, inter1, inter2, center)
    point_cloud_mapped = line_symmetry_point(far_wall, point_cloud) if flag else point_cloud
    return point_cloud_mapped


def transform(radar_xy, delta_x, delta_y, yaw):
    """Transform xy from radar coordinate to the world coordinate.
    Args:
        radar_xy: Nx2，雷达坐标系下的点云
        delta_x, delta_y: 雷达在世界坐标系中的坐标
        yaw: 世界坐标系逆时针旋转yaw度与雷达坐标系重合

    Returns:
        world_xy: Nx2，雷达坐标系下的点云转换到世界坐标系下
    """
    yaw = yaw * np.pi / 180
    rotation_matrix = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
    translation_vector = np.array([[delta_x, delta_y]])
    world_xy = radar_xy.dot(rotation_matrix) + translation_vector
    return world_xy


def transform_inverse(world_xy, delta_x, delta_y, yaw):
    """Transform xy from world coordinate to the radar coordinate.
    Args:
        world_xy: Nx2，世界坐标系下的点云
        delta_x, delta_y: 雷达在世界坐标系中的坐标
        yaw: 世界坐标系逆时针旋转yaw度与雷达坐标系重合

    Returns:
        radar_xy: Nx2，世界坐标系下的点云转换到雷达坐标系下
    """
    yaw = yaw * np.pi / 180
    rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    translation_vector = np.array([[delta_x, delta_y]])
    radar_xy = (world_xy - translation_vector).dot(rotation_matrix)
    return radar_xy


def transform_line(coef, delta_x, delta_y, yaw):
    two_p = np.array([
        [0, coef[1]],
        [1, coef[0]+coef[1]]
    ])
    two_p_ = transform(two_p, delta_x, delta_y, yaw)
    coef_ = line_by_2p(two_p_[0], two_p_[1])
    return coef_


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


def registration(src, tar):
    """Compute R and T from source points to target points.
    R.dot(src) + T = tar
    Args:
        source: (2, N)
        target: (2, N)
    Returns:
        R: (2, 2)
        T: (2, 1)
    """
    src_avg = np.mean(src, axis=-1, keepdims=True)
    tar_avg = np.mean(tar, axis=-1, keepdims=True)
    src -= src_avg
    tar -= tar_avg
    S = src.dot(tar.T)
    U, Sigma, Vh = np.linalg.svd(S)
    R = U.dot(Vh).T
    T = tar_avg - R.dot(src_avg)
    return R, T


if __name__ == '__main__':
    src = np.random.randn(2, 2)
    theta = 68 * np.pi / 180
    Rt = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    dt = np.array([[1.], [1.]])
    tar = Rt.dot(src) + dt
    R, T = registration(src, tar)
    print(Rt)
    print(R)
    print(dt)
    print(T)
