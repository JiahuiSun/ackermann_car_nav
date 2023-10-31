import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from networkx.algorithms import bipartite
from scipy.optimize import linear_sum_assignment
import torch.nn as nn
import torch as th
import math
import random
from PIL import Image


def sample_points(box, points_num):
    # 对box上进行均匀采样取点
    # input: box[x,y,l,w](box中心的横纵坐标、box的长和宽); points_num(采样点的数量)
    # output: 采样点的横纵坐标的二维列表[[x1,y1], [x2,y2], ..., [xn,yn]]
    points = []
    x = box[0]
    y = box[1]
    length = box[2]
    width = box[3]

    x_max = x + length / 2
    x_min = x - length / 2
    y_max = y + width / 2
    y_min = y - width / 2

    x_list = np.random.uniform(x_min, x_max, points_num)
    y_list = np.random.uniform(y_min, y_max, points_num)

    points = np.stack((x_list, y_list), axis=1)

    return points


def velocity_cal(pos_list, last_pos_list, timestep):
    val_list = []
    for i in range(len(pos_list)):
        val = []
        v_x = (pos_list[i][0] - last_pos_list[i][0]) / (timestep / 1000)
        v_y = (pos_list[i][1] - last_pos_list[i][1]) / (timestep / 1000)
        val.append(v_x)
        val.append(v_y)
        val_list.append(val)
    return val_list


def visualize_grid_map(grid_data, cmap='viridis'):
    """
    可视化网格地图。

    参数：
    grid_data (numpy.ndarray)：包含网格数据的二维NumPy数组。
    cmap (str)：颜色映射名称（默认为'viridis'）。
    """
    rows, cols = grid_data.shape

    plt.imshow([[row[i] for row in grid_data] for i in range(len(grid_data[0]))], cmap=cmap, origin='lower',
               extent=(0, cols, 0, rows))

    for i in range(cols + 1):
        plt.axvline(x=i, color='black', linewidth=1)
    for i in range(rows + 1):
        plt.axhline(y=i, color='black', linewidth=1)

    plt.colorbar(label='Value')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Grid Map Visualization')
    plt.show()


def bresenham(grid, x0, y0, x1, y1, n, flag):
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    err = dx - dy

    while True:
        if 0 <= x0 < n and 0 <= y0 < n:
            grid[x0][y0] = flag
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

def intersection(x1, y1, x2, y2, x3, y3, x4, y4, type='line'):
    # Calculate the slopes and y-intercepts of the two lines
    m1 = (y2 - y1) / (x2 - x1) if x2 - x1 != 0 else float('inf')
    b1 = y1 - m1 * x1 if x2 - x1 != 0 else x1
    m2 = (y4 - y3) / (x4 - x3) if x4 - x3 != 0 else float('inf')
    b2 = y3 - m2 * x3 if x4 - x3 != 0 else x3

    # Check if the lines are parallel
    if m1 == m2:
        return None

    # Calculate the intersection point
    if m1 == float('inf'):
        x = x1
        y = m2 * x + b2
    elif m2 == float('inf'):
        x = x3
        y = m1 * x + b1
    else:
        x = (b2 - b1) / (m1 - m2)
        y = m1 * x + b1

    # Check if the intersection point is within the bounds of both line segments
    if type == 'line':
        if (x < min(x1, x2) or x > max(x1, x2)) and x1 != x2:
            return None
        if (y < min(y1, y2) or y > max(y1, y2)) and y1 != y2:
            return None
        if (x < min(x3, x4) or x > max(x3, x4)) and x3 != x4:
            return None
        if (y < min(y3, y4) or y > max(y3, y4)) and y3 != y4:
            return None
    elif type == 'ray':
        if (x-x1)*(x2-x1) < 0 or (y-y1)*(y2-y1) < 0:
            return None
        if (x < min(x3, x4) or x > max(x3, x4)) and x3 != x4:
            return None
        if (y < min(y3, y4) or y > max(y3, y4)) and y3 != y4:
            return None

    p = [x, y]
    return p

def find_intersection_point(x_min, x_max, y_min, y_max, point):
    x = point[0]
    y = point[1]
    p1 = intersection(0, 0, x, y, x_min, y_min, x_min, y_max)
    if p1 != None:
        return p1
    p2 = intersection(0, 0, x, y, x_min, y_min, x_max, y_min)
    if p2 != None:
        return p2
    p3 = intersection(0, 0, x, y, x_max, y_min, x_max, y_max)
    if p3 != None:
        return p3
    p4 = intersection(0, 0, x, y, x_min, y_max, x_max, y_max)
    if p4 != None:
        return p4


    return None


def map_lidar_pointcloud(grid, points, n, l):

    # black = [0, 0 ,0]
    # blue = [0, 0, 255]
    gray = [128, 128, 128]
    gray = np.array(gray)
    d = l / n
    x_min, x_max = -l / 8, 7 * l / 8
    y_min, y_max = -l / 2, l / 2
    cx, cy = n // 8, n // 2
    if points == None:
        return grid
    for point in points:
        if x_min <= point[0] < x_max and y_min <= point[1] < y_max:
            x = point[0]
            y = point[1]

            i = int((x - x_min) // d)
            j = int((y - y_min) // d)
            if i >= n:
                i = n - 1
            if j >= n:
                j = n - 1

            # s = find_intersection_point(x_min, x_max, y_min, y_max, point)
            # sx, sy = s

            # si = int((sx - x_min) // d)
            # sj = int((sy - y_min) // d)

            bresenham(grid, cx, cy, i, j, n, gray)

        else:
            s = find_intersection_point(x_min, x_max, y_min, y_max, point)
            sx, sy = s

            si = int((sx - x_min) // d)
            sj = int((sy - y_min) // d)    

            if si >= n:
                si = n - 1
            if sj >= n:
                sj = n - 1

            bresenham(grid, cx, cy, si, sj, n, gray)





    return grid


def map_radar_res(grid, theta, n, l, radar_res, pos):
    d = l / n
    x_min, x_max = -l / 2, l / 2
    y_min, y_max = -l / 2, l / 2
    
    for object in radar_res:
        x = object[0] - pos[0]
        y = object[1] - pos[1]
        cx = x * math.cos(theta) + y * math.sin(theta)
        cy = -x * math.sin(theta) + y * math.cos(theta)
        if x_min <= cx < x_max and y_min <= cy < y_max:
            ci = int((cx - x_min) // d)
            cj = int((cy - y_min) // d)
            for i in range(ci - 2, ci + 2):
                for j in range(cj - 2, cj + 2):
                    if 0 <= i < n and 0 <= j < n:
                        grid[i][j] = 2

    return grid

def map_self(grid, n, l, size):
    red = [255, 0, 0]
    red = np.array(red)
    d = l / n
    cx, cy = n // 8, n // 2

    cl, cw = size[0], size[1]
    nl = int(cl // d)
    nw = int(cw // d)
    for i in range(cx - nl // 2, cx + nl // 2):
        for j in range(cy - nw // 2, cy + nw // 2):
            grid[i][j] = red 
    
    return grid


def area_corner_obstacles_getting(robot_pos, robot_vel):
    x = robot_pos[0]
    y = robot_pos[1]
    v_x = robot_vel[0]
    v_y = robot_vel[1]

    wall = []
    corner_type = ''
    absent_wall = [0]
    static_obstacle_idx = []
    moving_obstacle_idx = []

    if -18 <= x < -10 and -7.5 <= y < -4:  # Area A
        pass

    elif -18 <= x < -10 and -4 <= y < -2:  # Area B
        static_obstacle_idx = [3, 4]
        moving_obstacle_idx = [0]
        wall = [-10, -4, -10, -2, -7, -2]
        corner_type = 'T'
        absent_wall = [1, -10, -4]

    elif -10 <= x < -7 and -7.5 <= y < -4:  # Area C
        static_obstacle_idx = [2]
        # moving_obstacle_idx = [0]
        wall = [-7, -2, -10, -4]
        corner_type = 'L'
        absent_wall = [1, -10, -4]

    elif -10 <= x < -7 and -4 <= y < -2:  # Area D
        pass

    elif -10 <= x < -7 and -2 <= y < 4.5:  # Area E
        if v_y < 0:
            static_obstacle_idx = [2]
            # moving_obstacle_idx = [0]
            wall = [-7, -4, -10, -2]
            corner_type = 'L'
            absent_wall = [1, -10, -4]
        else:
            static_obstacle_idx = [5]
            moving_obstacle_idx = [1]
            wall = [-10, -7.5, -7, 4.5]
            corner_type = 'L'
            # absent_wall = []

    elif -10 <= x < -7 and 4.5 <= y < 7.5:  # Area F
        pass

    elif -7 <= x < 4.5 and 4.5 <= y < 7.5:  # Area G
        if v_x < 0:
            static_obstacle_idx = [4]
            moving_obstacle_idx = [0]
            wall = [-10, -7.5, -7, 4.5]
            corner_type = 'L'
            # absent_wall = []
        else:
            static_obstacle_idx = [6, 7, 8, 9]
            moving_obstacle_idx = [2]
            wall = [4.5, 4.5, 4.5, 7.5, 10, 4.5]
            corner_type = 'T'
            # absent_wall = []

    elif 4.5 <= x < 10 and -3.5 <= y < 4.5:  # Area H
        static_obstacle_idx = [5]
        moving_obstacle_idx = [1]
        wall = [4.5, 4.5, 4.5, 7.5, 10, 4.5]
        corner_type = 'T'
        # absent_wall = []

    elif 4.5 <= x < 10 and 4.5 <= y < 7.5:  # Area I
        pass

    elif 4.5 <= x < 10 and 7.5 <= y < 15.5:  # Area J
        static_obstacle_idx = [5]
        moving_obstacle_idx = [1]
        wall = [4.5, 4.5, 4.5, 7.5, 10, 4.5]
        corner_type = 'T'
        # absent_wall = []

    return wall, absent_wall, corner_type, static_obstacle_idx, moving_obstacle_idx


def area_local_obstacles_getting():
    static_obstacle_idx = []
    moving_obstacle_idx = []
    return static_obstacle_idx, moving_obstacle_idx


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    # th.backends.cudnn.deterministic = True


def explained_variance(ypred, y):
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]

    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    """
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    return np.nan if vary == 0 else 1 - np.var(y - ypred) / vary


def maximum_weight_matching(B, top_nodes, delta=-1e2):
    dids, oids = nx.bipartite.sets(B, top_nodes)
    dids, oids = list(dids), list(oids)
    weights_sparse = bipartite.biadjacency_matrix(B, row_order=dids, column_order=oids, weight='Qi', format="coo")
    weights = np.full(weights_sparse.shape, delta)
    weights[weights_sparse.row, weights_sparse.col] = weights_sparse.data
    left_matches = linear_sum_assignment(weights, maximize=True)
    disp_res = [(dids[u], oids[v]) for u, v in zip(*left_matches) if weights[u, v] > delta]
    return disp_res


def maximum_weight_full_matching(B, top_nodes, delta=-1e2):
    # NOTE: B可以是任意的二分图，连通或非联通的，但一定返回full matching，否则返回None
    dids, oids = bipartite.sets(B, top_nodes)
    # 如果司机多于订单，那么一定不存在full matching
    if len(dids) > len(oids):
        return None, None, None

    # 匹配
    dids, oids = list(dids), list(oids)
    weights_sparse = bipartite.biadjacency_matrix(B, row_order=dids, column_order=oids, weight='Qi', format="coo")
    # 把不存在的连边设为很大的负数
    weights = np.full(weights_sparse.shape, delta)
    weights[weights_sparse.row, weights_sparse.col] = weights_sparse.data
    row_ind, col_ind = linear_sum_assignment(weights, maximize=True)
    # 去掉不存在的边
    disp_res = [(dids[u], oids[v]) for u, v in zip(row_ind, col_ind) if weights[u, v] > delta]
    # 如果匹配后有司机没有连边，则不存在full matching
    if len(dids) > len(disp_res):
        return None, None, None

    min_Qi = weights[row_ind, col_ind].min()
    sum_Qi = weights[row_ind, col_ind].sum()
    return disp_res, sum_Qi, min_Qi


def fair_maximum_weight_full_matching(B, dids, alpha=1.0):
    # NOTE: B初始是连通图，必须带fake order，不管删边过程B离散
    matching_record = []
    # 循环删边操作，但司机节点认为不变，如果删到没有连边了，则停止
    while True:
        disp_res, sum_Qi, min_Qi = maximum_weight_full_matching(B, dids)
        if disp_res is None:
            break
        matching_record.append((disp_res, sum_Qi, min_Qi))
        # print(f"sum_Qi: {sum_Qi}, min_Qi: {min_Qi}")

        # 如果权重小于最小的边权重，移除边
        remove_list = [(e[0], e[1]) for e in B.edges(data=True) if e[2]['Qi'] <= min_Qi]
        B.remove_edges_from(remove_list)

    # 从所有的matching中找一个满足要求的
    fair_disp_res, fair_sum_Qi, fair_min_Qi = [], 0.0, 0.0
    obj = -1e9
    for (disp_res, sum_Qi, min_Qi) in matching_record:
        if (alpha * sum_Qi / len(disp_res) + min_Qi) > obj:
            fair_disp_res, fair_sum_Qi, fair_min_Qi = disp_res, sum_Qi, min_Qi
            obj = alpha * sum_Qi / len(disp_res) + min_Qi
    return fair_disp_res, fair_sum_Qi, fair_min_Qi


def hopcroft_karp_matching(B, top_nodes):
    disp_res_dict = bipartite.hopcroft_karp_matching(B, top_nodes=top_nodes)
    disp_res = [(did, disp_res_dict[did]) for did in disp_res_dict if isinstance(did, int)]
    return disp_res


def miniblock(inp, oup, activation=True):
    """Construct a miniblock with given input/output-size and norm layer."""
    ret = [nn.Linear(inp, oup)]
    if activation:
        ret += [nn.ReLU()]
    return ret


def get_observation(grid_drivers_orders):
    """
    channels:
        [idle driver num, manned driver num, order orig num, order dest num]
    """
    # global observation
    global_info = th.zeros((len(grid_drivers_orders), 4))
    for gid, gvalue in grid_drivers_orders.items():
        global_info[gid][0] = len(gvalue['idle'])
        global_info[gid][1] = len(gvalue['manned'])
        global_info[gid][2] = len(gvalue['orig'])
        global_info[gid][3] = len(gvalue['dest'])
    return global_info


def get_observation2(edges, idle_drivers, manned_drivers, orders, G):
    """
    channels:
        [idle driver num, manned driver num, order orig num, order dest num]
    """
    # global observation
    obs = th.zeros((G.num_nodes(), 4))
    for driver in idle_drivers:
        obs[driver.cur_id][0] += 1
    for driver in manned_drivers:
        obs[driver.cur_id][1] += 1
    for order in orders:
        obs[order.orig_id][2] += 1
        obs[order.dest_id][3] += 1

    dids = list(set([od[0] for od in edges]))
    # edges += [(did, f'fake_{did}') for did in dids]
    oids = list(set([od[1] for od in edges]))
    B = nx.Graph()
    B.add_nodes_from(dids, bipartite=0)
    B.add_nodes_from(oids, bipartite=1)
    B.add_edges_from(edges)
    return obs, B


def get_group_feats(sim_round, edges, scalar, drivers={}, orders={}, config={}):
    """Construct bipartite graph B, and its feature matrix.

    """
    dids = list(set([od[0] for od in edges]))
    did2idx = {did: i for i, did in enumerate(dids)}
    oids = list(set([od[1] for od in edges]))
    oid2idx = {oid: i for i, oid in enumerate(oids)}
    edges = [(did, oid) for did in dids for oid in oids if (did, oid) in edges]
    edge2idx = {edge: i for i, edge in enumerate(edges)}

    # feature matrix
    driver_feats, hidden_feats, order_feats = [], [], []
    for did in dids:
        driver = drivers[did]
        # cur_time, cur_x, cur_y, is_busy
        driver_feats.append([sim_round, driver.cur_id, driver.is_busy])
        # NOTE: hidden_state is already tensor on the device
        hidden_feats.append(drivers[did].get_hidden_state())
    for oid in oids:
        # eta, dest_x, dest_y, fee
        order = orders[oid]
        order_feats.append([order.eta, order.dest_id, order.fee])
    driver_feats = np.array(driver_feats)
    order_feats = np.array(order_feats)
    # normalization for driver state
    # driver_feats[:, :-1] = scalar.transform(driver_feats[:, :-1], field=['departure_time', 'dest_x', 'dest_y'])
    # copy edges features
    edge_driver_feats = []
    edge_order_feats = []
    edge_hidden_feats = []
    for did, oid in edges:
        edge_driver_feats.append(driver_feats[did2idx[did]])
        edge_order_feats.append(order_feats[oid2idx[oid]])
        edge_hidden_feats.append(hidden_feats[did2idx[did]])

    driver_feats = th.from_numpy(driver_feats).float().to(config['device'])
    order_feats = th.from_numpy(order_feats).float().to(config['device'])
    hidden_feats = th.cat(hidden_feats, dim=0)
    edge_driver_feats = th.from_numpy(np.array(edge_driver_feats)).float().to(config['device'])
    edge_order_feats = th.from_numpy(np.array(edge_order_feats)).float().to(config['device'])
    edge_hidden_feats = th.cat(edge_hidden_feats, dim=0)

    # get group
    B = nx.Graph()
    B.add_nodes_from(dids, bipartite=0)
    B.add_nodes_from(oids, bipartite=1)
    B.add_edges_from(edges)
    Bu = bipartite.biadjacency_matrix(B, row_order=dids, column_order=oids).toarray().astype(np.bool)
    adj = th.tensor(Bu).float().to(config['device'])
    driver_group = [c.intersection(dids) for c in nx.connected_components(B)]

    feats = {'driver_feats': driver_feats,
             'order_feats': order_feats,
             'hidden_feats': hidden_feats,
             'edge_driver_feats': edge_driver_feats,
             'edge_order_feats': edge_order_feats,
             'edge_hidden_state': edge_hidden_feats,
             'adj': adj}
    info = {'dids': dids, 'did2idx': did2idx,
            'oids': oids, 'oid2idx': oid2idx,
            'edges': edges, 'edge2idx': edge2idx,
            'Bu': Bu, 'driver_group': driver_group, 'B': B}
    return feats, info


def get_single_feats(sim_round, drivers, scalar, config={}):
    driver_feats = []
    order_feats = []
    hidden_feats = []
    for driver in drivers:
        driver_feats.append([sim_round, driver.cur_id, driver.is_busy])
        order_feats.append([0, driver.cur_id, 0])
        hidden_feats.append(driver.get_hidden_state())
    driver_feats = np.array(driver_feats)
    order_feats = np.array(order_feats)
    # normalization for driver state and order action.
    # driver_feats[:, :-1] = scalar.transform(driver_feats[:, :-1], field=['departure_time', 'dest_x', 'dest_y'])
    # NOTE: fake order's eta, fee = 0, less than min, lead to negative number.
    # order_feats[:, :-1] = scalar.transform(order_feats[:, :-1], field=['eta', 'dest_x', 'dest_y'])
    driver_feats = th.from_numpy(driver_feats).float().to(config['device'])
    order_feats = th.from_numpy(order_feats).float().to(config['device'])
    hidden_feats = th.cat(hidden_feats, dim=0)
    feats = {'edge_driver_feats': driver_feats,
             'edge_order_feats': order_feats,
             'edge_hidden_state': hidden_feats}
    return feats


class StandardScalar(object):
    """正态分布归一化
    """

    def __init__(self, stats):
        self.stats = stats

    def transform(self, X, field=[]):
        """
        Args:
            X (np.array): N x d
            field (list): d features in order.

            X = (X - mean_vec) / std_vec
        """
        mean_vec = np.array([self.stats[key]['mean'] for key in field])
        std_vec = np.array([self.stats[key]['std'] for key in field])
        return (X - mean_vec) / std_vec

    def inverse(self, X, field=[]):
        mean_vec = np.array([self.stats[key]['mean'] for key in field])
        std_vec = np.array([self.stats[key]['std'] for key in field])
        return X * std_vec + mean_vec

    def inv(self, t, num=1, field='fee'):
        return t * self.stats[field]['std'] + num * self.stats[field]['mean']


class MinMaxScalar(object):
    """最小最大归一化
    """

    def __init__(self, stats):
        self.stats = stats

    def transform(self, X, field=[]):
        """
        Args:
            X (np.array): N x d
            field (list): d features in order.

            X = (X - min_vec) / (max_vec - min_vec)
        """
        min_vec = np.array([self.stats[key]['min'] for key in field])
        max_vec = np.array([self.stats[key]['max'] for key in field])
        return (X - min_vec) / (max_vec - min_vec)

    def inverse(self, X, field=[]):
        min_vec = np.array([self.stats[key]['min'] for key in field])
        max_vec = np.array([self.stats[key]['max'] for key in field])
        return X * (max_vec - min_vec) + min_vec

    def inv(self, t, num=1, field='fee'):
        """
        如果很多值加起来怎么反归一化
        """
        return t * (self.stats[field]['max'] - self.stats[field]['min']) + num * self.stats[field]['min']


class DecayThenFlatSchedule(object):
    def __init__(self, start, finish, time_length, decay="exp"):
        self.start = start
        self.finish = finish
        self.time_length = time_length
        self.delta = (self.start - self.finish) / self.time_length
        self.decay = decay

        if self.decay in ["exp"]:
            self.exp_scaling = (-1) * self.time_length / np.log(self.finish) if self.finish > 0 else 1

    def eval(self, T):
        if self.decay in ["linear"]:
            return max(self.finish, self.start - self.delta * T)
        elif self.decay in ["exp"]:
            return min(self.start, max(self.finish, np.exp(- T / self.exp_scaling)))
        else:
            raise Exception("No such decay method.")
        
def image_visualization(grid):
    x = y = 256
    im = Image.new("RGB", (x, y))
    for i in range(x):
        for j in range(y):
            im.putpixel((255 - j, 255-i), (int(grid[0][i][j]), int(grid[1][i][j]), int(grid[2][i][j])))
    # im.show()
    im.save('test_obs.png')
