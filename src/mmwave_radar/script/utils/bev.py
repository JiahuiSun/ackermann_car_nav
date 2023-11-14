import numpy as np
from PIL import Image
from corner_type import fit_lines
from nlos_sensing import pc_filter


green = np.array([0, 255, 0])
red = np.array([255, 0, 0])
blue = np.array([0, 0, 255])
gray = np.array([105, 105, 105])
black = np.array([0, 0, 0])
pink = np.array([255, 192, 203])

class BEV():
    def __init__(self, height, width, n, ratio_l=6, ratio_w=2):
        self.height = height  # 长度6m
        self.width = width  # 宽度8m
        self.n = n  # 宽分桶数量 256
        self.d = width / n  # 分桶边长
        self.nx = int(height // self.d)  # x这条边分成多少份
        self.ny = n  # y这条边分成多少份
        self.ratio_l = ratio_l  # 中心位置在l的1/6处
        self.ratio_w = ratio_w  # 中心位置在w的1/2
        self.cx = self.nx // self.ratio_l  # 中心格子
        self.cy = self.ny // self.ratio_w  # 中心格子，即小车位置
        self.x_min, self.x_max = -1, 5
        self.y_min, self.y_max = -4, 4
        self.n_wall = 2
        self.car_size = [0.39, 0.24]

        self.init()

    def init(self):
        self.grid = np.zeros((3, self.nx, self.ny))
        self.map_car()

    def image_visualization(self, fpath='test.png'):
        im = Image.new("RGB", (self.nx, self.ny))
        for i in range(self.nx):
            for j in range(self.ny):
                im.putpixel((i, j), self.grid[:, i, j].astype(np.int8))
        im.save(fpath)

    def map_bbox(self, points, delta_x=0.0, delta_y=0.0, color=blue):
        x_min, x_max = points[:, 0].min()-delta_x, points[:, 0].max()+delta_x
        y_min, y_max = points[:, 1].min()-delta_y, points[:, 1].max()+delta_y
        i_low = int((x_min - self.x_min) // self.d)
        i_high = int((x_max - self.x_min) // self.d)
        j_low = int((y_min - self.y_min) // self.d)
        j_high = int((y_max - self.y_min) // self.d)
        for i in range(i_low, i_high):
            for j in range(j_low, j_high):
                if 0 <= i < self.nx and 0 <= j < self.ny:
                    self.grid[:, i, j] = color

    def map_points(self, points, color=pink):
        for point in points:
            i = int((point[0] - self.x_min) // self.d)
            j = int((point[1] - self.y_min) // self.d)
            if 0 <= i < self.nx and 0 <= j < self.ny:
                self.grid[:, i, j] = color

    def map_obstacle(self, points, color=blue):
        if len(points) == 0:
            return
        db = self.cluster.fit(points)
        labels = db.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        for i in range(n_clusters):
            cluster_points = points[labels == i]
            self.map_bbox(cluster_points, color=color)

    def map_car(self, color=red):
        cl, cw = self.car_size[0], self.car_size[1]
        nl = int(cl // self.d)
        nw = int(cw // self.d)
        for i in range(self.cx - nl // 2, self.cx + nl // 2):
            for j in range(self.cy - nw // 2, self.cy + nw // 2):
                self.grid[:, i, j] = color

    def mapping(self, onboard_laser_pc, *person_laser_pc_list):
        """
        Args:
            onboard_laser_pc: (N, 2)
            person_laser_pc_list: [(N, 2), ]
        Returns:
            grid: BEV map
        """
        onboard_laser_pc = pc_filter(onboard_laser_pc, self.x_min-5*self.d, self.x_max+5*self.d, self.y_min-5*self.d, self.y_max+5*self.d)
        fitted_lines, remaining_pc = fit_lines(onboard_laser_pc, self.n_wall)
        for coef, wall_pc in fitted_lines:
            self.map_points(wall_pc, color=pink)
        self.map_obstacle(remaining_pc, color=blue)
        for person_pc in person_laser_pc_list:
            self.map_bbox(person_pc, delta_x=0.1, delta_y=0.1, color=green)
        return self.grid
