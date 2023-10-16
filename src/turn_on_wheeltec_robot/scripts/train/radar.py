from utils import *
import math


class MyRadar:
    def __init__(self, fov, sample_points_num):

        # 初始化毫米波雷达的一些参数
        # fov: 雷达的视场角
        # sample_points_num: 采样点的个数

        self.radar_pos = None
        self.radar_rot = None
        self.fov = fov
        self.sample_points_num = sample_points_num

        self.offset_angle = 0
        self.wall = []
        self.absent_wall = []
        self.box_list = []
        self.vel_list = []
        self.corner_type = ''

    def set_radar(self, radar_pos, radar_rot, wall, absent_wall, box_list, vel_list, corner_type):

        # 根据雷达当前的位置，设置雷位置相关的一些参数
        # radar_pos: 雷达的位置[x,y]
        # radar_rot: 雷达的偏离x轴正方向的角度(rad)
        # wall: 雷达周围的墙壁的墙角坐标
        # absent_wall: 雷达周围的墙壁的缺失的墙壁线段[0/1, x1, x2]
        # box_list: 雷达周围的障碍物的情况

        self.radar_pos = radar_pos
        self.radar_rot = radar_rot
        self.wall = wall
        self.absent_wall = absent_wall
        self.box_list = box_list
        self.vel_list = vel_list
        self.corner_type = corner_type

    @staticmethod
    def check_intersection_with_box(line_segment, box):

        # 判断线段与box是否相交
        # input: line_segment[k,b,x1,x2](斜率、截距、起始和终止的横坐标); box[x,y,l,w](box中心的横纵坐标、box的长和宽)
        # output: 线段与box是否相交

        k, b, x1, x2 = line_segment
        x_line_min = min(x1, x2)
        x_line_max = max(x1, x2)

        x, y, l, w = box
        y_box_min = y - w / 2
        y_box_max = y + w / 2
        x_box_min = x - l / 2
        x_box_max = x + l / 2
        # 判断box的横坐标和线段起点终点的位置关系
        if x_box_max < x_line_min or x_box_min > x_line_max:  # 横坐标没有重叠 case 1&6
            intersection = 0
        else:  # 横坐标有重叠 case 2~5
            x_left = max(x_box_min, x_line_min)
            x_right = min(x_box_max, x_line_max)
            # 判断纵坐标有没有相交
            if max(k * x_left + b, k * x_right + b) < y_box_min or min(k * x_left + b, k * x_right + b) > y_box_max:
                intersection = 0
            else:
                intersection = 1
        return intersection

    @staticmethod
    def check_intersection_with_wall(line_segment, wall):

        # 判断线段与墙是否相交
        # input: line_segment[k,b,x1,x2](斜率、截距、起始和终止的横坐标); wall[x1,y1,x2,y2](外侧、内侧两个墙角的坐标)
        # output: 线段与墙是否相交

        intersection = 0
        # 判断墙角横坐标在直线上对应的横坐标是否大于墙角的纵坐标,只考虑右下的墙角，左上不会相交
        k, b, x1, x2 = line_segment
        x_line_min = min(x1, x2)
        x_line_max = max(x1, x2)

        x_wall_outside, y_wall_outside, x_wall_inside, y_wall_inside = wall

        # 判断墙角和光线起点终点的位置关系
        if x_wall_inside < x_line_min:  # 墙角横坐标在线段左边，不会相交
            if k * x_line_max + b < y_wall_inside:  # 理论上该墙角不会出现，为了后续一般化
                intersection = 1
        elif x_line_min <= x_wall_inside <= x_line_max:  # 墙角横坐标在线段中间
            if y_wall_outside > y_wall_inside:  # AB类的墙角（外侧墙的高于内侧墙）
                if y_wall_inside > k * x_wall_inside + b:
                    intersection = 1
            else:  # CD类的墙角（内侧墙的高于外侧墙）
                if y_wall_inside < k * x_wall_inside + b:
                    intersection = 1
        elif x_wall_inside > x_line_max:  # 墙角横坐标在线段右边，不会相交
            intersection = 0

        return intersection

    def check_in_range(self, line_segment):

        # 判断线段是否处于雷达的视场角的范围内
        # input: radar_pos[x,y](雷达的坐标); line_segment[k,b,x1,x2](斜率、截距、起始和终止的横坐标);
        #        fov(雷达的视场角); offset_angle(雷达的方向偏移中轴线的角度)
        # output: 线段是否处于雷达的FOV范围内

        in_range = 0
        fov = self.fov
        radar_pos = self.radar_pos
        offset_angle = self.radar_rot
        k, b, x1, x2 = line_segment

        x = x1 if x2 == radar_pos[0] else x2
        if x > radar_pos[0]:
            angle = math.atan(k)
        elif k > 0:
            angle = math.atan(k) - math.pi
        else:
            angle = math.atan(k) + math.pi

        angle_1 = offset_angle - fov / 2
        angle_2 = offset_angle + fov / 2

        if angle_1 <= angle <= angle_2:
            in_range = 1

        return in_range

    @staticmethod
    def judge_constraint(points_verify, illegal_surf, line_segments):

        # 判断线段是否满足约束,输出满足约束的线段
        # points_verify: 需要验证的反射点的坐标[x,y]
        # illegal_surf: 约束线段的list[[0,x11,x12], [0,x21,x22]]

        iter = 0
        for point in points_verify:
            for line in illegal_surf:
                if line[0] == 0 and point[1] == line[3]:  # x方向的线段,且点的纵坐标与线段的纵坐标相同
                    if min(line[1], line[2]) < point[0] < max(line[1], line[2]):  # 点落在约束面上，去除对应的line(2条线段)
                        del line_segments[iter]
                        del line_segments[iter]
                        break  # 任意一个约束不满足就删除线段
                elif line[0] == 1 and point[0] == line[3]:  # y方向的线段,且点的横坐标与线段的横坐标相同
                    if min(line[1], line[2]) < point[1] < max(line[1], line[2]):  # 点落在约束面上，去除对应的line(2条线段)
                        del line_segments[iter]
                        del line_segments[iter]
                        break
            iter = iter + 1

        return line_segments

    def calculation_of_reflection(self, tar_pos):

        # 给定雷达坐标、反射位置的坐标以及墙壁的位置，计算反射路径的两条线段
        # input: tar_pos[x,y]（目标点的坐标）
        # output: line_segments[[k1,b1,x11,x12],[k2,b2,x21,x22]](两条线段的斜率、截距、起始和终止的横坐标)

        radar_pos = self.radar_pos
        wall = self.wall
        corner_type = self.corner_type
        absent_wall = self.absent_wall

        direct_see_flag = False

        line_segments = []
        x0, y0 = radar_pos
        x3, y3 = tar_pos
        illegal_surf = []
        points_verify = []

        if corner_type == 'L':
            # L型,wall[x1,y1,x2,y2](L型时：wall中包含(x1,y1)外侧、(x2,y2)内侧两个墙角的坐标)
            # absent_wall[0,x,y]0/1判断是否不完整（0-完整，1-不完整），x1,y1为不完整的墙的特征点坐标
            ###################################################################################################
            ## L型转角[分上表面反射和左右表面反射两种情况]，其余转角场景数学公式相同。
            ### L-1.从上反射面反射
            y1 = wall[1]
            x4_L1 = (x3 * y0 - x3 * y1 - x0 * y1 + x0 * y3) / (y0 - 2 * y1 + y3)
            y4_L1 = y1
            tan_a_L1 = (x3 - x4_L1) / (y1 - y3)

            k1_L1 = - 1 / tan_a_L1
            b1_L1 = x3 / tan_a_L1 + y3
            x11_L1 = x4_L1
            x12_L1 = x3

            k2_L1 = 1 / tan_a_L1
            b2_L1 = - x0 / tan_a_L1 + y0
            x21_L1 = x0
            x22_L1 = x4_L1

            points_verify.append([x4_L1, y4_L1])
            line_segments.append([k1_L1, b1_L1, x11_L1, x12_L1])
            line_segments.append([k2_L1, b2_L1, x21_L1, x22_L1])
            ### L-2.从左/右反射面反射
            x1 = wall[0]
            x4_L2 = x1
            y4_L2 = (-x0 * y3 + x4_L2 * y3 - x3 * y0 + x4_L2 * y0) / (2 * x4_L2 - x3 - x0)
            tan_a_L2 = (y3 - y4_L2) / (x3 - x4_L2)

            k1_L2 = tan_a_L2
            b1_L2 = -x3 * tan_a_L2 + y3
            x11_L2 = x4_L2
            x12_L2 = x3

            k2_L2 = -tan_a_L2
            b2_L2 = x0 * tan_a_L2 + y0
            x21_L2 = x4_L2
            x22_L2 = x0
            points_verify.append([x4_L2, y4_L2])
            line_segments.append([k1_L2, b1_L2, x11_L2, x12_L2])
            line_segments.append([k2_L2, b2_L2, x21_L2, x22_L2])

            # 计算约束面
            if absent_wall[0] == 1:  # 不完整
                x_ab = absent_wall[1]
                y_ab = absent_wall[2]
                if x_ab == x1:
                    illegal_surf.append([1, y1, y_ab, x1])
                elif y_ab == y1:
                    illegal_surf.append([0, x1, x_ab, y1])

                line_segments = self.judge_constraint(points_verify, illegal_surf, line_segments)

        elif corner_type == 'T':
            #   T型,wall[x1,y1,x2,y2,x5,y5](T型时：wall中包含T型左右两个墙角的坐标和墙面任一点的坐标(x5,y5),
            #   absent_wall[0,x1,y1]  0/1判断是否不完整（0-完整，1-不完整），x1,y1为不完整的墙的特征点坐标
            #################################################################
            #  1.把问题转换成L型，根据给定参数相对位置求得对应L型的(x1,y1,x2,y2)，并计算对应的约束面
            #################################################################

            x1_T, y1_T, x2_T, y2_T, x5_T, y5_T = wall
            x1_L, y1_l, x2_L, y2_L = 0, 0, 0, 0

            if y1_T == y2_T:  ##### 正的T型转角

                line_1 = [1, y1_T, y5_T, min(x1_T, x2_T)]
                line_2 = [0, x1_T, x2_T, y1_T]
                line_3 = [1, y1_T, y5_T, max(x1_T, x2_T)]
                ### 判断墙面是否完整并添加相应约束
                abs_len = 2  # 这里设置缺失的墙的长度为2m，可以修正
                if absent_wall[0] == 1:  # 墙面不完整
                    if y5_T > absent_wall[2]:  # 缺失的是下面
                        illegal_surf.append([1, y1_T, y1_T - abs_len, absent_wall[1]])
                    else:
                        illegal_surf.append([1, y1_T, y1_T + abs_len, absent_wall[1]])

                if x0 < min(x1_T, x2_T):  # 雷达在左侧
                    illegal_surf.append(line_2)
                    illegal_surf.append(line_3)

                    x1_L = max(x1_T, x2_T)  # 外侧
                    y1_L = y5_T
                    x2_L = min(x1_T, x2_T)  # 内侧
                    y2_L = y1_T  # y1或者y2

                elif min(x1_T, x2_T) <= x0 < max(x1_T, x2_T):  # 雷达在中间
                    illegal_surf.append(line_1)
                    illegal_surf.append(line_3)
                    #######################################################
                    ####### 雷达在中间的时候需要额外根据target位置决定内、外侧的墙点
                    #######################################################
                    if x3 < min(x1_T, x2_T):  # target在墙角左侧
                        x1_L = max(x1_T, x2_T)  # 外侧
                        y1_L = y5_T
                        x2_L = min(x1_T, x2_T)  # 内侧
                        y2_L = y1_T  # y1或者y2
                    elif x3 > max(x1_T, x2_T):  # target在墙角右侧
                        x1_L = min(x1_T, x2_T)  # 外侧
                        y1_L = y5_T
                        x2_L = max(x1_T, x2_T)  # 内侧
                        y2_L = y1_T  # y1或者y2
                    else:
                        direct_see_flag = True
                    #### 不考虑中间target的位置

                else:  # 雷达在右侧
                    illegal_surf.append(line_1)
                    illegal_surf.append(line_2)

                    x1_L = min(x1_T, x2_T)  # 外侧
                    y1_L = y5_T
                    x2_L = max(x1_T, x2_T)  # 内侧
                    y2_L = y1_T  # y1或者y2

            elif x1_T == x2_T:  # 左/右转90度的T型转角

                line_1 = [0, x1_T, x5_T, max(y1_T, y2_T)]
                line_2 = [1, y1_T, y2_T, x1_T]
                line_3 = [0, x1_T, x5_T, min(y1_T, y2_T)]

                ### 判断墙面是否完整并添加相应约束
                abs_len = 2  # 这里设置缺失的墙的长度为2m，可以修正
                if absent_wall[0] == 1:  # 墙面不完整
                    if x5_T > absent_wall[1]:  # 缺失的是左边
                        illegal_surf.append([0, x1_T, x1_T - abs_len, absent_wall[2]])
                    else:
                        illegal_surf.append([0, x1_T, x1_T + abs_len, absent_wall[2]])

                if y0 < min(y1_T, y2_T):  # 雷达在下侧
                    illegal_surf.append(line_1)
                    illegal_surf.append(line_2)
                    x1_L = x5_T  # 外侧
                    y1_L = max(y1_T, y2_T)
                    x2_L = x1_T  # 内侧 x1/x2
                    y2_L = min(y1_T, y2_T)

                elif min(y1_T, y2_T) <= y0 < max(y1_T, y2_T):  ############ 雷达在中间
                    illegal_surf.append(line_1)
                    illegal_surf.append(line_3)
                    #######################################################
                    ####### 雷达在中间的时候需要额外根据target位置决定内、外侧的墙点
                    #######################################################
                    if y3 > max(y1_T, y2_T):  # target在上方
                        x1_L = x5_T  # 外侧
                        y1_L = min(y1_T, y2_T)
                        x2_L = x1_T  # 内侧 x1/x2
                        y2_L = max(y1_T, y2_T)
                    elif y3 < min(y1_T, y2_T):
                        x1_L = x5_T  # 外侧
                        y1_L = max(y1_T, y2_T)
                        x2_L = x1_T  # 内侧 x1/x2
                        y2_L = min(y1_T, y2_T)
                    else:
                        direct_see_flag = True
                    # 不考虑在中间的target(在可视范围内)

                else:  # 雷达在上侧
                    illegal_surf.append(line_2)
                    illegal_surf.append(line_3)
                    x1_L = x5_T  # 外侧
                    y1_L = min(y1_T, y2_T)
                    x2_L = x1_T  # 内侧 x1/x2
                    y2_L = max(y1_T, y2_T)

            #################################################################
            # 2.按照L型计算4条线段
            #################################################################
            # points_verify = []  # 3中需要验证约束条件的中间反射点
            ### L-1.从上反射面反射
            if not direct_see_flag:
                y1 = y1_L
                x4_L1 = (x3 * y0 - x3 * y1 - x0 * y1 + x0 * y3) / (y0 - 2 * y1 + y3)
                y4_L1 = y1
                tan_a_L1 = (x3 - x4_L1) / (y1 - y3)

                k1_L1 = - 1 / tan_a_L1
                b1_L1 = x3 / tan_a_L1 + y3
                x11_L1 = x4_L1
                x12_L1 = x3

                k2_L1 = 1 / tan_a_L1
                b2_L1 = - x0 / tan_a_L1 + y0
                x21_L1 = x0
                x22_L1 = x4_L1

                line_segments.extend([[k1_L1, b1_L1, x11_L1, x12_L1], [k2_L1, b2_L1, x21_L1, x22_L1]])
                points_verify.append([x4_L1, y4_L1])
                ### L-2.从左/右反射面反射
                x1 = x1_L
                x4_L2 = x1
                y4_L2 = (-x0 * y3 + x4_L2 * y3 - x3 * y0 + x4_L2 * y0) / (2 * x4_L2 - x3 - x0)
                tan_a_L2 = (y3 - y4_L2) / (x3 - x4_L2)

                k1_L2 = tan_a_L2
                b1_L2 = -x3 * tan_a_L2 + y3
                x11_L2 = x4_L2
                x12_L2 = x3

                k2_L2 = -tan_a_L2
                b2_L2 = x0 * tan_a_L2 + y0
                x21_L2 = x4_L2
                x22_L2 = x0
                line_segments.extend([[k1_L2, b1_L2, x11_L2, x12_L2], [k2_L2, b2_L2, x21_L2, x22_L2]])
                points_verify.append([x4_L2, y4_L2])
            #################################################################
            # 3.根据约束面判断光线是否有效
            #################################################################
            line_segments = self.judge_constraint(points_verify, illegal_surf, line_segments)

        elif corner_type == 'X':
            # 十字路口型,wall[x1,y1,x2,y2,x5,y5,x6,y6](十字路口型时：wall中包含4个墙角的坐标)
            # absent_wall[0,x1,y1] 0/1判断是否不完整（0-完整，1-不完整），x1,y1为不完整的墙的特征点坐标
            #################################################################
            #  1.把问题转换成L型，根据给定参数相对位置求得对应L型的(x1,y1,x2,y2)，并计算对应的约束面
            #################################################################
            x1_T, y1_T, x2_T, y2_T, x5_T, y5_T, x6_T, y6_T = wall
            x1_L, y1_L, x2_L, y2_L = 0, 0, 0, 0

            line_1 = [1, y1_T, y5_T, min(x1_T, x2_T)]
            line_2 = [0, x1_T, x2_T, y1_T]
            line_3 = [1, y1_T, y5_T, max(x1_T, x2_T)]
            line_4 = []
            # 左上，右上，左下，右下 = 0,1,2,3
            wall_points = []

            p1, p2, p5, p6 = [x1_T, y1_T], [x2_T, y2_T], [x5_T, y5_T], [x6_T, y6_T]

            # 判断4个wall坐标的相对位置(可优化)
            if x1_T == x2_T:  # x5=x6
                if y1_T > y2_T:
                    if x1_T > x5_T:
                        if y5_T > y6_T:
                            wall_points.extend([p5, p1, p6, p2])
                        else:
                            wall_points.extend([p6, p1, p5, p2])
                    else:
                        if y5_T > y6_T:
                            wall_points.extend([p1, p5, p2, p6])
                        else:
                            wall_points.extend([p5, p1, p6, p2])
                else:  # (y1_T <= y2_T)
                    if x1_T > x5_T:
                        if y5_T > y6_T:
                            wall_points.extend([p5, p2, p6, p1])
                        else:
                            wall_points.extend([p6, p2, p5, p1])
                    else:
                        if y5_T > y6_T:
                            wall_points.extend([p2, p5, p1, p6])
                        else:
                            wall_points.extend([p5, p2, p6, p1])

            elif x1_T == x5_T:  # x2 = x6
                if y1_T > y5_T:
                    if x1_T > x6_T:
                        if y2_T > y6_T:
                            wall_points.extend([p2, p1, p6, p5])
                        else:
                            wall_points.extend([p6, p1, p2, p5])
                    else:
                        if y2_T > y6_T:
                            wall_points.extend([p1, p2, p5, p6])
                        else:
                            wall_points.extend([p1, p6, p5, p2])
                else:  # y1_T <= y5_T
                    if x1_T > x6_T:
                        if y2_T > y6_T:
                            wall_points.extend([p2, p5, p6, p1])
                        else:
                            wall_points.extend([p6, p5, p2, p1])
                    else:
                        if y2_T > y6_T:
                            wall_points.extend([p5, p2, p1, p6])
                        else:
                            wall_points.extend([p5, p6, p1, p2])
            elif x1_T == x6_T:  # x2 = x5
                if y1_T > y6_T:
                    if x1_T > x2_T:
                        if y2_T > y5_T:
                            wall_points.extend([p2, p1, p5, p6])
                        else:
                            wall_points.extend([p5, p1, p2, p6])
                    else:
                        if y2_T > y5_T:
                            wall_points.extend([p1, p2, p6, p5])
                        else:
                            wall_points.extend([p1, p5, p6, p2])
                else:
                    if x1_T > x2_T:
                        if y2_T > y5_T:
                            wall_points.extend([p2, p6, p5, p1])
                        else:
                            wall_points.extend([p5, p6, p2, p1])
                    else:
                        if y2_T > y5_T:
                            wall_points.extend([p6, p2, p1, p5])
                        else:
                            wall_points.extend([p6, p5, p1, p2])

            # 根据墙的完整度添加约束
            if absent_wall[0] == 1:  # 不完整
                x_ab, y_ab = absent_wall[1:]
                if y_ab == wall_points[0][1] or y_ab == wall_points[2][1]:  # 约束线段在x方向,判断左右
                    if x_ab < wall_points[0][0]:  # 约束线段在转角左侧
                        illegal_surf.append([0, x_ab, wall_points[0][0], y_ab])
                    elif x_ab > wall_points[1][0]:  # 约束线段在转角右侧
                        illegal_surf.append([0, x_ab, wall_points[1][0], y_ab])
                elif x_ab == wall_points[0][0] or x_ab == wall_points[1][0]:  # 约束线段在y方向，判断上下
                    if y_ab < wall_points[2][1]:  # 约束线段在转角下侧
                        illegal_surf.append([1, y_ab, wall_points[2][1], x_ab])
                    elif y_ab > wall_points[0][1]:  # 约束线段在转角上侧
                        illegal_surf.append([1, y_ab, wall_points[0][1], x_ab])

            line_1 = [0, wall_points[0][0], wall_points[1][0], wall_points[0][1]]
            line_2 = [1, wall_points[0][1], wall_points[2][1], wall_points[0][0]]
            line_3 = [0, wall_points[2][0], wall_points[3][0], wall_points[2][1]]
            line_4 = [1, wall_points[1][1], wall_points[3][1], wall_points[1][0]]

            # 计算对应L型求解方式的参数和约束面
            if x0 < wall_points[0][0]:  # 雷达在左边
                illegal_surf.extend([line_1, line_3, line_4])
                if y3 > wall_points[0][1]:  # target在上方
                    x1_L, y1_L = wall_points[3]  # 外侧
                    x2_L, y2_L = wall_points[0]  # 内侧
                elif y3 < wall_points[2][1]:  # target在下方
                    x1_L, y1_L = wall_points[1]  # 外侧
                    x2_L, y2_L = wall_points[2]  # 内侧
            elif x0 > wall_points[1][0]:  # 雷达在右边
                illegal_surf.extend([line_1, line_2, line_3])
                if y3 > wall_points[0][1]:  # target在上方
                    x1_L, y1_L = wall_points[2]  # 外侧
                    x2_L, y2_L = wall_points[1]  # 内侧
                elif y3 < wall_points[2][1]:  # target在下方
                    x1_L, y1_L = wall_points[0]  # 外侧
                    x2_L, y2_L = wall_points[3]  # 内侧
            elif y0 > wall_points[0][1]:  # 雷达在上方
                illegal_surf.extend([line_2, line_3, line_4])
                if y3 > wall_points[0][1]:  # target在左方
                    x1_L, y1_L = wall_points[3]  # 外侧
                    x2_L, y2_L = wall_points[0]  # 内侧
                elif y3 < wall_points[2][1]:  # target在右方
                    x1_L, y1_L = wall_points[2]  # 外侧
                    x2_L, y2_L = wall_points[1]  # 内侧
            elif y0 < wall_points[2][1]:  # 雷达在下方
                illegal_surf.extend([line_1, line_2, line_4])
                if y3 > wall_points[0][1]:  # target在左方
                    x1_L, y1_L = wall_points[1]  # 外侧
                    x2_L, y2_L = wall_points[2]  # 内侧
                elif y3 < wall_points[2][1]:  # target在右方
                    x1_L, y1_L = wall_points[3]  # 外侧
                    x2_L, y2_L = wall_points[0]  # 内侧

            #################################################################
            # 2.按照L型计算4条线段
            #################################################################
            # points_verify = []  # 3中需要验证约束条件的中间反射点
            ### L-1.从上反射面反射
            y1 = y1_L
            x4_L1 = (x3 * y0 - x3 * y1 - x0 * y1 + x0 * y3) / (y0 - 2 * y1 + y3)
            y4_L1 = y1
            tan_a_L1 = (x3 - x4_L1) / (y1 - y3)

            k1_L1 = - 1 / tan_a_L1
            b1_L1 = x3 / tan_a_L1 + y3
            x11_L1 = x4_L1
            x12_L1 = x3

            k2_L1 = 1 / tan_a_L1
            b2_L1 = - x0 / tan_a_L1 + y0
            x21_L1 = x0
            x22_L1 = x4_L1

            line_segments.extend([[k1_L1, b1_L1, x11_L1, x12_L1], [k2_L1, b2_L1, x21_L1, x22_L1]])
            points_verify.append([x4_L1, y4_L1])
            ### L-2.从左/右反射面反射
            x1 = x1_L
            x4_L2 = x1
            y4_L2 = (-x0 * y3 + x4_L2 * y3 - x3 * y0 + x4_L2 * y0) / (2 * x4_L2 - x3 - x0)
            tan_a_L2 = (y3 - y4_L2) / (x3 - x4_L2)

            k1_L2 = tan_a_L2
            b1_L2 = -x3 * tan_a_L2 + y3
            x11_L2 = x4_L2
            x12_L2 = x3

            k2_L2 = -tan_a_L2
            b2_L2 = x0 * tan_a_L2 + y0
            x21_L2 = x4_L2
            x22_L2 = x0

            line_segments.extend([[k1_L2, b1_L2, x11_L2, x12_L2], [k2_L2, b2_L2, x21_L2, x22_L2]])
            points_verify.append([x4_L2, y4_L2])
            #################################################################
            # 3.根据约束面判断光线是否有效
            #################################################################
            line_segments = self.judge_constraint(points_verify, illegal_surf, line_segments)

        return line_segments

    def get_wall_position(self, tar_pos):

        # 根据转角的类型和墙的坐标给出对应L型内外侧墙的坐标
        # tar_pos: 目标位置(x,y)

        x0, y0 = self.radar_pos
        x3, y3 = tar_pos
        corner_type = self.corner_type
        wall = self.wall


        if corner_type == 'L':
            x1_L, y1_L, x2_L, y2_L = wall
        if corner_type == 'T':
            x1_T, y1_T, x2_T, y2_T, x5_T, y5_T = wall
            x1_L, y1_L, x2_L, y2_L = 0, 0, 0, 0

            if y1_T == y2_T:  ##### 正的T型转角
                if x0 < min(x1_T, x2_T):  # 雷达在左侧
                    x1_L = max(x1_T, x2_T)  # 外侧
                    y1_L = y5_T
                    x2_L = min(x1_T, x2_T)  # 内侧
                    y2_L = y1_T  # y1或者y2
                elif min(x1_T, x2_T) <= x0 < max(x1_T, x2_T):  # 雷达在中间
                    #######################################################
                    ####### 雷达在中间的时候需要额外根据target位置决定内、外侧的墙点
                    #######################################################
                    if x3 < min(x1_T, x2_T):  # target在墙角左侧
                        x1_L = max(x1_T, x2_T)  # 外侧
                        y1_L = y5_T
                        x2_L = min(x1_T, x2_T)  # 内侧
                        y2_L = y1_T  # y1或者y2
                    elif x3 > max(x1_T, x2_T):  # target在墙角右侧
                        x1_L = min(x1_T, x2_T)  # 外侧
                        y1_L = y5_T
                        x2_L = max(x1_T, x2_T)  # 内侧
                        y2_L = y1_T  # y1或者y2
                    #### 不考虑中间target的位置
                else:  # 雷达在右侧
                    x1_L = min(x1_T, x2_T)  # 外侧
                    y1_L = y5_T
                    x2_L = max(x1_T, x2_T)  # 内侧
                    y2_L = y1_T  # y1或者y2
            elif x1_T == x2_T:  # 左/右转90度的T型转角
                if y0 < min(y1_T, y2_T):  # 雷达在下侧
                    x1_L = x5_T  # 外侧
                    y1_L = max(y1_T, y2_T)
                    x2_L = x1_T  # 内侧 x1/x2
                    y2_L = min(y1_T, y2_T)
                elif min(y1_T, y2_T) <= y0 < max(y1_T, y2_T):  ############ 雷达在中间
                    #######################################################
                    ####### 雷达在中间的时候需要额外根据target位置决定内、外侧的墙点
                    #######################################################
                    if y3 > max(y1_T, y2_T):  # target在上方
                        x1_L = x5_T  # 外侧
                        y1_L = min(y1_T, y2_T)
                        x2_L = x1_T  # 内侧 x1/x2
                        y2_L = max(y1_T, y2_T)
                    elif y3 < min(y1_T, y2_T):
                        x1_L = x5_T  # 外侧
                        y1_L = max(y1_T, y2_T)
                        x2_L = x1_T  # 内侧 x1/x2
                        y2_L = min(y1_T, y2_T)

                    # 不考虑在中间的target(在可视范围内)
                else:  # 雷达在上侧
                    x1_L = x5_T  # 外侧
                    y1_L = min(y1_T, y2_T)
                    x2_L = x1_T  # 内侧 x1/x2
                    y2_L = max(y1_T, y2_T)
        elif corner_type == 'X':
            x1_T, y1_T, x2_T, y2_T, x5_T, y5_T, x6_T, y6_T = wall
            x1_L, y1_L, x2_L, y2_L = 0, 0, 0, 0
            p1, p2, p5, p6 = [x1_T, y1_T], [x2_T, y2_T], [x5_T, y5_T], [x6_T, y6_T]

            # 左上，右上，左下，右下 = 0,1,2,3
            wall_points = []
            # 判断4个wall坐标的相对位置(可优化)
            if x1_T == x2_T:  # x5=x6
                if y1_T > y2_T:
                    if x1_T > x5_T:
                        if y5_T > y6_T:
                            wall_points.extend([p5, p1, p6, p2])
                        else:
                            wall_points.extend([p6, p1, p5, p2])
                    else:
                        if y5_T > y6_T:
                            wall_points.extend([p1, p5, p2, p6])
                        else:
                            wall_points.extend([p5, p1, p6, p2])
                else:  # (y1_T <= y2_T)
                    if x1_T > x5_T:
                        if y5_T > y6_T:
                            wall_points.extend([p5, p2, p6, p1])
                        else:
                            wall_points.extend([p6, p2, p5, p1])
                    else:
                        if y5_T > y6_T:
                            wall_points.extend([p2, p5, p1, p6])
                        else:
                            wall_points.extend([p5, p2, p6, p1])
            elif x1_T == x5_T:  # x2 = x6
                if y1_T > y5_T:
                    if x1_T > x6_T:
                        if y2_T > y6_T:
                            wall_points.extend([p2, p1, p6, p5])
                        else:
                            wall_points.extend([p6, p1, p2, p5])
                    else:
                        if y2_T > y6_T:
                            wall_points.extend([p1, p2, p5, p6])
                        else:
                            wall_points.extend([p1, p6, p5, p2])
                else:  # y1_T <= y5_T
                    if x1_T > x6_T:
                        if y2_T > y6_T:
                            wall_points.extend([p2, p5, p6, p1])
                        else:
                            wall_points.extend([p6, p5, p2, p1])
                    else:
                        if y2_T > y6_T:
                            wall_points.extend([p5, p2, p1, p6])
                        else:
                            wall_points.extend([p5, p6, p1, p2])
            elif x1_T == x6_T:  # x2 = x5
                if y1_T > y6_T:
                    if x1_T > x2_T:
                        if y2_T > y5_T:
                            wall_points.extend([p2, p1, p5, p6])
                        else:
                            wall_points.extend([p5, p1, p2, p6])
                    else:
                        if y2_T > y5_T:
                            wall_points.extend([p1, p2, p6, p5])
                        else:
                            wall_points.extend([p1, p5, p6, p2])
                else:
                    if x1_T > x2_T:
                        if y2_T > y5_T:
                            wall_points.extend([p2, p6, p5, p1])
                        else:
                            wall_points.extend([p5, p6, p2, p1])
                    else:
                        if y2_T > y5_T:
                            wall_points.extend([p6, p2, p1, p5])
                        else:
                            wall_points.extend([p6, p5, p1, p2])
            # 计算对应L型求解方式的参数和约束面
            if x0 < wall_points[0][0]:  # 雷达在左边
                if y3 > wall_points[0][1]:  # target在上方
                    x1_L, y1_L = wall_points[3]  # 外侧
                    x2_L, y2_L = wall_points[0]  # 内侧
                elif y3 < wall_points[2][1]:  # target在下方
                    x1_L, y1_L = wall_points[1]  # 外侧
                    x2_L, y2_L = wall_points[2]  # 内侧
            elif x0 > wall_points[1][0]:  # 雷达在右边
                if y3 > wall_points[0][1]:  # target在上方
                    x1_L, y1_L = wall_points[2]  # 外侧
                    x2_L, y2_L = wall_points[1]  # 内侧
                elif y3 < wall_points[2][1]:  # target在下方
                    x1_L, y1_L = wall_points[0]  # 外侧
                    x2_L, y2_L = wall_points[3]  # 内侧
            elif y0 > wall_points[0][1]:  # 雷达在上方
                if y3 > wall_points[0][1]:  # target在左方
                    x1_L, y1_L = wall_points[3]  # 外侧
                    x2_L, y2_L = wall_points[0]  # 内侧
                elif y3 < wall_points[2][1]:  # target在右方
                    x1_L, y1_L = wall_points[2]  # 外侧
                    x2_L, y2_L = wall_points[1]  # 内侧
            elif y0 < wall_points[2][1]:  # 雷达在下方
                if y3 > wall_points[0][1]:  # target在左方
                    x1_L, y1_L = wall_points[1]  # 外侧
                    x2_L, y2_L = wall_points[2]  # 内侧
                elif y3 < wall_points[2][1]:  # target在右方
                    x1_L, y1_L = wall_points[3]  # 外侧
                    x2_L, y2_L = wall_points[0]  # 内侧

        return [x1_L, y1_L, x2_L, y2_L]  # 输出等价的墙内侧外侧坐标

    def nlos_judgement(self):

        # 判断各box是否能够被毫米波雷达检测到;
        # output: detected_idx_list[idx1, idx2, ..., idxn](被检测到的box的index的列表)

        detected_idx_list = []
        box_list = self.box_list
        sample_points_num = self.sample_points_num
        radar_pos = self.radar_pos

        for idx in range(len(box_list)):
            points = sample_points(box_list[idx], sample_points_num)
            detected = 0
            for i in range(sample_points_num):
                line_segments = self.calculation_of_reflection(points[i])
                for line1, line2 in zip(line_segments[::2], line_segments[1::2]):
                    line_radar = line1 if (line1[2] == radar_pos[0] or line1[3] == radar_pos[0]) else line2
                    line_obj = line2 if (line1[2] == radar_pos[0] or line1[3] == radar_pos[0]) else line1
                    in_range = self.check_in_range(line_radar)
                    if not in_range:
                        continue
                    wall_ = self.get_wall_position(points[i])
                    intersection_with_wall = self.check_intersection_with_wall(line_radar, wall_) \
                                             or self.check_intersection_with_wall(line_obj, wall_)
                    if intersection_with_wall:
                        continue
                    intersection_with_box = 0
                    for j in range(len(box_list)):
                        if j != idx:
                            if self.check_intersection_with_box(line_radar, box_list[j]) \
                                    or self.check_intersection_with_box(line_obj, box_list[j]):
                                intersection_with_box = 1
                        else:
                            if self.check_intersection_with_box(line_radar, box_list[j]):
                                intersection_with_box = 1
                        if intersection_with_box:
                            break
                    if in_range and not intersection_with_wall and not intersection_with_box:
                        detected = 1
            if detected:
                detected_idx_list.append(1)
            else:
                detected_idx_list.append(0)
        return detected_idx_list

    def get_res(self):

        # 调用nlos_judgement()函数，得到被检测到的box的index的列表，并据此输出检测结果
        # output: radar_res[[x1,y1,vx1,vy1], [x2,y2,vx2,vy2], ..., [xn,yn,vxn,vyn]](被检测到的box的位置和速度的列表)

        detected_idx_list = self.nlos_judgement()
        radar_res = []
        for i in range(len(detected_idx_list)):
            if detected_idx_list[i] == 1:
                radar_res.append(self.box_list[i][:2] + self.vel_list[i])
        return detected_idx_list, radar_res
