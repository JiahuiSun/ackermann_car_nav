import rosbag
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
import sys
from PIL import Image
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

def image_visualization(grid):
    x = y = 256
    im = Image.new("RGB", (x, y))
    for i in range(x):
        for j in range(y):
            im.putpixel((255 - j, 255 - i), (int(grid[0][i][j]), int(grid[1][i][j]), int(grid[2][i][j])))
    im.show()
    im.save('test_obs.png')


def bresenham(grid, x0, y0, x1, y1, flag):
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    err = dx - dy

    while True:
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

def intersection(x1, y1, x2, y2, x3, y3, x4, y4):
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
    if (x < min(x1, x2) or x > max(x1, x2) or y < min(y1, y2) or y > max(y1, y2) or
            x < min(x3, x4) or x > max(x3, x4) or y < min(y3, y4) or y > max(y3, y4)):
        return None

    p = [x, y]
    return p

def find_intersection_point(x_min, x_max, y_min, y_max, point):
    x = point[0]
    y = point[1]
    p1 = intersection(x_min, y_min, x_min, y_max, 0, 0, x, y)
    p2 = intersection(x_min, y_min, x_max, y_min, 0, 0, x, y)
    p3 = intersection(x_max, y_min, x_max, y_max, 0, 0, x, y)
    p4 = intersection(x_min, y_max, x_max, y_max, 0, 0, x, y)
    if p1 != None:
        return p1
    elif p2 != None:
        return p2
    elif p3 != None:
        return p3
    elif p4 != None:
        return p4
    else:
        return None

def map_lidar_pointcloud(grid, points, n, l):

    grey = [105, 105, 105]
    grey = np.array(grey)
    d = l / n
    x_min, x_max = -l / 8, 7 * l / 8
    y_min, y_max = -l / 2, l / 2
    cx, cy = n // 8, n // 2
    if points == None:
        return grid
    for point in points:
        point = -np.array(point)
        if x_min <= point[0] < x_max and y_min <= point[1] < y_max:
            x = point[0]
            y = point[1]
        else:
            x, y = find_intersection_point(x_min, x_max, y_min, y_max, point)
        i = int((x - x_min) // d)
        j = int((y - y_min) // d)
        
        if x == x_max:
            i = n - 1
        if y == y_max:
            j = n - 1
        
        if np.array_equal(grid[i][j], grey) == False:
            bresenham(grid, cx, cy, i, j, grey)
            # grid[i][j] = 1   
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


if __name__ == '__main__':
    bag_file = 'test_obs_2023-10-14-17-00-06.bag'
    bag = rosbag.Bag(bag_file, 'r')
    info = bag.get_type_and_topic_info()
    # print(info)
    bag_data = bag.read_messages('/scan')
    # for topic, msg, t in bag_data:
    #     points = list(point_cloud2.read_points_list(msg, field_names=("x", "y", "z")))
    #     # print(points)
    #     bev = np.zeros((256, 256, 3))
    #     bev = map_lidar_pointcloud(bev, points, 256, 8)
    #     bev = map_self(bev, 256, 8, [0.39, 0.24])
    #     obs = np.transpose(bev, (2, 0, 1))
    #     image_visualization(obs)
    #     # save image


    for topic, msg, t in bag_data:
        # print(msg)
        angle = 0
        angle_increment = msg.angle_increment
        points = []
        for i in range(len(msg.ranges)):
            r = msg.ranges[i]
            if r == float('inf'):
                if msg.ranges[i-1] != float('inf'):
                    r = msg.ranges[i-1]
                elif msg.ranges[(i+1)%len(msg.ranges)] != float('inf'):
                    r = msg.ranges[(i+1)%len(msg.ranges)]
                else:
                    r = 10
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            points.append([-x, -y])
            angle += angle_increment

        bev = np.zeros((256, 256, 3))
        bev = map_lidar_pointcloud(bev, points, 256, 8)
        bev = map_self(bev, 256, 8, [0.39, 0.24])
        obs = np.transpose(bev, (2, 0, 1))
        image_visualization(obs)

        break


