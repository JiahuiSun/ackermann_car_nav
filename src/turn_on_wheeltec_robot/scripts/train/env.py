import gym
import torch
from gym import spaces
from multiprocessing import Process
from scipy.spatial.transform import Rotation as R
import time
from tf import transformations
import globalvar as gl
from radar import MyRadar
from controller import *
from vehicle import *

from utils import *
from ros_utils import *

torch.set_printoptions(threshold=np.inf)

class CornerEnv(gym.Env):
    def __init__(self, config, timestep=96):
        # 初始化环境
        # world_file: world文件路径
        # fov: 毫米波雷达的视场角大小(rad)
        # sample_points_num: 每个box上采样点的数量
        super().__init__()

        self.left_steer = None
        self.right_steer = None
        self.touch_sensor = None
        self.supervisor = None
        self.front_left_motor = None
        self.front_right_motor = None
        self.rear_left_motor = None
        self.rear_right_motor = None
        self.distance_sensors = None
        self.lidar = None
        self.radar = None
        self.last_obs = None
        self.obs = None
        self.min_distance = None

        self.conf = config
        self.world_file = self.conf['world']
        self.fov = self.conf['fov']
        self.offset_angle = 0
        self.sample_points_num = self.conf['sample_points_num']
        self.navigation_goal = None

        self.wheel_base = 0.324
        self.wheel_tread = 0.3
        self.wheel_radius = 0.05
        self.min_turning_radius = 0.77
        self.linear_velocity_ratio = 0.6944
        self.robot_size = [0.39, 0.24]

        self.max_v = 1
        self.max_w = math.pi/5

        self.obs_dim = self.conf['map_side_grids']
        self.act_dim = 2
        self.n_agent = 1
        self.action_space = spaces.Box(low=np.array([0, -math.pi/5]), high=np.array([1.5, math.pi/5]), dtype=np.float32)

        # open_args = ['webots', '--batch']
        # open_args.append('--mode=fast')
        # open_args.append('--no-rendering')
        # open_args.append(self.world_file)

        # open webots
        # subprocess.Popen(open_args, stdout=subprocess.PIPE)
        # time.sleep(2)

        self.done = False
        self.info_dict = {
            'delta_distance': 0.0,
            'collision': False
        }
        self.supervisor = Supervisor()
        self.driver = Driver()

        self.timestep = timestep

        self.init_device()

        self.robot_node = self.supervisor.getFromDef("base_link")
        self.pedestrian_size = [0.7, 0.7]
        self.static_obstacle_list = []
        self.moving_obstacle_last_list = []
        self.moving_obstacle_node = [self.supervisor.getFromDef("p0"), self.supervisor.getFromDef("p1"), self.supervisor.getFromDef("p2")]
        

    def init_device(self):

        # 初始化接触传感器

        self.touch_sensor = self.supervisor.getDevice("touch_sensor")
        self.touch_sensor.enable(self.timestep)

        # 初始化激光雷达

        self.lidar = self.supervisor.getDevice("lidar")
        self.lidar.enable(self.timestep)
        self.lidar.enablePointCloud()

        # 初始化毫米波雷达

        self.radar = MyRadar(self.fov, self.sample_points_num)

    def step(self, action):

        # input: action 
        # output: observation, reward, done, info
        self.info_dict = {
            'delta_distance': 0.0,
            'collision': False
        }
        # self.apply_action(action)
        # st1 = time.time()
        # obs = self.get_observations()
        # end1 = time.time()
        # print(f"obs time cost: {end1 - st1}s")

        # 执行动作
        # action: [linear_velocity, angular_velocity]
        for i in range(1):
            self.apply_action(action)
            self.supervisor.step(self.timestep)
            # 获取观测
            obs = self.get_observations()

            reward = self.get_reward(action)
            done = self.is_done()
            info = self.get_info()

        return obs, reward, done, info

    def apply_action(self, action):

        # 执行动作

        linear_velocity = action[0]
        angular_velocity = action[1]
        turning_radius = 0
        if angular_velocity != 0:
            turning_radius = linear_velocity / angular_velocity
        if 0 < turning_radius < self.min_turning_radius:
            turning_radius = self.min_turning_radius
        if -self.min_turning_radius < turning_radius < 0:
            turning_radius = -self.min_turning_radius
        angle = 0 if angular_velocity == 0 else math.atan(self.wheel_base / turning_radius)

        self.driver.setSteeringAngle(action[1])
        self.driver.setCruisingSpeed(action[0])

        

    def reset(self):
        # 重置环境

        self.supervisor.simulationResetPhysics()
        self.init_device()
        self.last_obs = None
        self.obs = None
        self.min_distance = None

        self.unsafe = False
        self.collision = False
        self.done = False

        # start_flag_pos_field = self.start_flag_node.getField("translation")
        # start_flag_rot_field = self.start_flag_node.getField("rotation")

        # 重设小车的轮子的速度和角度
        self.driver.setSteeringAngle(0)
        self.driver.setCruisingSpeed(0)

        start_pos = [10, 2, 0]
        # start_pos[0] += 0.5
        start_rot = [0, 0, 1, -math.pi / 4]
        # start_rot[3] -= math.pi / 2

        robot_pos_field = self.robot_node.getField("translation")
        robot_rot_field = self.robot_node.getField("rotation")
        robot_pos_field.setSFVec3f(start_pos)
        robot_rot_field.setSFRotation(start_rot)

        self.supervisor.step(self.timestep)
        default_obs = self.get_default_observation()

        return default_obs

    def render(self):
        pass

    def close(self):

        # 关闭环境
        # self.supervisor.simulationQuit(0)
        pass

    def get_default_observation(self):

        # 获取初始观测
        moving_obstacle_trans_field_list = []
        for node in self.moving_obstacle_node:
            moving_obstacle_trans_field_list.append(node.getField("translation"))
        for field in moving_obstacle_trans_field_list:
            self.moving_obstacle_last_list.append(field.getSFVec3f()[:2] + self.pedestrian_size)
        return self.get_observations(default=True)

    def get_observations(self, default = False):

        # 获取观测

        self.last_obs = self.obs
        lidar_point_cloud = self.lidar.getPointCloud()
        map_len = self.conf['map_len']
        map_side_block_num = self.obs_dim

        lidar_res = []
        p = 0
        for point in lidar_point_cloud:
            p += 1
            x, y = list(vars(point).values())[:2]
            if math.isinf(x) or math.isinf(y) or math.isnan(x) or math.isnan(y):
                x, y = 0, 0
            lidar_res.append([x, y])

        robot_pos_list, robot_vel_list, robot_rot_list, wall, absent_wall, \
            corner_type, obstacle_list, obstacle_vel_list, local_obstacle_list = self.get_env_around()

        robot_rot = robot_rot_list[3] if robot_rot_list[2] >= 0 else -robot_rot_list[3]
        # 初始化地图
        if not default:
            self.collision_detection()

        self.radar.set_radar(radar_pos=robot_pos_list,
                             radar_rot=robot_rot,
                             wall=wall,
                             absent_wall=absent_wall,
                             box_list=obstacle_list,
                             vel_list=obstacle_vel_list,
                             corner_type=corner_type)

        radar_detected_list, radar_res = self.radar.get_res()

        pos_map = np.full((map_side_block_num, map_side_block_num), -1)
        pos_map = map_lidar_pointcloud(pos_map, lidar_res, map_side_block_num, map_len)
        pos_map = map_radar_res(pos_map, robot_rot, map_side_block_num, map_len, radar_res, robot_pos_list)
        # pos_map = map_wall_pos(pos_map, map_side_block_num, map_len, wall)
        # pos_map = map_robot_pos(pos_map, map_side_block_num, map_len, robot_pos_list, robot_rot, self.robot_size)

        # visualize_grid_map(pos_map)

        self.obs = {
            'pos': np.array(robot_pos_list, dtype=np.float32).reshape(1, -1),
            'vel': np.array(robot_vel_list[:2], dtype=np.float32).reshape(1, -1),
            'rot': np.array(robot_rot, dtype=np.float32).reshape(1, -1),
            'radar': np.array(radar_res, dtype=np.float32).reshape(1, -1),
            # 'map': np.array(pos_map, dtype=np.float32)
        }
        #
        # obs = np.concatenate((self.obs['pos'], self.obs['vel'], self.obs['rot'], self.obs['goal'], self.obs['map']),
        #                      axis=1)

        return pos_map.squeeze()

    def collision_detection(self):
        # rostopic_touch_sensor_get()
        touch_sensor_value = self.touch_sensor.getValue()
        if touch_sensor_value == True:
            self.collision = True
            self.info_dict['collision'] = True
            self.done = True

    def get_env_around(self):
        # 获取环境信息，作为毫米波雷达模块的输入

        robot_trans_field = self.robot_node.getField("translation")
        robot_pos_list = robot_trans_field.getSFVec3f()[:2]
        robot_vel_list = self.robot_node.getVelocity()
        robot_rot_field = self.robot_node.getField("rotation")
        robot_rot_list = robot_rot_field.getSFRotation()

        moving_obstacle_trans_field_list = []
        for node in self.moving_obstacle_node:
            moving_obstacle_trans_field_list.append(node.getField("translation"))

        static_obstacle_list = self.static_obstacle_list
        moving_obstacle_list = []
        for field in moving_obstacle_trans_field_list:
            moving_obstacle_list.append(field.getSFVec3f()[:2] + self.pedestrian_size)
        moving_obstacle_vel_list = velocity_cal(moving_obstacle_list, self.moving_obstacle_last_list,
                                                self.timestep)

        static_obstacle_vel_list = []
        for i in self.static_obstacle_list:
            static_obstacle_vel_list.append([0, 0])

        wall = [23.999, 0, 20, 4]
        absent_wall = [0]
        corner_type = 'L'
        static_obstacle_idx = []
        moving_obstacle_idx = [0, 1, 2]
        # wall, absent_wall, corner_type, static_obstacle_idx, moving_obstacle_idx \
        #     = area_corner_obstacles_getting(robot_pos_list, robot_vel_list)
        local_static_obstacle_idx = []
        local_moving_obstacle_idx = []
        # local_static_obstacle_idx, local_moving_obstacle_idx = area_local_obstacles_getting()

        obstacle_list = [static_obstacle_list[i] for i in static_obstacle_idx] + \
                        [moving_obstacle_list[i] for i in moving_obstacle_idx]

        local_obstacle_list = [static_obstacle_list[i] for i in local_static_obstacle_idx] + \
                              [moving_obstacle_list[i] for i in local_moving_obstacle_idx]

        obstacle_vel_list = [static_obstacle_vel_list[i] for i in static_obstacle_idx] + \
                            [moving_obstacle_vel_list[i] for i in moving_obstacle_idx]

        self.moving_obstacle_last_list = moving_obstacle_list

        return robot_pos_list, robot_vel_list, robot_rot_list, wall, absent_wall, \
            corner_type, obstacle_list, obstacle_vel_list, local_obstacle_list

    def get_reward(self, action):
        # d_g = self.conf['reward_distance_goal']
        # d_m = self.conf['reward_distance_move']
        #
        # w_g = self.conf['reward_weight_goal']
        # w_p = self.conf['reward_weight_proximity']
        # w_m = self.conf['reward_weight_move']
        # w_c = self.conf['reward_weight_collision']
        # w_s = -0.05
        # w_r = self.conf['reward_weight_turning']
        # w_w = self.conf['reward_weight_w']

        goal = self.navigation_goal
        robot_pos = self.obs['pos']
        robot_rot = self.obs['rot']
        robot_last_pos = self.last_obs['pos']

        r_g = 0
        # if np.linalg.norm(np.array(robot_pos) - np.array(goal)) <= d_g:
        #     r_g = w_g
        #     self.done = True
        # self.info_dict['delta_distance'] = np.linalg.norm(np.array(robot_pos) - np.array(robot_last_pos))
        # r_p = w_p * -(np.linalg.norm(np.array(robot_pos) - np.array(goal)) - np.linalg.norm(
        #     np.array(robot_last_pos) - np.array(goal)))
        # r_m = w_m if np.linalg.norm(np.array(robot_pos) - np.array(robot_last_pos)) <= d_m else 0
        # # r_m = 0
        # r_c = w_c if self.collision else 0
        # # r_s = w_s * (self.unsafe_distance - self.min_distance) if self.unsafe else 0
        # r_r = w_r * (action[0] - w_w * abs(action[1]))
        # r_r = 0

        w_l = 30
        w_c = -50
        w_x = 0.05
        w_y = 0.05
        w_r = 0.4
        w_m = -0.2
        r1 = 0
        if self.obs['radar'].shape[1] != 0:
            for i in range(len(self.obs['radar'])):
                r1 += w_x * (24 - self.obs['radar'][i][0]) + w_y * (self.obs['radar'][i][1] - 4)
        r2 = 0
        r3 = w_l * (robot_pos[0][0] - robot_last_pos[0][0])
        r4 = 0
        if robot_pos[0][0] >= 20:
            r4 = 100
            self.done = True
        r5 = -0.3 * abs(robot_rot[0][0])
        r_c = w_c if self.collision else 0
        r_r = w_r * (action[0] - 2 * abs(action[1]))
        r_back = -1 if action[0] < 0 else 0
        r_m = w_m if np.linalg.norm(np.array(robot_pos) - np.array(robot_last_pos)) <= 0.06 else 0
        r = r1 + r2 + r3 + r4 + r5 + r_c + r_m + r_r + r_back
        return r

    def is_done(self):
        return self.done

    def get_info(self):
        return self.info_dict
