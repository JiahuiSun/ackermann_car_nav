import rospy
import message_filters
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from geometry_msgs.msg import Twist

import numpy as np
import torch as th
from torch.distributions import Independent, Normal
from train.network import Actor
import train.utils as utils
import sys, select, termios, tty

import math

class deployment:
    def __init__(self, config):
        rospy.init_node('policy_deployment')
        self.config = config

        self.obs_dim = config['map_side_block_num']
        self.act_dim = config['act_dim']
        self.map_side_block_num = config['map_side_block_num']
        self.map_len = config['map_len']
        self.actor = Actor(self.obs_dim, self.act_dim).to(config['device'])
        self.actor.load_state_dict(th.load(config['actor_path'], map_location=config['device']))
        self.actor.eval()

        self.laser_sub = rospy.Subscriber('/laser_point_cloud', PointCloud2, self.callback_pointcloud, queue_size=5)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=5)

        self.wheel_base = config['wheel_base']
        self.linear_velocity_ratio = config['linear_velocity_ratio']

        self.rate = 1000.0 / config['timestep']
        self.move_cmd = Twist()

        self.stop = True
        self.key = ''
    
    def getKey(self):
        property_list = termios.tcgetattr(sys.stdin)
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
        if rlist:
            key = sys.stdin.read(1)
        else:
            key = ''

        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, property_list)
        return key

    def dist_fn(self, mu, sigma):
        return Independent(Normal(mu, sigma), 1)

    def get_action(self, obs):
        obs = th.tensor(obs, dtype=th.float32, device=config['device'])
        with th.no_grad():
            para1, para2 = self.actor(obs)
            dist = self.dist_fn(para1, para2)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.cpu().numpy().squeeze(), log_prob.cpu().numpy().squeeze(), \
            para1.cpu().numpy().squeeze(), para2.cpu().numpy().squeeze()

    def map_action(self, act):
        
        action = np.clip(act, -1, 1).squeeze()
        v = action[0] * config['max_v']
        w = action[1] * config['max_w']
        return [v, w]
    
    def apply_action(self, action):
        speed = action[0]
        angle = action[1]

        linear_velocity = speed * self.linear_velocity_ratio
        turning_radius = self.wheel_base / np.tan(angle)
        angular_velocity = linear_velocity / turning_radius

        self.move_cmd.linear.x = linear_velocity
        self.move_cmd.angular.z = angular_velocity

    def callback_pointcloud(self, msg):
        assert isinstance(msg, PointCloud2)

        self.key = self.getKey()
        if self.key == 's':
            self.stop = 1 - self.stop
        if self.key == ' ':
            self.stop = True
        
        points = list(point_cloud2.read_points_list(msg, field_names=("x", "y", "z")))

        grid = np.full((self.map_side_block_num, self.map_side_block_num), -1)
        obs = utils.map_lidar_pointcloud(grid, points, self.map_side_block_num, self.map_len)
        np.save('obsnpy/obs.npy', obs)
        obs = obs.squeeze()
        action, _, _, _ = self.get_action(obs[np.newaxis,:])
        mapped_action = self.map_action(action)
        self.apply_action(mapped_action)

    def spin(self):
        rate = rospy.Rate(self.rate)
        while not rospy.is_shutdown():
            if self.stop:
                self.move_cmd.linear.x = 0
                self.move_cmd.angular.z = 0
            self.cmd_vel_pub.publish(self.move_cmd)
            #print(self.move_cmd)
            rate.sleep()


if __name__ == '__main__':
    config = {
        'device': 'cuda' if th.cuda.is_available() else 'cpu',
        'actor_path': 'scripts/actor.pth',
        'map_side_block_num': 224,
        'map_len': 18,
        'act_dim': 2,
        'wheel_base': 0.324,
        'linear_velocity_ratio': 0.6944,
        'timestep': 384,
        'max_v': 0.4,
        'max_w': math.pi/5,
    }

    try:
        deploy = deployment(config)
        deploy.spin()
    except rospy.ROSInterruptException:
        pass
