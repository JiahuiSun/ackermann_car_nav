import rospy
import message_filters
from sensor_msgs.msg import LaserScan
from sensor_msgs import point_cloud2
from geometry_msgs.msg import Twist

import numpy as np
import torch as th
from torch.distributions import Independent, Normal
from train.network import Actor
import train.utils as utils
import sys, select, termios, tty
import argparse

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

        self.laser_sub = rospy.Subscriber('/scan', LaserScan, self.callback_pointcloud, queue_size=1000)
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
        assert isinstance(msg, LaserScan)

        self.key = self.getKey()
        if self.key == 's':
            self.stop = 1 - self.stop
        if self.key == ' ':
            self.stop = True
        
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
        bev = utils.map_lidar_pointcloud(bev, points, 256, 8)
        bev = utils.map_self(bev, 256, 8, [0.39, 0.24])
        obs = np.transpose(bev, (2, 0, 1))
        obs = obs.squeeze()
        # utils.image_visualization(obs)
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
            # print(self.move_cmd)
            rate.sleep()


if __name__ == '__main__':
    paser = argparse.ArgumentParser()
    paser.add_argument('-p', '--policy', type=str, default='left')


    args = paser.parse_args()
    
    config = {
        'device': 'cuda' if th.cuda.is_available() else 'cpu',
        'actor_path': 'scripts/left.pth',
        'map_side_block_num': 256,
        'map_len': 8,
        'act_dim': 2,
        'wheel_base': 0.324,
        'linear_velocity_ratio': 0.6944,
        'timestep': 384,
        'max_v': 0.4,
        'max_w': math.pi/5,
    }

    if args.policy == 'left':
        config['actor_path'] = 'scripts/left.pth'
    elif args.policy == 'right':
        config['actor_path'] = 'scripts/right.pth'

    try:
        deploy = deployment(config)
        deploy.spin()
    except rospy.ROSInterruptException:
        pass
