import argparse
import os
import torch as th
from torch.distributions import Independent, Normal
import yaml
import time
import math
import numpy as np
import random
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from env import CornerEnv
from ppo import PPO
from network import Actor


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)


def main(config, writer):
    env = CornerEnv(config)
    obs_dim = env.obs_dim
    act_dim = env.act_dim
    
    actor = Actor(obs_dim, act_dim).to(config['device'])
    actor.load_state_dict(th.load(config['actor_path'], map_location=config['device']))
    def dist_fn(mu, sigma):
        return Independent(Normal(mu, sigma), 1)
    
    agent = PPO(
        env=env,
        actor=actor,
        critic=None,
        dist_fn=dist_fn,
        actor_optim=None,
        critic_optim=None,
        config=config,
        writer=writer
    )
    
    success = 0
    ep_time = []
    ep_rew = []
    ep_distance = []
    success_time = []
    # success_rew = []
    success_distance = []
    for epoch in tqdm(range(config['n_epoch'])):
        buffer, total_reward, ep_len, distance, ep_collision = agent.rollout()
        ep_time.append(ep_len * 0.1)
        ep_rew.append(total_reward)
        ep_distance.append(distance)
        if not ep_collision:
            success += 1
            success_time.append(ep_len * 0.1)
            # success_rew.append(total_reward)
            success_distance.append(distance)
    
    print(
        'success: {}, avg_reward: {}, std_reward: {}, avg_time: {}, std_time: {}, '
        'avg_distance: {}, std_distance: {}, avg_sucess_time: {}, std_success_time: {},'
        ' avg_success_distance: {}, std_success_distance: {}'.format(
            success,
            np.mean(ep_rew),
            np.std(ep_rew),
            np.mean(ep_time),
            np.std(ep_time),
            np.mean(ep_distance),
            np.std(ep_distance),
            np.mean(success_time),
            np.std(success_time),
            np.mean(success_distance),
            np.std(success_distance)
        ))
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--world', type=str, default='C:/webots/corner_nav/worlds/indoor_corner_nav.wbt',
                        help='world file path')
    parser.add_argument('--fov', type=float, default=math.pi / 3, help='field of view')
    parser.add_argument('--sample_points_num', type=int, default=10, help='number of sample points')
    parser.add_argument('--config', type=str, default='scripts/train/conf/test.yaml')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--actor_path', type=str,
                        default='output_radar/ppo_agent/20231006_223910/actor_499.pth')
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    set_seed(args.seed)
    logid = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_dir = os.path.join(config['log_dir'], config['name'], logid)
    writer = SummaryWriter(log_dir)
    
    config['world'] = args.world
    config['fov'] = args.fov
    config['sample_points_num'] = args.sample_points_num
    config['seed'] = args.seed
    config['actor_path'] = args.actor_path
    config['device'] = f'cuda:{args.cuda}' if args.cuda >= 0 else 'cpu'
    config['log_dir'] = log_dir
    with open(os.path.join(log_dir, "config.yaml"), 'w') as f:
        yaml.dump(config, f)

main(config, writer)
