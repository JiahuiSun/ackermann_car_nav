# coding: utf-8
import argparse
import os
import torch as th
from torch.distributions import Independent, Normal, Beta
import yaml
import time
import math
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
from env import CornerEnv
from ppo import PPO
from network import Actor, Critic, ActorBeta, RecurrentActor, RecurrentCritic


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

    if config['use_rnn']:
        actor = RecurrentActor(obs_dim, act_dim).to(config['device'])
        critic = RecurrentCritic(obs_dim, 1).to(config['device'])
    else:
        if config['dist_beta']:
            actor = ActorBeta(obs_dim, act_dim).to(config['device'])
        else:
            actor = Actor(obs_dim, act_dim).to(config['device'])
        critic = Critic(obs_dim, 1).to(config['device'])

    def dist_fn(para1, para2):
        if config['dist_beta']:
            return Independent(Beta(para1, para2), 1)
        else:
            return Independent(Normal(para1, para2), 1)
        
    if config['load']:
        actor.load_state_dict(th.load(config['actor_path'], map_location=config['device']))
        critic.load_state_dict(th.load(config['critic_path'], map_location=config['device']))

    actor_optim = th.optim.Adam(actor.parameters(), lr=config['lr_actor'], eps=1e-5)
    critic_optim = th.optim.Adam(critic.parameters(), lr=config['lr_critic'], eps=1e-5)

    agent = PPO(
        env=env,
        actor=actor,
        critic=critic,
        dist_fn=dist_fn,
        actor_optim=actor_optim,
        critic_optim=critic_optim,
        config=config,
        writer=writer
    )
    agent.learn()

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Corner Navigation')
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--world', type=str, default='C:/webots/corner_nav/worlds/indoor_corner_nav.wbt',
                        help='world file path')
    parser.add_argument('--fov', type=float, default=math.pi , help='field of view')
    parser.add_argument('--sample_points_num', type=int, default=10, help='number of sample points')
    parser.add_argument('--config', type=str, default='scripts/train/conf/ppo.yaml')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Read config, set seed and logger
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
    config['device'] = f'cuda:{args.cuda}' if args.cuda >= 0 else 'cpu'
    config['log_dir'] = log_dir
    with open(os.path.join(log_dir, "config.yaml"), 'w') as f:
        yaml.dump(config, f)
    
    main(config, writer)
