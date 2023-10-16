from tqdm import tqdm
from os.path import join as pjoin
import torch
import torch.nn as nn
import torch.nn.functional as F
from numba import njit
import numpy as np


# torch.set_printoptions(threshold=np.inf)

@njit
def _gae_return(
        v_s: np.ndarray,
        v_s_: np.ndarray,
        rew: np.ndarray,
        end_flag: np.ndarray,
        gamma: float,
        gae_lambda: float,
) -> np.ndarray:
    returns = np.zeros(rew.shape)
    delta = rew + v_s_ * gamma - v_s
    m = (1.0 - end_flag) * (gamma * gae_lambda)
    gae = 0.0
    for i in range(len(rew) - 1, -1, -1):
        gae = delta[i] + m[i] * gae
        returns[i] = gae
    return returns


@njit
def _discount_cumsum(rew, end_flag, gamma):
    returns = np.zeros(rew.shape)
    m = (1.0 - end_flag) * gamma
    gae = 0.0
    for i in range(len(rew) - 1, -1, -1):
        gae = rew[i] + m[i] * gae
        returns[i] = gae
    return returns


def kl_divergence(old_probs, probs):
    eps = 1e-7
    old_logits = np.log(np.clip(old_probs, a_min=eps, a_max=1 - eps))
    logits = np.log(np.clip(probs, a_min=eps, a_max=1 - eps))
    p_log_p_q = old_probs * (old_logits - logits)
    return p_log_p_q.sum(-1).mean()


class PPO:
    def __init__(
            self,
            env,
            actor_critic,
            shared_optim,
            actor_optim,
            critic_optim,
            dist_fn,
            config,
            writer,
            device,
            clip=0.2
    ):
        self.env = env
        self.actor_critic = actor_critic
        self.actor_optim = actor_optim
        self.critic_optim = critic_optim
        self.shared_optim = shared_optim
        self.dist_fn = dist_fn
        self.config = config
        self.writer = writer
        self.device = device
        self.clip = clip
    
    def learn(self):
        for i in range(self.config['num_epochs']):
            buffer, total_reward = self.rollout()
            print('epoch:', i, "total_reward:", total_reward)
            self.update(buffer)
    
    def rollout(self):
        buffer = {
            'obs': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'dones': [],
            'log_probs': [],
        }
        
        obs = self.env.reset()
        # obs = torch.tensor(obs, dtype=torch.float32).view(-1, 1)
        # print(obs)
        done = False
        total_reward = 0
        ep_len = 0
        last_action = [0, 0]
        while not done:
            buffer['obs'].append(obs)
            with torch.no_grad():
                value, action, log_prob = self.actor_critic.get_action(obs.unsqueeze(0))
            action = self.map_action(action, last_action)
            # print(action)
            next_obs, reward, done, _ = self.env.step(action)
            
            buffer['actions'].append(action)
            buffer['rewards'].append(torch.FloatTensor([reward]).to(self.device))
            buffer['values'].append(value)
            buffer['dones'].append(torch.FloatTensor([done]).to(self.device))
            buffer['log_probs'].append(log_prob)
            
            total_reward += reward
            ep_len += 1
            
            obs = next_obs
            last_action = action
            
            if ep_len == self.config['max_ep_len'] or done:
                if ep_len == self.config['max_ep_len']:
                    last_value, _, _ = self.actor_critic.get_action(obs.unsqueeze(0))
                    buffer['values'].append(last_value)
                else:
                    buffer['values'].append(torch.FloatTensor([[0]]).to(self.device))
                break
        
        buffer['advantages'] = self.compute_gae(buffer)
        
        # for key in buffer:
        #     buffer[key] = np.array(buffer[key], dtype=np.float32)
        
        return buffer, total_reward
    
    def compute_gae(self, buffer):
        values = buffer['values']
        dones = buffer['dones']
        rewards = buffer['rewards']
        
        gae = 0
        advantages = []
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.config['gamma'] * values[i + 1] * (1 - dones[i]) - values[i]
            gae = delta + self.config['gamma'] * self.config['gae_lambda'] * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        
        # advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        return advantages
    
    def update(self, buffer):
        obs = torch.stack(buffer['obs'], dim=0).to(self.device)
        actions = torch.FloatTensor(np.array(buffer['actions'])).to(self.device)
        old_log_probs = torch.stack(buffer['log_probs'], dim=0).unsqueeze(1).to(self.device)
        values = torch.stack(buffer['values'], dim=0).unsqueeze(1).to(self.device)
        rewards = torch.stack(buffer['rewards'], dim=0).unsqueeze(1).to(self.device)
        dones = torch.stack(buffer['dones'], dim=0).unsqueeze(1).to(self.device)
        advantages = torch.stack(buffer['advantages'], dim=0).unsqueeze(1).to(self.device)
        
        # Compute advantages and target values
        returns = self.compute_returns(buffer['rewards'], buffer['dones'], buffer['values'][-1])
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        target_values = rewards + torch.stack(returns, dim=0)
        
        # Update the actor-critic
        for _ in range(self.config['repeat_per_epoch']):
            # Shuffle the batch
            indices = np.random.permutation(len(obs))
            obs, actions, old_log_probs, values, advantages, target_values = \
                obs[indices], actions[indices], old_log_probs[indices], values[indices], advantages[indices], \
                    target_values[indices]
            
            # Compute the new log probabilities and ratio
            new_values, new_log_probs, entropy = self.actor_critic.evaluate(obs, actions)
            ratio = torch.exp(new_log_probs - old_log_probs)
            torch.cuda.empty_cache()
            # Compute the actor and critic losses
            actor_loss = -torch.min(ratio * advantages, torch.clamp(ratio, 1 - self.config['clip_param'],
                                                                    1 + self.config['clip_param']) * advantages).mean()
            critic_loss = (values - target_values).pow(2).mean()
            
            # Compute the overall loss and update the parameters
            loss = actor_loss + self.config['value_loss_coef'] * critic_loss - self.config[
                'entropy_coef'] * entropy.mean()
            
            self.shared_optim.zero_grad()
            self.actor_optim.zero_grad()
            self.critic_optim.zero_grad()
            
            loss.backward()
            
            self.shared_optim.step()
            self.actor_optim.step()
            self.critic_optim.step()
            
            # Check if the KL divergence is too high and stop the training if necessary
            if self.config['kl_stop'] and (new_log_probs - old_log_probs).mean().item() > 1.5 * self.config[
                'target_kl']:
                break
    
    def compute_returns(self, rewards, dones, last_value):
        returns = []
        running_return = last_value
        for step in reversed(range(len(rewards))):
            running_return = rewards[step] + self.config['gamma'] * running_return * (1 - dones[step])
            returns.insert(0, running_return)
        return returns
    
    def map_action(self, action, last_action):
        action = action.cpu().numpy()
        action = np.clip(action, -1.0, 1.0).squeeze()
        low, high = self.env.action_space.low, self.env.action_space.high
        action = low + (high - low) * (action + 1.0) / 2.0
        action = np.clip(action, [last_action[0] - 1, last_action[1] - 0.5], [last_action[0] + 1, last_action[1] + 0.5])
        # print(action)
        return action
