from tqdm import tqdm
from os.path import join as pjoin
import torch as th
import numpy as np
from numba import njit
from utils import *
import scipy


class RewardScaling:
    class RunningMeanStd:
        # Dynamically calculate mean and std
        def __init__(self, shape):  # shape:the dimension of input data
            self.n = 0
            self.mean = np.zeros(shape)
            self.S = np.zeros(shape)
            self.std = np.sqrt(self.S)

        def update(self, x):
            x = np.array(x)
            self.n += 1
            if self.n == 1:
                self.mean = x
                self.std = x
            else:
                old_mean = self.mean.copy()
                self.mean = old_mean + (x - old_mean) / self.n
                self.S = self.S + (x - old_mean) * (x - self.mean)
                self.std = np.sqrt(self.S / self.n)

    def __init__(self, shape, gamma):
        self.shape = shape  # reward shape=1
        self.gamma = gamma  # discount factor
        self.running_ms = self.RunningMeanStd(shape=self.shape)
        self.R = np.zeros(self.shape)

    def __call__(self, x):
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-8)  # Only divided std
        return x

    def reset(self):  # When an episode is done,we should reset 'self.R'
        self.R = np.zeros(self.shape)


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


def gaussian_kl(mu0, log_std0, mu1, log_std1):
    """Returns average kl divergence between two batches of dists"""
    var0, var1 = np.exp(2 * log_std0), np.exp(2 * log_std1)
    pre_sum = 0.5 * (((mu1 - mu0) ** 2 + var0) / (var1 + 1e-8) - 1) + log_std1 - log_std0
    all_kls = np.sum(pre_sum, axis=1)
    return np.mean(all_kls)


def beta_kl(alpha0, beta0, alpha1, beta1):
    """Returns average kl divergence between two batches of dists"""
    sum0 = np.log(beta1) - np.log(beta0) + (alpha0 - alpha1) * (scipy.special.digamma(alpha0) - np.log(beta0))
    sum1 = (alpha1 - alpha0) * scipy.special.digamma(alpha1) + scipy.special.gammaln(alpha0) - scipy.special.gammaln(
        alpha1)
    return np.mean(sum0 + sum1)


class PPO:
    def __init__(
            self,
            env,
            actor,
            critic,
            dist_fn,
            actor_optim,
            critic_optim,
            config,
            writer
    ):
        self.env = env
        self.actor = actor
        self.critic = critic
        self.actor_optim = actor_optim
        self.critic_optim = critic_optim
        self.dist_fn = dist_fn
        self.conf = config
        self.writer = writer

        self.device = config['device']
        self.save_thre = self.conf['n_epoch'] * 0.1

        self.rewardscaling = RewardScaling(shape=1, gamma=self.conf['gamma'])

    def learn(self):
        for epoch in tqdm(range(self.conf['n_epoch'])):
            buffer, total_reward, ep_len, ep_distance, _ = self.rollout()
            print(f"Epoch {epoch}: Total reward = {total_reward}, Episode length = {ep_len}")

            ep_obs, ep_act, ep_log_prob, adv, target = self.compute_gae(buffer)

            actor_loss_list, critic_loss_list, kl_list = [], [], []
            policy_update_flag = 1
            for repeat in range(self.conf['repeat_per_epoch']):

                if self.conf['use_rnn']:
                    self.actor.hidden_reset()
                    self.critic.hidden_reset()

                value, curr_log_probs, para1, para2 = self.evaluate(ep_obs, ep_act)

                critic_loss = (target - value).pow(2).mean()

                ratios = th.exp(curr_log_probs - ep_log_prob)
                clip_ratios = th.clamp(ratios, 1 - self.conf['clip_param'], 1 + self.conf['clip_param'])
                surr1 = ratios * adv
                surr2 = clip_ratios * adv
                rew_loss = -th.min(surr1, surr2).mean()

                if self.conf['dist_beta']:
                    total_kl = beta_kl(para1, para2, buffer['para1'], buffer['para2'])
                else:
                    total_kl = gaussian_kl(para1, para2, buffer['para1'], buffer['para2'])

                self.critic_optim.zero_grad()
                critic_loss.backward()
                if self.conf['max_grad_norm']:
                    th.nn.utils.clip_grad_norm_(
                        self.critic.parameters(),
                        max_norm=self.conf['max_grad_norm']
                    )
                self.critic_optim.step()

                if policy_update_flag:
                    self.actor_optim.zero_grad()
                    rew_loss.backward()
                    if self.conf['max_grad_norm']:
                        th.nn.utils.clip_grad_norm_(
                            self.actor.parameters(),
                            max_norm=self.conf['max_grad_norm']
                        )
                    self.actor_optim.step()

                    if self.conf['kl_stop'] and total_kl > self.conf['kl_margin'] * self.conf['target_kl']:
                        policy_update_flag = 0
                        print(f'Early stopping at step {repeat} due to reaching max kl.')

                actor_loss_list.append(rew_loss.item())
                critic_loss_list.append(critic_loss.item())
                kl_list.append(total_kl)
            # log everything
            self.writer.add_scalar('loss/actor_loss', np.mean(actor_loss_list), epoch)
            self.writer.add_scalar('loss/critic_loss', np.mean(critic_loss_list), epoch)
            self.writer.add_scalar('loss/kl', np.mean(kl_list), epoch)
            self.writer.add_scalar('metric/return', total_reward, epoch)
            # save model
            if (epoch + 1) >= self.save_thre and (epoch + 1) % 10 == 0:
                th.save(self.actor.state_dict(), pjoin(self.conf['log_dir'], f'actor_{epoch}.pth'))
                th.save(self.critic.state_dict(), pjoin(self.conf['log_dir'], f'critic_{epoch}.pth'))

            if self.conf['lr_decay']:
                self.lr_decay(epoch)

    def get_action(self, obs):
        obs = th.tensor(obs, dtype=th.float32, device=self.device)
        with th.no_grad():
            para1, para2 = self.actor(obs)
            dist = self.dist_fn(para1, para2)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.cpu().numpy().squeeze(), log_prob.cpu().numpy().squeeze(), \
            para1.cpu().numpy().squeeze(), para2.cpu().numpy().squeeze()

    def map_action(self, act):
        
        action = np.clip(act, -1, 1).squeeze()
        v = action[0] * self.env.max_v
        w = action[1] * self.env.max_w
        return [v, w]
    
    def rollout(self):
        """Agent interacts with environment for one trajectory.
        """
        if self.conf['use_rnn']:
            self.actor.hidden_reset()

        buffer = {
            'obs': [],
            'act': [],
            'rew': [],
            'obs_next': [],
            'done': [],
            'log_prob': [],
            'para1': [],
            'para2': []
        }
        obs = self.env.reset()
        total_reward = 0
        ep_len = 0
        ep_distace = 0
        ep_collision = False

        self.rewardscaling.reset()

        while True:
            with th.no_grad():
                action, log_prob, para1, para2 = self.get_action(obs[np.newaxis, :])
            action_mapped = self.map_action(action)
            obs_next, reward, done, info = self.env.step(action_mapped)
            if self.conf['reward_scaling']:
                reward = self.rewardscaling(reward)[0]
            buffer['obs'].append(obs)
            buffer['act'].append(action)
            buffer['rew'].append(reward)
            buffer['obs_next'].append(obs_next)
            buffer['done'].append(done)
            buffer['log_prob'].append(log_prob)
            buffer['para1'].append(para1)
            buffer['para2'].append(para2)
            total_reward += reward
            ep_len += 1
            ep_distace += info['delta_distance']
            ep_collision = info['collision']
            if ep_len == self.conf['max_ep_len'] or done:
                break
            obs = obs_next
            # if ep_len % 20 == 0:
            #     visualize_grid_map(obs)
        for key in buffer:
            buffer[key] = np.array(buffer[key], dtype=np.float32)
        return buffer, total_reward, ep_len, ep_distace, ep_collision

    def compute_gae(self, buffer):
        """Compute adv, gae for training. Convert batch from numpy to tensor.
        """
        ep_obs = th.tensor(buffer['obs'], dtype=th.float32, device=self.device)
        ep_act = th.tensor(buffer['act'], dtype=th.float32, device=self.device)
        ep_log_prob = th.tensor(buffer['log_prob'], dtype=th.float32, device=self.device)
        ep_obs_next = th.tensor(buffer['obs_next'], dtype=th.float32, device=self.device)
        ep_rew = buffer['rew']
        ep_done = buffer['done']

        with th.no_grad():
            vs, _, _, _ = self.evaluate(ep_obs)
            vs_next, _, _, _ = self.evaluate(ep_obs_next)
        vs_numpy, vs_next_numpy = vs.cpu().numpy(), vs_next.cpu().numpy()
        if self.conf['is_target_gae']:
            adv_numpy = _gae_return(
                vs_numpy, vs_next_numpy, ep_rew, ep_done, self.conf['gamma'], self.conf['gae_lambda']
            )
            target_numpy = adv_numpy + vs_numpy
        else:
            target_numpy = _discount_cumsum(ep_rew, ep_done, self.conf['gamma'])
            adv_numpy = target_numpy - vs_numpy
        if self.conf['norm_adv']:
            adv_numpy = (adv_numpy - adv_numpy.mean()) / (adv_numpy.std() + 1e-8)
        adv = th.tensor(adv_numpy, dtype=th.float32, device=self.device)
        target = th.tensor(target_numpy, dtype=th.float32, device=self.device)
        return ep_obs, ep_act, ep_log_prob, adv, target

    def evaluate(self, ep_obs, ep_act=None):
        vs = self.critic(ep_obs).squeeze()
        log_probs, para1, para2 = None, None, None
        if ep_act is not None:
            para1, para2 = self.actor(ep_obs)
            dist = self.dist_fn(para1, para2)
            log_probs = dist.log_prob(ep_act)
            para1 = para1.detach().cpu().numpy()
            para2 = para2.detach().cpu().numpy()
        return vs, log_probs, para1, para2

    def lr_decay(self, epoch):
        lr_a_now = self.conf['lr_actor'] * (1 - epoch / self.conf['n_epoch'])
        lr_c_now = self.conf['lr_critic'] * (1 - epoch / self.conf['n_epoch'])
        for p in self.actor_optim.param_groups:
            p['lr'] = lr_a_now
        for p in self.critic_optim.param_groups:
            p['lr'] = lr_c_now
