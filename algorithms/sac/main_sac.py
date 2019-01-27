"""
Adapted from: https://github.com/higgsfield/RL-Adventure-2/blob/master/7.soft%20actor-critic.ipynb
Example of use:
`cd algorithms/sac`
`python main_sac.py`

Runs SAC on our custom environment for Cozmo (robot by anki)
"""

from __future__ import print_function

import time
import argparse
import os
import sys
# sys.path.append('../..')
sys.path.append('/home/beduffy/all_projects/cozmo-anki-experiments/')

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from tensorboardX import SummaryWriter
import cozmo

import gym

from anki_env import AnkiEnv
from algorithms.sac.model import ReplayBuffer, NormalizedActions, ValueNetwork, SoftQNetwork, PolicyNetwork


# Based on
# https://github.com/pytorch/examples/tree/master/mnist_hogwild
# Training settings
parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--tau', type=float, default=1.00,
                    help='parameter for GAE (default: 1.00)')
parser.add_argument('--entropy-coef', type=float, default=0.01,
                    help='entropy term coefficient (default: 0.01)')
parser.add_argument('--value-loss-coef', type=float, default=0.5,
                    help='value loss coefficient (default: 0.5)')
parser.add_argument('--max-grad-norm', type=float, default=50,
                    help='value loss coefficient (default: 50)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--test-sleep-time', type=int, default=200,
                    help='number of seconds to wait before testing again (default: 200)')
parser.add_argument('--num-processes', type=int, default=4,
                    help='how many training processes to use (default: 1)')
parser.add_argument('--num-steps', type=int, default=20,
                    help='number of forward steps in A3C (default: 20)')
parser.add_argument('--max-episode-length', type=int, default=100,
                    help='maximum length of an episode (default: 1000000)')

parser.add_argument('--natural-language', dest='natural-language', action='store_true',
                    help='')
parser.set_defaults(natural_language=True)  # todo
parser.add_argument('--render', dest='render', action='store_true', help='render env with cv2')
parser.set_defaults(render=True)
parser.add_argument('--no-shared', default=False,
                    help='use an optimizer without shared momentum.')

def plot_episode_rewards(frame_idx, rewards):
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.draw()
    plt.pause(0.001)


def soft_q_update(batch_size,
                  gamma=0.99,
                  mean_lambda=1e-3,
                  std_lambda=1e-3,
                  z_lambda=0.0,
                  soft_tau=1e-2,
                  ):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = torch.FloatTensor(state).to(device)
    next_state = torch.FloatTensor(next_state).to(device)
    action = torch.FloatTensor(action).to(device)
    reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
    done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

    expected_q_value = soft_q_net(state, action)
    expected_value = value_net(state)
    new_action, log_prob, z, mean, log_std = policy_net.evaluate(state)

    target_value = target_value_net(next_state)
    next_q_value = reward + (1 - done) * gamma * target_value
    q_value_loss = soft_q_criterion(expected_q_value, next_q_value.detach())

    expected_new_q_value = soft_q_net(state, new_action)
    next_value = expected_new_q_value - log_prob
    value_loss = value_criterion(expected_value, next_value.detach())

    log_prob_target = expected_new_q_value - expected_value
    policy_loss = (log_prob * (log_prob - log_prob_target).detach()).mean()

    mean_loss = mean_lambda * mean.pow(2).mean()
    std_loss = std_lambda * log_std.pow(2).mean()
    z_loss = z_lambda * z.pow(2).sum(1).mean()

    policy_loss += mean_loss + std_loss + z_loss

    soft_q_optimizer.zero_grad()
    q_value_loss.backward()
    soft_q_optimizer.step()

    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()

    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()

    for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - soft_tau) + param.data * soft_tau
        )


if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    plt.ion()
    plt.show()
    env = NormalizedActions(gym.make("Pendulum-v0"))

    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]
    hidden_dim = 256

    value_net = ValueNetwork(state_dim, hidden_dim).to(device)
    target_value_net = ValueNetwork(state_dim, hidden_dim).to(device)

    soft_q_net = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
    policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, device).to(device)

    for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
        target_param.data.copy_(param.data)

    value_criterion = nn.MSELoss()
    soft_q_criterion = nn.MSELoss()

    value_lr = 3e-4
    soft_q_lr = 3e-4
    policy_lr = 3e-4

    value_optimizer = optim.Adam(value_net.parameters(), lr=value_lr)
    soft_q_optimizer = optim.Adam(soft_q_net.parameters(), lr=soft_q_lr)
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=policy_lr)

    replay_buffer_size = 1000000
    replay_buffer = ReplayBuffer(replay_buffer_size)

    max_frames = 40000
    max_steps = 500
    frame_idx = 0
    rewards = []
    batch_size = 128

    while frame_idx < max_frames:
        state = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            action = policy_net.get_action(state)
            next_state, reward, done, _ = env.step(action)

            replay_buffer.push(state, action, reward, next_state, done)
            if len(replay_buffer) > batch_size:
                soft_q_update(batch_size)

            state = next_state
            episode_reward += reward
            frame_idx += 1

            env.render()

            if frame_idx % 1000 == 0:
                print('Frame idx: {}. Episode num: {}'.format(frame_idx, len(rewards)))
                plot_episode_rewards(frame_idx, rewards)

            if done:
                break

        rewards.append(episode_reward)
