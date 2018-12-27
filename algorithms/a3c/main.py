"""
Adapted from: https://github.com/ikostrikov/pytorch-a3c/blob/master/main.py
The main file needed within a3c. Runs of the train and test functions from their respective files.
Example of use:
`cd algorithms/a3c`
`python main.py`

Runs A3C on our AI2ThorEnv wrapper with default params (4 processes). Optionally it can be
run on any atari environment as well using the --atari and --atari-env-name params.
"""

from __future__ import print_function

import time
import argparse
import os
import sys
# sys.path.append('../..')
sys.path.append('/home/beduffy/all_projects/cozmo-anki-experiments/')

import torch
import torch.multiprocessing as mp
import torch.optim as optim
import cozmo

# from gym_ai2thor.envs.ai2thor_env import AI2ThorEnv
from anki_env import AnkiEnv
from algorithms.a3c.envs import create_atari_env
from algorithms.a3c import my_optim
from algorithms.a3c.model import ActorCritic, A3C_LSTM_GA
from algorithms.a3c.test import test
from algorithms.a3c.train import train

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
parser.add_argument('--max-episode-length', type=int, default=1000,
                    help='maximum length of an episode (default: 1000000)')
parser.add_argument('--natural-language', dest='natural-language', action='store_true',
                    help='')
parser.set_defaults(natural_language=False)
parser.add_argument('--no-shared', default=False,
                    help='use an optimizer without shared momentum.')
parser.add_argument('-sync', '--synchronous', dest='synchronous', action='store_true',
                    help='Useful for debugging purposes e.g. import pdb; pdb.set_trace(). '
                         'Overwrites args.num_processes as everything is in main thread. '
                         '1 train() function is run and no test()')
parser.add_argument('-async', '--asynchronous', dest='synchronous', action='store_false')
parser.set_defaults(synchronous=True)

# Atari arguments. Good example of keeping code modular and allowing algorithms to run everywhere
parser.add_argument('--atari', dest='atari', action='store_true',
                    help='Run atari env instead with name below instead of ai2thor')
parser.add_argument('--atari-render', dest='atari_render', action='store_true',
                    help='Render atari')
parser.add_argument('--atari-env-name', default='PongDeterministic-v4',
                    help='environment to train on (default: PongDeterministic-v4)')
#
parser.set_defaults(atari=False)
parser.set_defaults(atari_render=False)


from algorithms.a3c.envs import create_atari_env
from algorithms.a3c.model import ActorCritic, A3C_LSTM_GA


def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


# def train(rank, args, shared_model, counter, lock, optimizer=None):
def train(robot: cozmo.robot.Robot):
    rank = 0
    torch.manual_seed(args.seed + rank)

    # if args.atari:
    #     env = create_atari_env(args.atari_env_name)
    # else:
    #     env = AI2ThorEnv(config_dict=args.config_dict)

    env = AnkiEnv(robot)
    env.seed(args.seed + rank)

    if args.natural_language:
        model = A3C_LSTM_GA(env.observation_space.shape[0], env.action_space.n, args.frame_dim)
    else:
        model = ActorCritic(env.observation_space.shape[0], env.action_space.n, args.frame_dim)

    if optimizer is None:
        optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)

    model.train()

    state = env.reset()
    state = torch.from_numpy(state)
    # (image, instruction) = env.reset()
    # instruction_idx = []
    # for word in instruction.split(" "):
    #     instruction_idx.append(env.word_to_idx[word])
    # instruction_idx = np.array(instruction_idx)
    # image = torch.from_numpy(image)
    # instruction_idx = torch.from_numpy(instruction_idx).view(1, -1)
    done = True

    # monitoring
    total_reward_for_num_steps_list = []
    episode_total_rewards_list = []
    all_rewards_in_episode = []
    avg_reward_for_num_steps_list = []
    episode_lengths = []
    p_losses = []
    v_losses = []

    start = time.time()
    total_length = 0
    episode_length = 0
    num_backprops = 0
    while True:
        # Sync with the shared model
        model.load_state_dict(shared_model.state_dict())
        if done:
            cx = torch.zeros(1, 256)
            hx = torch.zeros(1, 256)
        else:
            cx = cx.detach()
            hx = hx.detach()

        values = []
        log_probs = []
        rewards = []
        entropies = []

        interaction_start_time = time.time()
        for step in range(args.num_steps):
            episode_length += 1
            total_length += 1
            value, logit, (hx, cx) = model((state.unsqueeze(0).float(), (hx, cx)))
            prob = F.softmax(logit, dim=-1)
            log_prob = F.log_softmax(logit, dim=-1)
            entropy = -(log_prob * prob).sum(1, keepdim=True)
            entropies.append(entropy)

            action = prob.multinomial(num_samples=1).detach()
            log_prob = log_prob.gather(1, action)

            action_int = action.numpy()[0][0].item()
            state, reward, done, _ = env.step(action_int)
            # (image, _), reward, done = env.step(action)

            done = done or episode_length >= args.max_episode_length

            with lock:
                counter.value += 1

            if done:
                episode_length = 0
                total_length -= 1
                total_reward_for_episode = sum(all_rewards_in_episode)
                episode_total_rewards_list.append(total_reward_for_episode)
                all_rewards_in_episode = []
                state = env.reset()
                print('Episode Over. Total Length: {}. Total reward for episode: {}'.format(
                                            total_length,  total_reward_for_episode))
                print('Step no: {}. total length: {}'.format(episode_length, total_length))

            state = torch.from_numpy(state)
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)
            all_rewards_in_episode.append(reward)

            if done:
                break

        # No interaction with environment below.
        # Monitoring
        total_reward_for_num_steps = sum(rewards)
        total_reward_for_num_steps_list.append(total_reward_for_num_steps)
        avg_reward_for_num_steps = total_reward_for_num_steps / len(rewards)
        avg_reward_for_num_steps_list.append(avg_reward_for_num_steps)

        # Backprop and optimisation
        R = torch.zeros(1, 1)
        if not done:  # to change last reward to predicted value to ....
            value, _, _ = model((state.unsqueeze(0).float(), (hx, cx)))
            R = value.detach()

        values.append(R)
        policy_loss = 0
        value_loss = 0
        # import pdb;pdb.set_trace() # good place to breakpoint to see training cycle
        gae = torch.zeros(1, 1)
        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimation
            delta_t = rewards[i] + args.gamma * values[i + 1] - values[i]
            gae = gae * args.gamma * args.tau + delta_t

            policy_loss = policy_loss - log_probs[i] * gae.detach() - \
                          args.entropy_coef * entropies[i]

        optimizer.zero_grad()

        (policy_loss + args.value_loss_coef * value_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        # ensure_shared_grads(model, shared_model)
        optimizer.step()


if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ""

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    args.config_dict = {'max_episode_length': args.max_episode_length,
                        'natural_language': args.natural_language}
    # env = AnkiEnv(config_dict=args.config_dict)
    # env = AnkiEnv()
    # args.frame_dim = env.config['resolution'][-1]

    if args.natural_language:
        shared_model = A3C_LSTM_GA(env.observation_space.shape[0], env.action_space.n,
                                   args.frame_dim)
    else:
        shared_model = ActorCritic(env.observation_space.shape[0], env.action_space.n,
                                   args.frame_dim)
    shared_model.share_memory()

    if args.no_shared:
        optimizer = None
    else:
        optimizer = my_optim.SharedAdam(shared_model.parameters(), lr=args.lr)
        optimizer.share_memory()

    processes = []

    counter = mp.Value('i', 0)
    lock = mp.Lock()
    cozmo.run_program(train)

