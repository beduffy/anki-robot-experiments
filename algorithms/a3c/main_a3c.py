"""
Adapted from: https://github.com/ikostrikov/pytorch-a3c/blob/master/main.py
Example of use:
`cd algorithms/a3c`
`python main_a3c.py`

Runs A3C on our custom environment for Cozmo (robot by anki) with only 1 process (so A2C essentially).
"""

from __future__ import print_function

import time
import argparse
import os
import uuid
import glob
import sys
# sys.path.append('../..')
sys.path.append('/home/beduffy/all_projects/cozmo-anki-experiments/')

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import cozmo

from anki_env import AnkiEnv
from algorithms.a3c.model import ActorCritic, A3C_LSTM_GA

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
parser.add_argument('--checkpoint-freq', type=int, default=100, help='how often to checkpoint')
parser.add_argument('--total-length', type=int, default=0, help='initial length if resuming')
parser.add_argument('--episode-number', type=int, default=0, help='episode-number passed if resuming')
parser.add_argument('-eid', '--experiment-id', default=uuid.uuid4(),
                    help='random or chosen guid for folder creation for plots and checkpointing. '
                         'If experiment taken, will resume training!')
parser.add_argument('--natural-language', dest='natural-language', action='store_true',
                    help='')
parser.set_defaults(natural_language=True)  # todo
parser.add_argument('--render', dest='render', action='store_true', help='render env with cv2')
parser.set_defaults(render=True)
parser.add_argument('--no-shared', default=False,
                    help='use an optimizer without shared momentum.')

args = parser.parse_args()

def save_checkpoint(fp, state, is_best=False):
    torch.save(state, fp)
    print('Saved model to path: {}'.format(fp))
    # if is_best:
    #     shutil.copyfile(filepath, 'model_best.pth.tar')

def train(robot: cozmo.robot.Robot):
    rank = 0
    torch.manual_seed(args.seed + rank)

    env = AnkiEnv(robot, natural_language=args.natural_language, render=args.render)
    env.seed(args.seed + rank)

    if args.natural_language:
        model = A3C_LSTM_GA(env.observation_space.shape[0], env.action_space.n, args.frame_dim, len(env.word_to_idx)).float()
    else:
        model = ActorCritic(env.observation_space.shape[0], env.action_space.n, args.frame_dim)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # Checkpoint creation/loading below
    if not os.path.exists(args.checkpoint_path):
        print('Tensorboard created experiment folder: {} and checkpoint folder made here: {}'.format(
            args.experiment_path, args.checkpoint_path))
        os.makedirs(args.checkpoint_path)
    else:
        print('Checkpoints path already exists at path: {}'.format(args.checkpoint_path))
        checkpoint_paths = glob.glob(os.path.join(args.checkpoint_path, '*'))
        if checkpoint_paths:
            # Take checkpoint path with most experience e.g. 2000 from checkpoint_total_length_2000.pth.tar
            checkpoint_file_name_ints = [int(x.split('/')[-1].split('.pth.tar')[0].split('_')[-1])
                                         for x in checkpoint_paths]
            idx_of_latest = checkpoint_file_name_ints.index(max(checkpoint_file_name_ints))
            checkpoint_to_load = checkpoint_paths[idx_of_latest]
            print('Loading latest checkpoint: {}'.format(checkpoint_to_load))

            if os.path.isfile(checkpoint_to_load):
                print("=> loading checkpoint '{}'".format(checkpoint_to_load))
                checkpoint = torch.load(checkpoint_to_load)
                args.total_length = checkpoint['total_length']
                args.episode_number = checkpoint['episode_number']
                print('Values from checkpoint: total_length: {}. episode_number: {}'.format(
                    checkpoint['total_length'], checkpoint['episode_number']))
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(
                    checkpoint['optimizer'])  # todo check if overwrites learning rate. probably does

                for param_group in optimizer.param_groups:
                    print('Learning rate: ', param_group['lr'])  # oh it doesn't work?

                print("=> loaded checkpoint '{}' (total_length {})"
                      .format(checkpoint_to_load, checkpoint['total_length']))
        else:
            print('No checkpoint to load')
        # todo have choice of checkpoint as well? args.resume could override the above

    model.train()

    state = env.reset()
    if args.natural_language:
        (state, instruction) = state
        instruction_idx = []
        for word in instruction.split(" "):
            instruction_idx.append(env.word_to_idx[word])
        instruction_idx = np.array(instruction_idx)
        instruction_idx = torch.from_numpy(instruction_idx).view(1, -1)

    state = torch.from_numpy(state)
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
    total_length = args.total_length if args.total_length is not None else 0
    episode_number = args.episode_number if args.episode_number is not None else 0
    episode_length = 0
    num_backprops = 0
    while True:
        # Sync with the shared model
        # model.load_state_dict(shared_model.state_dict())
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
            if total_length > 0 and total_length % args.checkpoint_freq == 0:
                fp = os.path.join(args.checkpoint_path, 'checkpoint_total_length_{}.pth.tar'.format(total_length))
                save_checkpoint(fp,
                     {
                         'total_length': total_length,
                         'episode_number': episode_number,
                         'state_dict': model.state_dict(),
                         'optimizer': optimizer.state_dict(),
                     })
                print('Saved checkpoint to: {}'.format(fp))
            if args.natural_language:
                tx = torch.from_numpy(np.array([episode_length])).long()
                value, logit, (hx, cx) = model((state.unsqueeze(0).float(),
                                                instruction_idx.float(),
                                                (tx, hx, cx)))
            else:
                value, logit, (hx, cx) = model((state.unsqueeze(0).float(), (hx, cx)))
            prob = F.softmax(logit, dim=-1)
            log_prob = F.log_softmax(logit, dim=-1)
            entropy = -(log_prob * prob).sum(1, keepdim=True)
            entropies.append(entropy)

            action = prob.multinomial(num_samples=1).detach()
            log_prob = log_prob.gather(1, action)

            action_int = action.numpy()[0][0].item()
            state, reward, done, _ = env.step(action_int)
            if args.natural_language:
                state, instruction = state

            with lock:
                counter.value += 1

            if done:
                episode_number += 1
                total_reward_for_episode = sum(all_rewards_in_episode)
                episode_total_rewards_list.append(total_reward_for_episode)
                all_rewards_in_episode = []
                state = env.reset()
                if args.natural_language:
                    state, instruction = state
                    instruction_idx = []
                    for word in instruction.split(" "):
                        instruction_idx.append(env.word_to_idx[word])
                    instruction_idx = np.array(instruction_idx)
                    instruction_idx = torch.from_numpy(instruction_idx).view(1, -1)
                print('Episode Over. Total Length: {}. Total reward for episode: {}'.format(
                                            total_length,  total_reward_for_episode))
                print('Episode number: {}. Step no: {}. total length: {}'.format(episode_number, episode_length, total_length))
                writer.add_scalar('episode_lengths', episode_length, episode_number)
                # todo do running mean reward
                writer.add_scalar('episode_total_rewards', total_reward_for_episode, episode_number)
                # writer.add_image('Image', torch.from_numpy(state).permute(2, 0, 1), episode_num)
                # writer.add_image('Image', state, episode_number)  # was state
                writer.add_text('Text', 'text logged at step: {}. '
                                        'Episode num {}'.format(episode_length, episode_number), episode_length)
                episode_length = 0
                total_length -= 1  # so no off by 1 errors. hmm shouldn't be necessary

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
            if args.natural_language:
                tx = torch.from_numpy(np.array([episode_length])).long()
                value, logit, (hx, cx) = model((state.unsqueeze(0).float(),
                                                instruction_idx.float(),
                                                (tx, hx, cx)))
            else:
                value, _, _ = model((state.unsqueeze(0).float(), (hx, cx)))
            R = value.detach()

        values.append(R)
        policy_loss = 0
        value_loss = 0
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

        writer.add_scalar('policy_loss', policy_loss.item(), episode_number)
        writer.add_scalar('value_loss', value_loss.item(), episode_number)
        (policy_loss + args.value_loss_coef * value_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        # ensure_shared_grads(model, shared_model)
        optimizer.step()


if __name__ == '__main__':
    # todo visualise A3C conv layers or attention
    # todo make cozmo to charge himself so we can train all night?
    # todo pretrain with imagenet?
    # todo try binary image input with objects in white (or segmented)
    # todo print action probabilities
    torch.manual_seed(args.seed)
    args.frame_dim = 80
    args.experiment_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'
                                                                     , '..', 'experiments',
                                                                     str(args.experiment_id))))
    args.checkpoint_path = os.path.join(args.experiment_path, 'checkpoints')
    args.tensorboard_path = os.path.join(args.experiment_path, 'tensorboard_logs')
    counter = mp.Value('i', 0)
    lock = mp.Lock()
    writer = SummaryWriter(comment='A3C', log_dir=args.tensorboard_path)

    try:
        cozmo.connect(train)
    except KeyboardInterrupt as e:
        pass
    except cozmo.ConnectionError as e:
        sys.exit("A connection error occurred: %s" % e)
    finally:
        writer.export_scalars_to_json("./all_scalars.json")
        writer.close()

