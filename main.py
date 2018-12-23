#!/usr/bin/env python3

# Copyright (c) 2016 Anki, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License in the file LICENSE.txt or at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import random
import time
import numpy as np
import argparse
from itertools import count
from collections import namedtuple

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import cozmo
from cozmo.util import degrees, distance_mm, speed_mmps


NUM_DEGREES_ROTATE = 5
STEPS_IN_EPISODE = 15


parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(14400, 128)
        self.action_head = nn.Linear(128, 2)
        self.value_head = nn.Linear(128, 1)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(action_scores, dim=-1), state_values


model = Policy()
optimizer = optim.Adam(model.parameters(), lr=3e-2)
eps = np.finfo(np.float32).eps.item()


def select_action(state, flatten=True):
    if flatten:
        state = state.flatten()
    state = torch.from_numpy(state).float()
    probs, state_value = model(state)
    m = Categorical(probs)
    action = m.sample()
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))
    return action.item()


def finish_episode():
    R = 0
    saved_actions = model.saved_actions
    policy_losses = []
    value_losses = []
    rewards = []
    for r in model.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    for (log_prob, value), r in zip(saved_actions, rewards):
        reward = r - value.item()
        policy_losses.append(-log_prob * reward)
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([r])))
    optimizer.zero_grad()
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    loss.backward()
    optimizer.step()
    del model.rewards[:]
    del model.saved_actions[:]

# -------------------
# Environment helper functions
# -------------------

def get_annotated_image(robot, size=(80, 60), return_PIL=False):
    image = robot.world.latest_image
    # if _display_debug_annotations != DEBUG_ANNOTATIONS_DISABLED:
    #     image = image.annotate_image(scale=2)
    # else:
    image = image.raw_image
    image.thumbnail(size, Image.ANTIALIAS)
    if return_PIL:
        return image
    return np.asarray(image)

def get_reward(step_num):
    if step_num >= STEPS_IN_EPISODE - 1:
        done = True
    else:
        done = False
    reward = 0

    return reward, done

# def show_image(pil_image):


def cozmo_run_training_loop(robot: cozmo.robot.Robot):
    robot = robot.wait_for_robot()
    robot.enable_device_imu(True, True, True)
    # Turn on image receiving by the camera
    robot.camera.image_stream_enabled = True

    robot.set_lift_height(0.0).wait_for_completed()
    robot.set_head_angle(degrees(-.0)).wait_for_completed()

    time.sleep(1)
    running_reward = 10

    for episode_num in range(1):
        # state = env.reset()
        state = get_annotated_image(robot)  # todo get default image?
        for step_num in range(STEPS_IN_EPISODE):

            # robot.drive_straight(distance_mm(150), speed_mmps(50)).wait_for_completed()

            # action = random.randint(0, 1)
            action = select_action(state)
            # todo put below into step function
            if action == 0:
                print('Turning right')
                robot.turn_in_place(degrees(-NUM_DEGREES_ROTATE)).wait_for_completed()
            elif action == 1:
                print('Turning left')
                robot.turn_in_place(degrees(NUM_DEGREES_ROTATE)).wait_for_completed()
            # image = robot.world.latest_image
            state = get_annotated_image(robot)
            reward, done = get_reward(step_num)

            model.rewards.append(reward)

            # if state:
            #     print(state)

            if done:
                finish_episode()
                break

            if args.render:
                img = Image.fromarray(state, 'RGB')
                img.show()

            running_reward = running_reward * 0.99 + step_num * 0.01

            if episode_num % args.log_interval == 0:
                print('Episode {}\tLast length: {:5d}\t'.format(episode_num, step_num))

            # if running_reward > env.spec.reward_threshold:
            #     print("Solved! Running reward is now {} and "
            #           "the last episode runs to {} time steps!".format(running_reward, t))
            #     break
            # todo possible reset: robot.go_to_pose(Pose(100, 100, 0, angle_z=degrees(45)), relative_to_robot=True).wait_for_completed()

if __name__ == '__main__':
    # cozmo.run_program(cozmo_run_training_loop) # whats the difference?
    try:
        cozmo.connect(cozmo_run_training_loop)
    except KeyboardInterrupt as e:
        pass
    except cozmo.ConnectionError as e:
        sys.exit("A connection error occurred: %s" % e)
