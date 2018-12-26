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

import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import cozmo
from cozmo.util import degrees, distance_mm, speed_mmps

#from PyTorch_YOLOv3 import detect_function
from PyTorch_YOLOv3.detect_function import *

NUM_DEGREES_ROTATE = 10
STEPS_IN_EPISODE = 1000


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


class PolicyAffine(nn.Module):
    def __init__(self):
        super(PolicyAffine, self).__init__()
        # self.affine1 = nn.Linear(14400, 128)
        self.affine1 = nn.Linear(57600, 128)
        self.action_head = nn.Linear(128, 2)
        self.value_head = nn.Linear(128, 1)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(action_scores, dim=-1), state_values

class PolicyConv(nn.Module):
    def __init__(self):
        """
        (160 - 5 + 0 (padding)) / 2 + 1 = 78.5
        """
        super(PolicyConv, self).__init__()
        # self.affine1 = nn.Linear(14400, 128)
        self.conv1 = nn.Conv2d(3, 20, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(20, 20, kernel_size=5, stride=2) # todo maybe one more
        self.affine1 = nn.Linear(37 * 27 * 20, 128)
        self.action_head = nn.Linear(128, 2)
        self.value_head = nn.Linear(128, 1)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.affine1(x))
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(action_scores, dim=-1), state_values


model = PolicyConv()
optimizer = optim.Adam(model.parameters(), lr=3e-4)
eps = np.finfo(np.float32).eps.item()


def select_action(state, flatten=False):
    if flatten:
        state = state.flatten()
    else:
        state = np.expand_dims(np.transpose(state, (2, 0, 1)), axis=0)
    state = torch.from_numpy(state).float()
    probs, state_value = model(state)
    m = Categorical(probs)
    action = m.sample()
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))
    return action.item()

def finish_episode():  # todo rename
    print('Calculating returns, losses and optimising')
    print('raw rewards: {}'.format(model.rewards))
    if len(model.rewards) < 2:
        del model.rewards[:]
        del model.saved_actions[:]
        return
    R = 0
    saved_actions = model.saved_actions
    policy_losses = []
    value_losses = []
    rewards = []
    for r in model.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards)
    print('raw returns: {}'.format(rewards))
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)  # todo maybe don't
    print('normalised returns: {}'.format(rewards))
    for (log_prob, value), r in zip(saved_actions, rewards):
        reward = r - value.item()
        policy_losses.append(-log_prob * reward)
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([r]).view(value.size(0), -1)))
    print('policy_losses: {}'.format(policy_losses))
    print('value_losses: {}'.format(value_losses))
    optimizer.zero_grad()
    policy_loss = torch.stack(policy_losses).sum()
    value_loss = torch.stack(value_losses).sum()
    loss = policy_loss + value_loss
    loss.backward()
    optimizer.step()

    print('Policy loss sum: {}. Value loss: {}'.format(policy_loss.item(), value_loss.item()))
    del model.rewards[:]
    del model.saved_actions[:]

# -------------------
# Environment helper functions
# -------------------

def reset_env(robot):
    """
    Rotate cozmo random amount
    """
    # todo possible reset: robot.go_to_pose(Pose(100, 100, 0, angle_z=degrees(45)), relative_to_robot=True).wait_for_completed() random is better
    random_degrees = random.randint(-150, 150)
    robot.turn_in_place(degrees(random_degrees)).wait_for_completed()
    print('Resetting environment, rotating Cozmo {} degrees'.format(random_degrees))

def cup_in_middle_of_screen(img, dead_centre=False):
    """
    Use YOLOv3 PyTorch
    image is 416x416. centre is 208, 208. Solid centre of screen object is around ~?
    """

    detections = detect_on_numpy_img(img)
    if detections is not None:
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            if dead_centre:
                if int(cls_pred.item()) == 41 and x1 > 108 and x2 < 308:
                        return True
            else:  # much easier, just have to see cup
                if int(cls_pred.item()) == 41:
                    return True
    return False

def get_reward(step_num, img):
    reward = 0
    if step_num >= STEPS_IN_EPISODE - 1:
        done = True
    elif cup_in_middle_of_screen(img):
        done = True
        reward = 10
    else:
        done = False

    return reward, done

def get_annotated_image(robot, resize=False, size=(160, 120), return_PIL=False):
    image = robot.world.latest_image
    # if _display_debug_annotations != DEBUG_ANNOTATIONS_DISABLED:
    #     image = image.annotate_image(scale=2)
    # else:
    image = image.raw_image
    if resize:
        image.thumbnail(size, Image.ANTIALIAS)
    if return_PIL:
        return image
    return np.asarray(image)

def cozmo_run_training_loop(robot: cozmo.robot.Robot):
    robot = robot.wait_for_robot()
    robot.enable_device_imu(True, True, True)
    # Turn on image receiving by the camera
    robot.camera.image_stream_enabled = True

    robot.set_lift_height(0.0).wait_for_completed()
    robot.set_head_angle(degrees(-.0)).wait_for_completed()

    time.sleep(1)
    running_reward = 10

    for episode_num in range(100):
        # state = env.reset()
        reset_env(robot)
        state = get_annotated_image(robot)  # todo get default image?
        for step_num in range(STEPS_IN_EPISODE):
            # action = random.randint(0, 1)
            action = select_action(cv2.resize(state, (160, 120)))
            # todo put below into step function
            if action == 0:
                print('Turning right')
                robot.turn_in_place(degrees(-NUM_DEGREES_ROTATE)).wait_for_completed()
            elif action == 1:
                print('Turning left')
                robot.turn_in_place(degrees(NUM_DEGREES_ROTATE)).wait_for_completed()
            state = get_annotated_image(robot) # todo don't resize here
            reward, done = get_reward(step_num, state)

            model.rewards.append(reward)

            if done:
                print('Episode over, final reward: {}'.format(reward))
                finish_episode()  # learn policy with backprop
                break

            if args.render:
                img = Image.fromarray(state, 'RGB')
                img.show()

            running_reward = running_reward * 0.99 + step_num * 0.01

            if step_num % args.log_interval == 0:
                print('Episode {}\t Step number: {:5d}\t'.format(episode_num, step_num))

if __name__ == '__main__':
    # cozmo.run_program(cozmo_run_training_loop) # whats the difference?
    try:
        cozmo.connect(cozmo_run_training_loop)
    except KeyboardInterrupt as e:
        pass
    except cozmo.ConnectionError as e:
        sys.exit("A connection error occurred: %s" % e)
