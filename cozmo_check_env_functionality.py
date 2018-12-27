from __future__ import print_function
import os
import time
import argparse
import os
import sys
# sys.path.append('../..')
sys.path.append('/home/beduffy/all_projects/cozmo-anki-experiments/')

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.optim as optim
import torch.nn.functional as F
import cozmo

from collections import namedtuple

import cv2
from tensorboardX import SummaryWriter
import cozmo
from cozmo.util import degrees, distance_mm, speed_mmps

#from PyTorch_YOLOv3 import detect_function  # should be able to swap to
from PyTorch_YOLOv3.detect_function import *  # needed to load in YOLO model. Remove both?
from anki_env import AnkiEnv, IMAGE_DIM_INPUT

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
parser.set_defaults(natural_language=True)  # todo
parser.add_argument('--no-shared', default=False,
                    help='use an optimizer without shared momentum.')
parser.add_argument('-sync', '--synchronous', dest='synchronous', action='store_true',
                    help='Useful for debugging purposes e.g. import pdb; pdb.set_trace(). '
                         'Overwrites args.num_processes as everything is in main thread. '
                         '1 train() function is run and no test()')
parser.add_argument('-async', '--asynchronous', dest='synchronous', action='store_false')
parser.set_defaults(synchronous=True)


def test_natural_language(robot: cozmo.robot.Robot):
    env = AnkiEnv(robot, natural_language=True, degrees_rotate=5)

    state = env.reset(random_rotate=False)
    for step_num, action in enumerate([0, 0, 0, 1, 1, 1, 1, 1, 1]):
        import pdb; pdb.set_trace()  # step through cozmo's movements and check if reward signal is right for bowl and cup
        # can change action here by doing `action = 1` 0 = right, 1 = left

        state, reward, done, _ = env.step(action)
        raw_state = env.get_raw_image()
        img = Image.fromarray(raw_state, 'RGB')
        img.show()

        print('With confidence threshold 0.8')
        detections = detect_on_numpy_img(raw_state)
        print('With confidence threshold 0.5')
        detections_half_conf = detect_on_numpy_img(raw_state, confidence_threshold=0.5)
        print('With confidence threshold 0.25')
        detections_quar_conf = detect_on_numpy_img(raw_state, confidence_threshold=0.25)

        # Create plot
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(raw_state)

        # The amount of padding that was added
        pad_x = max(raw_state.shape[0] - raw_state.shape[1], 0) * (opt.img_size / max(raw_state.shape))
        pad_y = max(raw_state.shape[1] - raw_state.shape[0], 0) * (opt.img_size / max(raw_state.shape))
        # Image height and width after padding is removed
        unpad_h = opt.img_size - pad_y
        unpad_w = opt.img_size - pad_x

        # Draw bounding boxes and labels of detections
        if detections is not None:
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                print('\t+ Label: %s, Conf: %.5f' % (classes[int(cls_pred)], cls_conf.item()))
                print('projected coordinates: x1, y1, x2, y2: {:.2f}, {:.2f}, {:.2f}, {:.2f}'.format(x1, y1, x2, y2))

                # Rescale coordinates to original dimensions
                box_h = ((y2 - y1) / unpad_h) * raw_state.shape[0]
                box_w = ((x2 - x1) / unpad_w) * raw_state.shape[1]
                y1 = ((y1 - pad_y // 2) / unpad_h) * raw_state.shape[0]
                x1 = ((x1 - pad_x // 2) / unpad_w) * raw_state.shape[1]


                print('original coordinates: x1, y1, x2, y2: {:.2f}, {:.2f}, {:.2f}, {:.2f}'.format(x1, y1, x1 + box_w, y1 + box_h))

                color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                # Create a Rectangle patch
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2,
                                         edgecolor=color,
                                         facecolor='none')
                # Add the bbox to the plot
                ax.add_patch(bbox)
                # Add label
                plt.text(x1, y1, s=classes[int(cls_pred)], color='white', verticalalignment='top',
                         bbox={'color': color, 'pad': 0})

        # Save generated image with detections
        plt.axis('off')
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        output_fig_path = 'output/test-pic-%d.png' % (step_num)
        plt.savefig(output_fig_path, bbox_inches='tight', pad_inches=0.0)
        print('Saved image path: {}'.format(output_fig_path))
        plt.close()

        # state_preprocessed = cv2.resize(state, IMAGE_DIM_INPUT)
        # state_preprocessed = np.moveaxis(state_preprocessed, 2, 0)
        #
        # img = Image.fromarray(state, 'RGB')
        # img.show()
        #
        # img = Image.fromarray(state_preprocessed, 'RGB')
        # img.show()

        if done:
            state = env.reset(random_rotate=False)

if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ""

    args = parser.parse_args()
    processes = []
    counter = mp.Value('i', 0)
    lock = mp.Lock()
    # cozmo.run_program(train)  # doesn't pass robot correctly
    try:
        cozmo.connect(test_natural_language)
    except KeyboardInterrupt as e:
        pass
    except cozmo.ConnectionError as e:
        sys.exit("A connection error occurred: %s" % e)
