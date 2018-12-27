import cv2
from PIL import Image
import cozmo
from cozmo.util import degrees, distance_mm, speed_mmps, Pose
import gym
from gym import error, spaces
from gym.utils import seeding

#from PyTorch_YOLOv3 import detect_function
from PyTorch_YOLOv3.detect_function import *  # needed to load in YOLO model

NUM_DEGREES_ROTATE = 10
STEPS_IN_EPISODE = 100
# IMAGE_DIM_INPUT = (80, 60)
IMAGE_DIM_INPUT = (80, 80)  # will stretch a bit but for A3C


class AnkiEnv(gym.Env):
    def __init__(self, robot, natural_language=False, degrees_rotate=NUM_DEGREES_ROTATE, render=False):
        self.robot = robot
        self.robot = self.robot.wait_for_robot()
        self.robot.enable_device_imu(True, True, True)
        # Turn on image receiving by the camera
        self.robot.camera.image_stream_enabled = True

        self.robot.set_lift_height(0.0).wait_for_completed()
        self.robot.set_head_angle(degrees(-.0)).wait_for_completed()
        time.sleep(1)

        self.step_num = 0
        self.degrees_rotate = degrees_rotate
        self.render = render

        self.config = {
            'grayscale': False,
            'resolution': IMAGE_DIM_INPUT
        }

        channels = 1 if self.config['grayscale'] else 3
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(channels, self.config['resolution'][0],
                                                   self.config['resolution'][1]))#,
                                            # dtype=np.uint8)
        self.action_space = spaces.Discrete(2)

        self.natural_language = natural_language
        if self.natural_language:
            # self.train_instructions = ['microwave', 'cup'] # todo begin simple
            self.nlp_instructions = [
                'cup',
                'bowl'
                # 'Look at the green cup',
                # 'Look at the yellow banana'  # hmm
            ]
            self.nlp_instruction_idx = random.randint(0, len(self.nlp_instructions) - 1)
            self.word_to_idx = self.get_word_to_idx()
            self.instruction = self.nlp_instructions[self.nlp_instruction_idx]

            self.object_class_to_id_mapping = {
                'bowl': 45,
                'cup': 41
            }

            # always last word of the sentence
            self.current_object_type = self.instruction.split(' ')[-1]
            self.current_object_idx = self.object_class_to_id_mapping[self.current_object_type]
            print('Current instruction: {}. object type (last word in sentence): {} to object YOLO index: {}'.format(
                    self.instruction, self.current_object_type, self.current_object_idx))
        else:
            self.current_object_type = 'cup'
            self.current_object_idx = 41

    def reset(self, random_rotate=True):
        """
        Rotate/put cozmo back to some random or specific position
        """

        print('Resetting environment')
        if random_rotate and not self.natural_language:
            random_degrees = random.randint(-180, 180)
            # Make sure turn is big enough
            if abs(random_degrees) < 35:
                if random_degrees < 0:
                    random_degrees = -35
                else:
                    random_degrees = 35
            print('rotating Cozmo {} degrees'.format(random_degrees))
            self.robot.turn_in_place(degrees(random_degrees)).wait_for_completed()
        else:
            self.robot.go_to_pose(Pose(0, 0, 0, angle_z=degrees(0.0)), relative_to_robot=False).wait_for_completed()
            # self.last_reset_pose = self.robot.pose
            # needed to go back to the same starting position
            # todo possible reset: self.robot.go_to_pose(Pose(100, 100, 0, angle_z=degrees(45)),  self.robot.go_to_pose(Pose(0, 0, 0, angle_z=0), relative_to_robot=True)
            # relative_to_robot=True).wait_for_completed() random is better for some other tasks

        self.step_num = 0
        state = self.get_raw_image()
        state_preprocessed = cv2.resize(state, IMAGE_DIM_INPUT)
        state_preprocessed = np.moveaxis(state_preprocessed, 2, 0)
        if self.natural_language:
            self.nlp_instruction_idx = random.randint(0, len(self.nlp_instructions) - 1)
            self.instruction = self.nlp_instructions[self.nlp_instruction_idx]

            # always last word of the sentence
            self.current_object_type = self.instruction.split(' ')[-1]
            self.current_object_idx = self.object_class_to_id_mapping[self.current_object_type]
            print('Current instruction: {}. object type (last word in sentence): {} to object YOLO index: {}'.format(
                    self.instruction, self.current_object_type, self.current_object_idx))

            self.robot.say_text("{}".format(self.instruction)).wait_for_completed()

            state_preprocessed = (state_preprocessed, self.instruction)
        return state_preprocessed

    def target_object_in_centre_of_screen(self, raw_img, dead_centre=True):
        """
        Use YOLOv3 PyTorch
        image is 416x416. centre is 208, 208. Solid centre of screen object is around ~?
        108 - 308? hard to get bowl in centre. maybe only for cup
        88 - 322
        122-288: 80 range in middle of screen with midpoints
        raw_img: numpy array
        """

        detections = detect_on_numpy_img(raw_img)
        if detections is not None:
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                if dead_centre:
                    # if int(cls_pred.item()) == self.current_object_idx and x1.item() > 88 and x2.item() < 322:  # roughly in middle
                    midpoint_x = (x1.item() + x2.item()) / 2.0  # better and more accurate
                    if midpoint_x > 108 and midpoint_x < 288:  # roughly in middle

                        if int(cls_pred.item()) == self.current_object_idx:
                            return 10
                        else:
                            # if wrong object looked at
                            return -5
                else:  # much easier, just have to see cup
                    if int(cls_pred.item()) == self.current_object_idx:
                        return True
        return 0

    def get_reward(self, img, movement_reward=-0.01):  # todo maybe env variable
        reward = movement_reward
        object_in_centre = self.target_object_in_centre_of_screen(img)
        if self.step_num >= STEPS_IN_EPISODE - 1:
            done = True
            reward = -10
        elif object_in_centre > 0:
            done = True
            reward = 10
            print('Stared at object type: {} correctly. Reward: {}'.format(self.current_object_type, reward))
            self.robot.say_text('yay').wait_for_completed()
        elif object_in_centre == 0:
            done = False
        else:
            reward = object_in_centre
            done = True
            print('Looked at wrong object. Reward: {}'.format(reward))
            self.robot.say_text('awww').wait_for_completed()

        return reward, done

    def get_raw_image(self, resize=False, size=IMAGE_DIM_INPUT, return_PIL=False):
        """

        :param resize: whether to resize using PIL
        :param size: tuple e.g. (160, 80)
        :param return_PIL: whether to return PIL image
        :return: numpy array raw image in shape (240, 320, 3)
        """
        image = self.robot.world.latest_image
        # if _display_debug_annotations != DEBUG_ANNOTATIONS_DISABLED:
        #     image = image.annotate_image(scale=2)
        # else:
        image = image.raw_image
        if resize:
            image.thumbnail(size, Image.ANTIALIAS)
        if return_PIL:
            return image
        return np.asarray(image)

    def step(self, action):
        if action == 0:
            print('Turning left')
            self.robot.turn_in_place(degrees(self.degrees_rotate)).wait_for_completed()
        elif action == 1:
            print('Turning right')
            self.robot.turn_in_place(degrees(-self.degrees_rotate)).wait_for_completed()

        raw_state = self.get_raw_image()  # todo find best place to resize
        reward, done = self.get_reward(raw_state)

        self.step_num += 1

        state_preprocessed = cv2.resize(raw_state, IMAGE_DIM_INPUT)
        state_preprocessed = np.moveaxis(state_preprocessed, 2, 0)
        if self.natural_language:
            self.instruction = self.nlp_instructions[self.nlp_instruction_idx]
            state_preprocessed = (state_preprocessed, self.instruction)
        return state_preprocessed, reward, done, None

    def get_word_to_idx(self):
        word_to_idx = {}
        for instruction_data in self.nlp_instructions:
            instruction = instruction_data # todo actual json ['instruction']
            for word in instruction.split(" "):
                if word not in word_to_idx:
                    word_to_idx[word] = len(word_to_idx)
        return word_to_idx
