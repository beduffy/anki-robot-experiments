import cv2
from PIL import Image
import cozmo
from cozmo.util import degrees, distance_mm, speed_mmps
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
    def __init__(self, robot, natural_language=False):
        self.robot = robot
        self.robot = self.robot.wait_for_robot()
        self.robot.enable_device_imu(True, True, True)
        # Turn on image receiving by the camera
        self.robot.camera.image_stream_enabled = True

        self.robot.set_lift_height(0.0).wait_for_completed()
        self.robot.set_head_angle(degrees(-.0)).wait_for_completed()
        time.sleep(1)

        self.step_num = 0

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
                'Look at the green cup',
                'Look at the yellow banana'  # hmm
            ]
            self.nlp_instruction_idx = random.randint(0, len(self.nlp_instructions) - 1)
            self.word_to_idx = self.get_word_to_idx()

            self.current_object_type = 'Microwave' if self.nlp_instruction_idx == 0 else 'Mug'

    def reset(self):
        """
        Rotate cozmo random amount
        """
        # todo needed to go back to the same starting position
        # todo possible reset: self.robot.go_to_pose(Pose(100, 100, 0, angle_z=degrees(45)), relative_to_robot=True).wait_for_completed() random is better. Not for other tasks
        random_degrees = random.randint(-180, 180)
        # Make sure turn is big enough
        if abs(random_degrees) < 35:
            if random_degrees < 0:
                random_degrees = -35
            else:
                random_degrees = 35
        self.robot.turn_in_place(degrees(random_degrees)).wait_for_completed()
        print('Resetting environment, rotating Cozmo {} degrees'.format(random_degrees))
        self.step_num = 0
        state = self.get_annotated_image()
        state_preprocessed = cv2.resize(state, IMAGE_DIM_INPUT)
        state_preprocessed = np.moveaxis(state_preprocessed, 2, 0)
        if self.natural_language:
            self.nlp_instruction_idx = random.randint(0, len(self.nlp_instructions) - 1)
            instruction = self.nlp_instructions[self.nlp_instruction_idx]
            print('New instruction: {}'.format(instruction))
            state_preprocessed = (state_preprocessed, instruction)
        return state_preprocessed

    def cup_in_middle_of_screen(self, img, dead_centre=True):
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

    def get_reward(self, img, movement_reward=-0.01):
        reward = movement_reward
        if self.step_num >= STEPS_IN_EPISODE - 1:
            done = True
            reward = -10
        elif self.cup_in_middle_of_screen(img):
            done = True
            reward = 10
        else:
            done = False

        return reward, done

    def get_annotated_image(self, resize=False, size=IMAGE_DIM_INPUT, return_PIL=False):
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
            print('Turning right')
            self.robot.turn_in_place(degrees(-NUM_DEGREES_ROTATE)).wait_for_completed()
        elif action == 1:
            print('Turning left')
            self.robot.turn_in_place(degrees(NUM_DEGREES_ROTATE)).wait_for_completed()

        state = self.get_annotated_image()  # todo find best place to resize
        reward, done = self.get_reward(state)

        self.step_num += 1

        state_preprocessed = cv2.resize(state, IMAGE_DIM_INPUT)
        state_preprocessed = np.moveaxis(state_preprocessed, 2, 0)
        if self.natural_language:
            instruction = self.nlp_instructions[self.nlp_instruction_idx]
            state_preprocessed = (state_preprocessed, instruction)
        return state_preprocessed, reward, done, None

    def get_word_to_idx(self):
        word_to_idx = {}
        for instruction_data in self.nlp_instructions:
            instruction = instruction_data # todo actual json ['instruction']
            for word in instruction.split(" "):
                if word not in word_to_idx:
                    word_to_idx[word] = len(word_to_idx)
        return word_to_idx
