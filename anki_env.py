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
IMAGE_DIM_INPUT = (80, 60)


class AnkiEnv(gym.Env):
    def __init__(self, robot):
        self.robot = robot
        self.robot = self.robot.wait_for_robot()
        self.robot.enable_device_imu(True, True, True)
        # Turn on image receiving by the camera
        self.robot.camera.image_stream_enabled = True

        self.robot.set_lift_height(0.0).wait_for_completed()
        self.robot.set_head_angle(degrees(-.0)).wait_for_completed()
        time.sleep(1)

        self.step_num = 0

    def reset(self):
        """
        Rotate cozmo random amount
        """
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
        return state

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

        return state, reward, done, None
