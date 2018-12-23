import glob
import cv2

class CustomPictureFolderEnv():
    def __init__(self, picture_folder_path='pictures/*', terminal_picture_idx=8):
        self.state_picture_index = 0
        self.all_pictures = self.read_all_pictures_states(picture_folder_path)
        self.terminal_picture_idx = terminal_picture_idx

        print('Number of image states: {}'.format(len(self.all_pictures)))

    def step(self, action):
        if action == 1:  # go right
            self.state_picture_index -= 1
        elif action == 0:  # go left
            self.state_picture_index += 1
        else:
            # do nothing
            pass

        if self.state_picture_index > len(self.all_pictures) - 1:
            self.state_picture_index = 0
        if self.state_picture_index < 0:
            self.state_picture_index = len(self.all_pictures) - 1

        done = True if self.state_picture_index == self.terminal_picture_idx else False
        if done:
            reward = 10
        else:
            reward = 0
        return self.all_pictures[self.state_picture_index], reward, done, None

    def reset(self):
        self.state_picture_index = 0
        return self.all_pictures[self.state_picture_index]

    def render(self):
        cv2.imshow('state image', self.all_pictures[self.state_picture_index])
        k = cv2.waitKey(1)

    def read_all_pictures_states(self, picture_folder_path):
        all_picture_paths = glob.glob(picture_folder_path)
        all_pictures = [cv2.imread(path) for path in all_picture_paths]
        # all_pictures = [cv2.resize(image, (80, 60)) for image in all_pictures]

        all_pictures_resized = []
        for idx, image in enumerate(all_pictures):
            try:
                all_pictures_resized.append(cv2.resize(image, (80, 60)))
            except Exception as e:
                print('Couldn\'t resize image with path: {}'.format(all_picture_paths[idx]))
                print(e)

        # Loop through and show all images
        for image in all_pictures:
            cv2.imshow('image', image)
            k = cv2.waitKey(100)

        cv2.destroyAllWindows()

        return all_pictures_resized

if __name__ == '__main__':
    env = CustomPictureFolderEnv(picture_folder_path='pictures/*')

    state, done = env.reset()

    while True:
        cv2.imshow('state image', state)
        k = cv2.waitKey(1)

        action = 2
        if k == ord('a'):
           print("pressed A, turned left")
           action = 0
        elif k == ord('d'):
            print('pressed D, turned right')
            action = 1
        elif k == ord('q'):
            break

        state, reward, done, _ = env.step(action=action)
