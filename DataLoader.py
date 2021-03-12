import cv2
import numpy as np
import os
import torch

class DataLoader():
    def __init__(self, data_dir, resize):
        self.data_dir = data_dir
        self.resize = resize
        self.H = resize[1]
        self.W = resize[0]
        self.l_dir = os.path.join(self.data_dir, "Left")
        self.r_dir = os.path.join(self.data_dir, "Right")
        self.co_dir = os.path.join(self.data_dir, "compare")
        self.frame_info = (os.listdir(self.l_dir), os.listdir(self.r_dir))

    def get_image(self, frame_path):
        frame = cv2.imread(frame_path)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.resize:
            frame_re = cv2.resize(frame_rgb, self.resize)
            return frame_re
        return frame_rgb

    def sorted_list(self, img_list):
        sample = len(img_list)
        print(f"{sample} samples in total")
        co_list = []

        for i in range(sample):
            idx_str = list(filter(str.isdigit,img_list[i]))
            idx = int("".join(idx_str))
            co_list.append((img_list[i], idx))
        # unsorted_list
        un_dict = dict(co_list)
        return sorted(un_dict,key=un_dict.__getitem__)

    def save_training_set(self, path):
        l_list, r_list = self.frame_info
        print(f"Data loaded from {self.l_dir}")
        l_ls_sorted = self.sorted_list(l_list)
        print(f"Data loaded from {self.r_dir}")
        r_ls_sorted = self.sorted_list(r_list)
        sample = len(r_list)
        co_list = [(l_ls_sorted[n], r_ls_sorted[n]) for n in range(sample)]
        l_train = np.ndarray(shape=(sample, self.H, self.W, 3), dtype="uint8")
        r_train = np.ndarray(shape=(sample, self.H, self.W, 3), dtype="uint8")
        for i in range(sample):
            l_name = co_list[i][0]
            r_name = co_list[i][1]
            l_frame = self.get_image(os.path.join(self.l_dir, l_name))
            r_frame = self.get_image(os.path.join(self.r_dir, r_name))
            l_train[i] = l_frame
            r_train[i] = r_frame

        training_set = np.concatenate((l_train, r_train), axis = -1)
        np.save(path, training_set)
        print(f"The training data is successfully saved in {path}, "
              f"shape: {training_set.shape}, dtype:{training_set.dtype}")



def save_training_set(data_dir, saved_path, resize=None):
    if os.path.exists(data_dir):
        data_loader = DataLoader(data_dir, resize)
        if not os.path.exists(saved_path):
            data_loader.save_training_set(saved_path)
        else:
            print("An existed file in the path for saving.")
    else:
        print("The data directory is not existed.")

def save_data_pt(training_set, path):
    if not os.path.exists(path):
        training_set_pt = torch.tensor(training_set.transpose((0, 3, 1, 2)), dtype=torch.float)
        torch.save(training_set_pt, path)
    return None

def save_data_np(data_dir, saved_path, resize=None):
    # load and save training data
    if os.path.exists(data_dir):
        if not os.path.exists(saved_path):
            save_training_set(data_dir, saved_path, resize)
