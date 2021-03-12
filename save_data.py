from DataLoader import *
import numpy as np

data_dir = "/home/ubuntu/fyp/data/Video3_rect"
saved_path = "/home/ubuntu/fyp/data/Video3_rect/64_80/training_set3_64_80.npy"
save_data_np(data_dir, saved_path, resize=(80, 64))

training_set_np = np.load(saved_path)
saved_path_pt = "/home/ubuntu/fyp/data/Video3_rect/64_80/training_set3_64_80.pt"
save_data_pt(training_set_np, saved_path_pt)

saved_path_pt_norm = "/home/ubuntu/fyp/data/Video3_rect/64_80/training_set3_64_80_norm.pt"
save_data_pt((training_set_np - 128.) / 255., saved_path_pt_norm)
