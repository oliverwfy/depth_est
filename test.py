import torch
import matplotlib.pyplot as plt
import numpy as np
from utils import *
from Net import Net

training_set = torch.load("/home/ubuntu/fyp/data/Video3_rect/128_160/training_set3_128_160.pt")
device = torch.device("cpu")
model = Net().to(device)
