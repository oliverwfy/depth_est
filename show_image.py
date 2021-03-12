import os
from Net import Net
from utils import *

def return_outputs(model_name, model_log, data_path, num=0, dis=True, est=True, return_dis=True):
    learning_curve(model_log, model_name)

    model_path = os.path.join("/home/ubuntu/fyp/model_dict", model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print("running on gpu")
    else:
        print(f"running on {device.type}")
    model = Net().to(device)

    model.load_state_dict(torch.load(model_path))
    training_set = torch.load(data_path)

    dis = show_outputs(training_set, model, device, num=num, dis=dis, est=est, return_dis=return_dis)
    if return_dis:
        return dis
