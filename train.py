from Trainer import Trainer_siamese
from Net import Net
from show_image import  return_outputs

data_path = "/home/ubuntu/fyp/data/Video3_rect/128_160/training_set3_128_160_norm.pt"
model = Net(img_size=(128, 160))
trainer = Trainer_siamese(model, data_path, epochs = 20, batch_size=32, device="gpu")
model_name, model_log = trainer.train()

dis = return_outputs(model_name, model_log, data_path, num=0, dis=True, est=True, return_dis=True)
