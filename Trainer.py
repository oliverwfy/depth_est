import torch
import torch.optim as optim
from torch.utils.data.dataset import TensorDataset
import time
import os
from Loss import gd_l2loss, l2loss, l1loss, dis_con_loss


class Trainer():

    def __init__(self, model, data_path, epochs=50, batch_size = 32, device=None):

        if device == "gpu" and torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("running on gpu")
        else:
            self.device = torch.device("cpu")
            print(f"running on {self.device.type}")
        self.epochs = epochs
        self.data_path = data_path
        self.Train_set = TensorDataset(torch.load(data_path)[0:2000])
        self.num_samples = len(self.Train_set)
        self.batch_size = batch_size
        self.train_loader = torch.utils.data.DataLoader(self.Train_set, batch_size=batch_size)
        self.model = model.to(self.device)

    def train(self, model_name=None, model_log=None, save=True):
        gamma = 0.01
        optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        if not model_name:
            model_name = f"model-{int(time.time())}"
        print(f"Model name: {model_name}")
        if not model_log:
            model_log = "/home/ubuntu/fyp/model.log"
        print(f"model.log path: {model_log}")
        self.model.train()
        with open(model_log, "a") as f:
            for epoch in range(self.epochs):
                training_loss = 0
                start = time.time()
                for i, data in enumerate(self.train_loader):
                    batch_l, batch_r = data[0][:, 0:3, :, :], data[0][:, 3:, :, :]
                    l_ = batch_l.to(self.device)
                    r_ = batch_r.to(self.device)
                    self.model.zero_grad()

                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()

                    est_dis = self.model.autoencoder(l_)
                    est_l = self.model.stn(est_dis, r_)

                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()

                    loss1 = l2loss(est_l, l_)
                    loss2 = gd_l2loss(est_dis, self.device)
                    loss = loss1 + gamma * loss2
                    loss.backward()
                    optimizer.step()
                    training_loss += float(loss)
                    scheduler.step()
                    if i % 10 == 0:
                        f.write(f"{model_name},{round(time.time(), 3)},{i},"
                                f"{round(float(loss1)/(len(self.train_loader)*10), 5)},"
                                f"{round(float(gamma * loss2)/(len(self.train_loader)*10), 5)}\n")
                print(f"Train epoch:{epoch+1} loss:{round(training_loss/self.num_samples, 5)} "
                      f"time:{round(time.time() - start, 3)} s")
        print(f"Training finished.")
        if save:
            path = "/home/ubuntu/fyp/model_dict"
            torch.save(self.model.state_dict(), os.path.join(path, model_name))
        return model_name, model_log


class Trainer_siamese():

    def __init__(self, model, data_path, epochs=50, batch_size = 32, device=None):

        if device == "gpu" and torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("running on gpu")
        else:
            self.device = torch.device("cpu")
            print(f"running on {self.device.type}")
        self.epochs = epochs
        self.data_path = data_path
        self.Train_set = TensorDataset(torch.load(data_path)[0:2000])
        self.num_samples = len(self.Train_set)
        self.batch_size = batch_size
        self.train_loader = torch.utils.data.DataLoader(self.Train_set, batch_size=batch_size)
        self.model = model.to(self.device)

    def train(self, model_name=None, model_log=None, save=True):
        gamma_gd = 0.1
        gamma_con = 0.5
        optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        if not model_name:
            model_name = f"model-{int(time.time())}"
        print(f"Model name: {model_name}")
        if not model_log:
            model_log = "/home/ubuntu/fyp/model.log"
        print(f"model.log path: {model_log}")
        self.model.train()
        with open(model_log, "a") as f:
            for epoch in range(self.epochs):
                training_loss = 0
                start = time.time()
                for i, data in enumerate(self.train_loader):
                    batch_l, batch_r = data[0][:, 0:3, :, :], data[0][:, 3:, :, :]
                    l_ = batch_l.to(self.device)
                    r_ = batch_r.to(self.device)
                    self.model.zero_grad()

                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()

                    est_dis_l = self.model.autoencoder(l_)
                    est_l = self.model.stn(est_dis_l, r_)
                    est_dis_r = self.model.autoencoder(r_)
                    est_r = self.model.stn(est_dis_r, l_)

                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()

                    rec_loss_l = l1loss(est_l, l_)
                    rec_loss_r = l1loss(est_r, r_)
                    rec_loss = (rec_loss_l + rec_loss_r)
                    gd_loss_l = gd_l2loss(est_dis_l, self.device)
                    gd_loss_r = gd_l2loss(est_dis_r, self.device)
                    gd_loss = (gd_loss_l + gd_loss_r) / 2
                    dc_loss = dis_con_loss(est_l, est_r)
                    loss = rec_loss + gamma_gd * gd_loss + gamma_con * dc_loss
                    loss.backward()
                    optimizer.step()
                    training_loss += float(loss)
                    scheduler.step()
                    if i % 10 == 0:
                        f.write(f"{model_name},{round(time.time(), 3)},{i},"
                                f"{round(float(rec_loss)/(len(self.train_loader)*10), 5)},"
                                f"{round(float(gamma_gd * gd_loss)/(len(self.train_loader)*10), 5)},"
                                f"{round(float(gamma_con * dc_loss)/(len(self.train_loader)*10), 5)}\n")
                print(f"Train epoch:{epoch+1} loss:{round(training_loss/self.num_samples, 5)} "
                      f"time:{round(time.time() - start, 3)} s")
        print(f"Training finished.")
        if save:
            path = "/home/ubuntu/fyp/model_dict"
            torch.save(self.model.state_dict(), os.path.join(path, model_name))
        return model_name, model_log

