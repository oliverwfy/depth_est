import torch
import torch.nn as nn
import torch.nn.functional as F

class Gradient(nn.Module):
    def __init__(self, device=torch.device("cuda")):
        super().__init__()
        kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
        kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0).to(device)

        kernel_y = [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
        kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0).to(device)

        self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
        self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False)

    def forward(self, x):
        grad_x = F.conv2d(x, self.weight_x)
        grad_y = F.conv2d(x, self.weight_y)
        gradient = torch.abs(grad_x) + torch.abs(grad_y)
        return gradient

def gradient(x, device=torch.device("cuda")):
    gradient_model = Gradient().to(device)
    g = gradient_model(x)
    return g


def gd_l2loss(x, device=torch.device("cuda")):
    g = gradient(x, device)
    loss = torch.norm(g, p=2) ** 2
    return loss

def l1loss(x, y):
    loss = torch.norm(abs(x-y), p=1)
    return loss

def l2loss(x, y):
    loss = torch.norm(abs(x-y), p=2) ** 2
    return loss

def dis_con_loss(dis_l, dis_r):
    samples = len(dis_l)
    H, W = dis_l.size(2), dis_l.size(3)
    loss = 0
    for n in range(samples):
        for i in range(H):
            for j in range(W):
                con_j = j+dis_l[n, 0, i, j]
                if con_j < W and con_j > 0:
                    con_dis_l = dis_r[n, 0, i, int(torch.floor(con_j))]
                    loss += torch.abs(dis_l[n, 0, i, j] - con_dis_l)
    return loss
