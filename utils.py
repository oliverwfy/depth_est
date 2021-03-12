import matplotlib.pyplot as plt
import numpy as np
import torch

def show_img(img_pt, uint8=True):
    img = img_pt.numpy().transpose([1, 2, 0])
    if uint8:
        img.astype(np.uint8)
    plt.imshow(img)
    plt.show()

def learning_curve(model_log, model_name):
    contents = open(model_log, "r").read().split("\n")
    times = []
    losses_1 = []
    losses_2 = []
    losses_3 = []
    ids = []
    for c in contents:
        if model_name in c:
            name, time_stamp, idx, loss1, loss2, loss3 = c.split(",")
            times.append(float(time_stamp))
            losses_1.append(float(loss1))
            losses_2.append(float(loss2))
            losses_3.append(float(loss3))
            ids.append(int(idx))
    plt.plot(times, losses_1, label="reconstruction")
    plt.plot(times, losses_2, label="smooth")
    plt.plot(times, losses_3, label="consistency")
    plt.legend()
    plt.show()

def split_data(training_set, num=0):
        l, r = training_set[num:num+1, 0:3, :, :], training_set[num:num+1, 3:, :, :]
        return (l, r)

def show_outputs(training_set, model=None, device=None, num=0, dis=True, est=True, return_dis=True):

    with torch.no_grad():
        if num < len(training_set):
            l, r = split_data(training_set, num)
            img_l = l.numpy().squeeze(0).transpose((1, 2, 0))
            img_r = r.numpy().squeeze(0).transpose((1, 2, 0))
            norm = not (np.mean(img_l) > 0.5)
            if norm:
                img_l = (img_l * 255 + 128) / 255.
                img_r = (img_r * 255 + 128) / 255.
            else:
                img_l /= 255.
                img_r /= 255.
            plt.figure()
            ax1 = plt.subplot2grid((2, 2), (0, 0))
            ax2 = plt.subplot2grid((2, 2), (0, 1))
            ax3 = plt.subplot2grid((2, 2), (1, 0))
            ax4 = plt.subplot2grid((2, 2), (1, 1))
            ax1.imshow(img_l)
            ax1.set_title("frame_l")
            ax2.imshow(img_r)
            ax2.set_title("frame_r")
            if model:
                model.eval()
                if est:
                    pre_l = model(l.to(device), r.to(device))
                    img_pre_l = pre_l.cpu().numpy().squeeze(0).transpose((1, 2, 0))
                    if norm:
                        img_pre_l = np.clip((img_pre_l * 255 + 128) / 255., 0, 1)
                    else:
                        img_pre_l /= 255.
                    ax3.imshow(img_pre_l)
                    ax3.set_title("esti_l")
                if dis:
                    pre_dis = model.autoencoder(l.to(device))
                    img_dis = pre_dis.cpu().numpy().squeeze(0).transpose((1, 2, 0)).squeeze(-1)

                    ax4.imshow(img_dis, cmap="gray")
                    ax4.set_title("disparity (l)")

        else:
            print(f"The training set only contains {len(training_set)} samples.")
        plt.tight_layout()
        plt.show()
        if return_dis:
            return img_dis