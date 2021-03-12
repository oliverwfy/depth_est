import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, img_size=(128, 160)):
        super().__init__()

        self.img_size = img_size
        # pool and unpool layers
        self.pool = nn.MaxPool2d(2, stride=2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2, stride=2)

        # conv block 1
        self.conv1 = conv_block(1)

        # conv block 2
        self.conv2 = conv_block(2)

        # conv block 3
        self.conv3 = conv_block(3)

        # conv block 4
        self.conv4 = conv_block(4)

        # conv block 5
        self.conv5 = conv_block(5)

        # mid conv layer
        self.conv6 = nn.Conv2d(512, 512, 3, 1, padding=1)

        # deconv block 1
        self.deconv1 = deconv_block(1)

        # deconv block 2
        self.deconv2 = deconv_block(2)

        # deconv block 3
        self.deconv3 = deconv_block(3)

        # deconv block 4
        self.deconv4 = deconv_block(4)

        # deconv block 5
        self.deconv5 = deconv_block(5)
        # indices of pooling
        self.pool_indices = {}

        # last conv layer
        self.conv0 = nn.Conv2d(3, 1, 3, 1, padding=1)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, kernel_size=5),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
        )

        self.linear_input_shape = int()
        x = torch.randn(1, 1, *self.img_size)
        self.stn_conv(x)
        # regression for the 2 * 3 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(self.linear_input_shape, 256),
            nn.ReLU(True),
            nn.Linear(256, 2 * 3)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))


    def autoencoder(self, x):
        x = self.conv1(x)
        x, pool_indice_1 = self.pool(x)
        self.pool_indices["pool_1"] = pool_indice_1

        x = self.conv2(x)
        x, pool_indice_2 = self.pool(x)
        self.pool_indices["pool_2"] = pool_indice_2

        x = self.conv3(x)
        x, pool_indice_3 = self.pool(x)
        self.pool_indices["pool_3"] = pool_indice_3

        x = self.conv4(x)
        x, pool_indice_4 = self.pool(x)
        self.pool_indices["pool_4"] = pool_indice_4

        x = self.conv5(x)
        x, pool_indice_5 = self.pool(x)
        self.pool_indices["pool_5"] = pool_indice_5

        x = self.conv6(x)

        x = self.unpool(x, self.pool_indices["pool_5"])
        x = self.deconv5(x)

        x = self.unpool(x, self.pool_indices["pool_4"])
        x = self.deconv4(x)

        x = self.unpool(x, self.pool_indices["pool_3"])
        x = self.deconv3(x)

        x = self.unpool(x, self.pool_indices["pool_2"])
        x = self.deconv2(x)

        x = self.unpool(x, self.pool_indices["pool_1"])
        x = self.deconv1(x)

        x = self.conv0(x)
        return x

    def stn_conv(self, x):
        x = self.localization(x)
        if self.linear_input_shape == 0:
            self.linear_input_shape = x.shape[1]*x.shape[2]*x.shape[3]
        return x

    def loc_net(self, x):
        x = self.stn_conv(x)
        x = x.view(-1, self.linear_input_shape)
        theta = self.fc_loc(x)
        theta = theta.view(-1, 2, 3)
        return theta

    def sampler(self, theta, y):
        grid = F.affine_grid(theta, y.size(), align_corners=False)
        x = F.grid_sample(y, grid, align_corners=False)
        return x

    def stn(self, dis, y):
        theta = self.loc_net(dis)
        est_x = self.sampler(theta, y)
        return est_x

    def forward(self, x, y):
        x = self.autoencoder(x)
        theta = self.loc_net(x)
        estimated_x = self.sampler(theta, y)

        return estimated_x


def conv_layer(inputs, outputs, kernel_size=3, stride=1, relu=True):
    conv2d = nn.Sequential(
        nn.Conv2d(inputs, outputs, kernel_size, stride, padding=1),
        nn.BatchNorm2d(outputs),
        nn.ReLU(relu),
    )
    return conv2d

def deconv_layer(inputs, outputs, kernel_size=3, stride=1, relu=True):
    deconv2d = nn.Sequential(
        nn.ConvTranspose2d(inputs, outputs, kernel_size, stride, padding=1),
        nn.BatchNorm2d(outputs),
        nn.ReLU(relu),
    )
    return deconv2d

def conv_block(num):
    block = None
    if num == 1:
        block = nn.Sequential(
            *conv_layer(3, 64).children(),
            *conv_layer(64, 64).children()
        )
    elif num == 2:
        block = nn.Sequential(
            *conv_layer(64, 128).children(),
            *conv_layer(128, 128).children()
        )
    elif num == 3:
        block = nn.Sequential(
            *conv_layer(128, 256).children(),
            *conv_layer(256, 256).children(),
            *conv_layer(256, 256).children()
        )
    elif num == 4:
        block = nn.Sequential(
            *conv_layer(256, 512).children(),
            *conv_layer(512, 512).children(),
            *conv_layer(512, 512).children()
        )
    elif num == 5:
        block = nn.Sequential(
            *conv_layer(512, 512).children(),
            *conv_layer(512, 512).children(),
            *conv_layer(512, 512).children()
        )
    return block

def deconv_block(num):
    block = None
    if num == 1:
        block = nn.Sequential(
            *deconv_layer(64, 64).children(),
            *deconv_layer(64, 3).children()
        )
    elif num == 2:
        block = nn.Sequential(
            *deconv_layer(128, 128).children(),
            *deconv_layer(128, 64).children()
        )
    elif num == 3:
        block = nn.Sequential(
            *deconv_layer(256, 256).children(),
            *deconv_layer(256, 256).children(),
            *deconv_layer(256, 128).children(),
        )
    elif num == 4:
        block = nn.Sequential(
            *deconv_layer(512, 512).children(),
            *deconv_layer(512, 512).children(),
            *deconv_layer(512, 256).children(),
        )
    elif num == 5:
        block = nn.Sequential(
            *deconv_layer(512, 512).children(),
            *deconv_layer(512, 512).children(),
            *deconv_layer(512, 512).children()
        )
    return block



