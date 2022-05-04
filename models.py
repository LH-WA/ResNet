import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as func
from utils import params_setting

args = params_setting()


def RELU(x):
    if args.activation == "ReLU":
        return func.relu(x, inplace=False)
    elif args.activation == "ReLU6":
        return func.relu6(x, inplace=False)
    elif args.activation == "LRU":
        return func.leaky_relu(x, negative_slope=args.activation_coe, inplace=False)
    elif args.activation == "ELU":
        return func.elu(x, alpha=args.activation_coe, inplace=False)
    else:
        return x * torch.sigmoid(args.activation_coe * x)


class BasicBlock(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, skip_kernel, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=kernel_size, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.shortcut = nn.Sequential()

        if stride != 1 or ch_in != ch_out:  # element-wise add: [b, ch_in, h, w] -> [b, ch_out, h, w]
            self.shortcut = nn.Sequential(  # use 1*1 convolution to match dimension
                nn.Conv2d(ch_in, ch_out, kernel_size=skip_kernel, stride=stride, bias=False),
                nn.BatchNorm2d(ch_out))

    def forward(self, x):
        # Type_a
        out = RELU(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = RELU(out)

        # Type_b
        # out = RELU(self.bn1(self.conv1(x)))
        # out = self.conv2(out)
        # out += self.shortcut(x)
        # self.bn2(out)
        # out = RELU(out)

        # Type_c
        # out = RELU(self.bn1(self.conv1(x)))
        # out = self.bn2(self.conv2(out))
        # out = RELU(out)
        # out += self.shortcut(x)

        # Type_d
        # out = self.bn1(self.conv1(RELU(x)))
        # out = self.bn2(self.conv2(RELU(out)))
        # out += self.shortcut(x)

        # out = self.conv1(RELU(x))
        # nn.Dropout(0.5)
        # out = self.conv2(RELU(out))
        # # nn.Dropout(0.5)
        # out += self.shortcut(x)

        return out


class ResNet(nn.Module):
    def __init__(self, N: int, B: list, C: list, F: list, K: list, P: int, num_classes=10):
        super(ResNet, self).__init__()
        self.ch_in = C[0]
        self.block = BasicBlock
        self.P = P  # Average pool kernel size
        self.layers = []  # layers container
        self.S = [2] * N  # strides for layers
        self.S[0] = 1

        self.outLayerInSize = C[N - 1] * (32 // (P * 2 ** (N - 1))) ** 2
        self.conv1 = nn.Conv2d(3, C[0], kernel_size=F[0], stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(C[0])
        for i in range(N):
            exec("self.layer{} = self._make_layer(self.block, C[{}], B[{}], F[{}], K[{}], self.S[{}])" \
                 .format(i + 1, i, i, i, i, i))
            exec("self.layers.append(self.layer{})".format(i + 1))

        self.linear = nn.Linear(self.outLayerInSize, num_classes)
        self._init_parameters()

    def _make_layer(self, block, ch_out, num_blocks, kernel_size, skip_kernel, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.ch_in, ch_out, kernel_size, skip_kernel, stride))
            self.ch_in = ch_out
        return nn.Sequential(*layers)

    def forward(self, x):
        out = RELU(self.bn1(self.conv1(x)))
        for layer in self.layers:
            out = layer(out)
        out = func.avg_pool2d(out, self.P)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def _init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.constant_(m.weight, 0)
                # nn.init.xavier_normal_(m.weight)
                # nn.init.uniform_(m.weight, 0, 1)
                # nn.init.normal_(m.weight, mean=0, std=1)
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")


# N: : # Residual Layers
# Bi : # Residual blocks in Residual Layer i
# Ci : # channels in Residual Layer i
# Fi : Conv. kernel size in Residual Layer i
# Ki : Skip connection kernel size in Residual Layer i
# P  : Average pool kernel size


def ResNet10():
    B = [1, 1, 1, 1]
    C = [64, 128, 256, 512]
    F = [3, 3, 3, 3]
    K = [1, 1, 1, 1]
    return ResNet(len(B), B, C, F, K, P=4)


def ResNet16():
    B = [2, 1, 2, 2]
    C = [64, 128, 256, 256]
    F = [3, 3, 3, 3]
    K = [1, 1, 1, 1]
    return ResNet(len(B), B, C, F, K, P=4)


def ResNet24():
    B = [3, 3, 2, 3]
    C = [64, 128, 128, 256]
    F = [3, 3, 3, 3]
    K = [1, 1, 1, 1]
    return ResNet(len(B), B, C, F, K, P=4)


def ResNet48():
    B = [8, 5, 5, 5]
    C = [64, 128, 128, 128]
    F = [3, 3, 3, 3]
    K = [1, 1, 1, 1]
    return ResNet(len(B), B, C, F, K, P=4)
