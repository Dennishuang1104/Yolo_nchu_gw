# 在 models/panet.py 中定義 PANet 類別

import torch
import torch.nn as nn

class PANet(nn.Module):
    def __init__(self, in_channels):
        super(PANet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.concat = nn.Identity()  # 可以用來處理多層級的特徵融合

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.concat(x)  # 可以進行特徵融合的操作
        return x
