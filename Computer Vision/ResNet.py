import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


CUDA = torch.cuda.is_available()
device = torch.device("cuda" if CUDA else "cpu")
print(device)


class ResidualBlock(nn.Module):  # ResNet 18, 34
    # 輸出通道乘的倍數
    expansion = 1

    def __init__(self, in_channels=None, out_channels=None, stride=None, downsample=None):
        super(ResidualBlock, self).__init__()      
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 在 shortcut 時，若維度不一樣，要更改維度
        self.downsample = downsample 

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class BottleNeck(nn.Module): # ResNet 50, 101, 152 layers
    # 輸出通道乘的倍數
    expansion = 4

    def __init__(self, in_channels=None, out_channels=None, stride=None, downsample=None):
        super(BottleNeck, self).__init__()      
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels*self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)

        # 在 shortcut 時，若維度不一樣，要更改維度
        self.downsample = downsample 

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out   

class ResNet(nn.Module):
    def __init__(self, net_block=ResidualBlock, layers=None, num_classes=None):  # net_block = [ResidualBlock, Bottleneck]
        super(ResNet, self).__init__()

        self.input_channel = 64        
        
        self.pre_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3), # padding=3
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layey1 = self._make_layer(net_block, out_channels=64, num_blocks=layers[0])
        self.layey2 = self._make_layer(net_block, out_channels=128, num_blocks=layers[1], stride=2)
        self.layey3 = self._make_layer(net_block, out_channels=256, num_blocks=layers[2], stride=2)
        self.layey4 = self._make_layer(net_block, out_channels=512, num_blocks=layers[3], stride=2)
        
        # final 
        self.avgpooling = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * net_block.expansion, num_classes) # modify the output layer if needed

    def _make_layer(self, net_block=None, out_channels=None, stride=1, num_blocks=None):
        # if no dimension changes, then downsample=None
        downsample = None

        if ((stride != 1) or (self.input_channel != out_channels*net_block.expansion)):  # block.expansion
            downsample = nn.Sequential(
                nn.Conv2d(self.input_channel, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        layers = []

        layers.append(net_block(self.input_channel, out_channels, stride, downsample))
        
        self.input_channel = out_channels * net_block.expansion

        for _ in range(1, num_blocks):
            layers.append(net_block(in_channels=self.input_channel, out_channels=out_channels, stride=stride, downsample=None))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre_layer(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpooling(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x

        

def resnet18(num_classes):
    model = ResNet(net_block=ResidualBlock, layers=[2, 2, 2, 2], num_classes=num_classes)
    return model


def resnet34(num_classes):
    model = ResNet(net_block=ResidualBlock, layers=[3, 4, 6, 3], num_classes=num_classes)
    return model


def resnet50(num_classes):
    model = ResNet(net_block=BottleNeck, layers=[3, 4, 6, 3], num_classes=num_classes)
    return model

"""
if CUDA:
    model = model.to(device)
"""