# Affected by inception module
# utilize 3 different kernel_size with 3x3 maxpooling to extract feature
# if the output size is different among these four output, use padding='same', stride=1 to remain the same size
# after improvement, the improved inception module will add 1x1 convolution layer first to reduce the output channel then connect to the original layer
# maxpooling layer cannot reduce the number of channel, thus add 1x1 convolution layer after maxpooling layer

import torch.nn as nn
import torch
import torch.functional as F

class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels1, out_channels2_step1, out_channels2_step2, out_channels3_step1, out_channels3_step2, out_channels4):
        super(InceptionModule, self).__init__()
        # 1x1 conv
        self.branch1_conv = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels1, kernel_size=1),
                                        nn.ReLU(inplace=True))
        # 1x1 conv + 3x3 conv
        self.branch2_conv1 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels2_step1, kernel_size=1),
                                            nn.ReLU(inplace=True))
        self.branch2_conv2 = nn.Sequential(nn.Conv2d(in_channels=out_channels2_step1, out_channels=out_channels2_step2, kernel_size=3, padding=1),
                                            nn.ReLU(inplace=True))
        
        # 1x1 conv + 5x5 conv
        self.branch3_conv1 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels3_step1, kernel_size=1),
                            nn.ReLU(inplace=True))
        self.branch3_conv2 = nn.Sequential(nn.Conv2d(in_channels=out_channels3_step1, out_channels=out_channels3_step2, kernel_size=5, padding=2),
                            nn.ReLU(inplace=True))
        # max pooling + 1x1 conv
        self.branch4_maxpooling = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.branch4_conv1 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels4, kernel_size=1),
                            nn.ReLU(inplace=True))
        
    def forward(self, x):
        out1 = self.branch1_conv(x)
        out2 = self.branch2_conv2(self.branch2_conv1(x))
        out3 = self.branch3_conv2(self.branch3_conv1(x))
        out4 = self.branch4_conv1(self.branch4_maxpooling(x))
        out = torch.cat([out1, out2, out3, out4], dim=1)

        return out

class AuxiliaryClassifiers(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AuxiliaryClassifiers, self).__init__()
        self.avgpooling = nn.AvgPool2d(kernel_size=5, stride=3)
        
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=1)
        
        self.fc1 = nn.Linear(in_features=128*4*4, out_features=1024)

        self.fc2 = nn.Linear(in_features=1024, out_features=out_channels)
     
    def forward(self, x):
        x = self.avgpooling(x)
        x = F.relu(self.conv(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5)
        x = self.fc2(x)

        return x

class InceptionV1(nn.Module):
    def __init__(self, num_classes, training=True):
        super(InceptionV1, self).__init__()
        self.training = training
        self.conv = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
                                    nn.ReLU(inplace=True),
                                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(inplace=True),
                                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        
        self.inception1 = InceptionModule(in_channels=192, out_channels1=64, out_channels2_step1=96, out_channels2_step2=128, out_channels3_step1=16, out_channels3_step2=32, out_channels4=32)
        self.inception2 = InceptionModule(in_channels=256, out_channels1=128, out_channels2_step1=128, out_channels2_step2=192, out_channels3_step1=32, out_channels3_step2=96, out_channels4=64)
        self.maxpooling1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception3 = InceptionModule(in_channels=480, out_channels1=192, out_channels2_step1=96, out_channels2_step2=208, out_channels3_step1=16, out_channels3_step2=48, out_channels4=64)

        if self.training == True:
            self.auxiliary1 = AuxiliaryClassifiers(in_channels=512, out_channels=num_classes)

        self.inception4 = InceptionModule(in_channels=512, out_channels1=160, out_channels2_step1=112, out_channels2_step2=224, out_channels3_step1=24, out_channels3_step2=64, out_channels4=64)
        self.inception5 = InceptionModule(in_channels=512, out_channels1=128, out_channels2_step1=128, out_channels2_step2=256, out_channels3_step1=24, out_channels3_step2=64, out_channels4=64)
        self.inception6 = InceptionModule(in_channels=512, out_channels1=112, out_channels2_step1=144, out_channels2_step2=288, out_channels3_step1=32, out_channels3_step2=64, out_channels4=64)

        if self.training == True:
            self.auxiliary2 = AuxiliaryClassifiers(in_channels=528, out_channels=num_classes)

        self.inception7 = InceptionModule(in_channels=528, out_channels1=256, out_channels2_step1=160, out_channels2_step2=320, out_channels3_step1=32, out_channels3_step2=128, out_channels4=128)
        self.maxpooling2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception8 = InceptionModule(in_channels=832, out_channels1=256, out_channels2_step1=160, out_channels2_step2=320, out_channels3_step1=32, out_channels3_step2=128, out_channels4=128)
        self.inception9 = InceptionModule(in_channels=832, out_channels1=384, out_channels2_step1=192, out_channels2_step2=384, out_channels3_step1=48, out_channels3_step2=128, out_channels4=128)

        self.avgpooling = nn.AvgPool2d(kernel_size=7,stride=1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(in_features=1024,out_features=num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.maxpooling1(x)
        x = self.inception3(x)
        aux1 = self.auxiliary1(x)
        x = self.inception4(x)
        x = self.inception5(x)
        x = self.inception6(x)
        aux2 = self.auxiliary2(x)
        x = self.inception7(x)
        x = self.maxpooling2(x)
        x = self.inception8(x)
        x = self.inception9(x)
        x = self.avgpooling(x)
        x = self.dropout(x)
        x = torch.flatten(x, start_dim=1)
        out = self.fc(x)

        if self.training == True:
            return aux1, aux2, out

        else:
            return out