import torch
from torch import nn
from torchvision.transforms import CenterCrop
import matplotlib.pyplot as plt

class Unet(torch.nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        #activation
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax2d()

        #first block
        self.conv1 = nn.Conv2d(channels, 64, (3, 3))
        self.conv2 = nn.Conv2d(64, 64, (3, 3))
        #first max pool
        self.maxpool3 = nn.MaxPool2d((2,2))

        #second block
        self.conv4 = nn.Conv2d(64, 128, (3,3))
        self.conv5 = nn.Conv2d(128, 128, (3,3))
        #second max pool
        self.maxpool6 = nn.MaxPool2d((2,2))

        #third block
        self.conv7 = nn.Conv2d(128, 256, (3,3))
        self.conv8 = nn.Conv2d(256, 256, (3,3))
        #third max pool
        self.maxpool9 = nn.MaxPool2d((2,2))

        #forth block
        self.conv10 = nn.Conv2d(256, 512, (3, 3))
        self.conv11 = nn.Conv2d(512, 512, (3,3))
        #forth max pool
        self.maxpool12 = nn.MaxPool2d((2,2))

        #-------------------------------------------

        self.conv13 = nn.Conv2d(512, 1024, (3,3))
        self.conv14 = nn.Conv2d(1024, 1024, (3,3))
        
        #-------------------------------------------

        #first block
        self.up15 = nn.ConvTranspose2d(1024, 512, (2,2), 2)
        #cat
        self.conv16 = nn.Conv2d(1024, 512, (3,3))
        self.conv17 = nn.Conv2d(512, 512, (3,3))

        #second block
        self.up18 = nn.ConvTranspose2d(512, 256, (2,2), 2)
        #cat
        self.conv19 = nn.Conv2d(512, 256, (3,3))
        self.conv20 = nn.Conv2d(256, 256, (3,3))

        #third block
        self.up21 = nn.ConvTranspose2d(256, 128, (2,2), 2)
        #cat
        self.conv22 = nn.Conv2d(256, 128, (3,3))
        self.conv23 = nn.Conv2d(128, 128, (3,3))

        #forth block
        self.up24 = nn.ConvTranspose2d(128, 64, (2,2), 2)
        #cat
        self.conv25 = nn.Conv2d(128, 64, (3,3))
        self.conv26 = nn.Conv2d(64, 64, (3,3))
        self.conv27 = nn.Conv2d(64, 1, (1,1))
    
    def forward(self, x):
        #first block
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x1 = x
        x = self.maxpool3(x)

        #second block
        x = self.conv4(x)
        x = self.activation(x)
        x = self.conv5(x)
        x = self.activation(x)
        x2 = x
        x = self.maxpool6(x)

        #third block
        x = self.conv7(x)
        x = self.activation(x)
        x = self.conv8(x)
        x = self.activation(x)
        x3 = x
        x = self.maxpool9(x)

        #forth block
        x = self.conv10(x)
        x = self.activation(x)
        x = self.conv11(x)
        x = self.activation(x)
        x4 = x
        x = self.maxpool12(x)

        #-------------------------------------------
        x = self.conv13(x)
        x = self.activation(x)
        x = self.conv14(x)
        x = self.activation(x)
        #-------------------------------------------

        #first block
        x = self.up15(x)
        crop = CenterCrop((x.shape[2], x.shape[2]))
        x4 = crop(x4)
        x = torch.cat([x4, x], dim=1)
        x = self.conv16(x)
        x = self.activation(x)
        x = self.conv17(x)
        x = self.activation(x)

        #second block
        x = self.up18(x)
        crop = CenterCrop((x.size()[2], x.size()[2]))
        x3 = crop(x3)
        x = torch.cat([x3, x], dim=1) 
        x = self.conv19(x)
        x = self.activation(x)
        x = self.conv20(x)
        x = self.activation(x)

        #third block
        x = self.up21(x)
        crop = CenterCrop((x.size()[2], x.size()[2]))
        x2 = crop(x2)
        x = torch.cat([x2, x], dim=1)
        x = self.conv22(x)
        x = self.activation(x)
        x = self.conv23(x)
        x = self.activation(x)
        
        #forth block
        x = self.up24(x)
        crop = CenterCrop((x.size()[2], x.size()[2]))
        x1 = crop(x1)
        x = torch.cat([x1, x], dim=1)
        x = self.conv25(x)
        x = self.activation(x)
        x = self.conv26(x)
        x = self.activation(x)
        x = self.conv27(x)
        x = self.sigmoid(x)

        return x

class ConvBlock(torch.nn.Module):
    def __init__(self, inchannels, outchannels, kernal):
        super().__init__()
        self.conv = nn.Conv2d(inchannels, outchannels, kernal, padding='same')
        self.activation = nn.ReLU()
        self.normalize = nn.BatchNorm2d(outchannels)
    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        x = self.normalize(x)
        return x
        
        
         

class Encoder(torch.nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        self.conv1 = ConvBlock(channels, 64, (3, 3))
        self.conv2 = ConvBlock(64, 64, (3, 3))
        self.maxPool3 = nn.MaxPool2d((2, 2), 2)
        
        self.conv4 = ConvBlock(64, 128, (3, 3))
        self.conv5 = ConvBlock(128, 128, (3, 3))
        self.maxPool6 = nn.MaxPool2d((2, 2), 2)
        
        self.conv7 = ConvBlock(128, 256, (3, 3))
        self.conv8 = ConvBlock(256, 256, (3, 3))
        self.conv9 = ConvBlock(256, 256, (3, 3))
        self.maxPool10 = nn.MaxPool2d((2, 2), 2)
        
        self.conv11 = ConvBlock(256, 512, (3, 3))
        self.conv12 = ConvBlock(512, 512, (3, 3))
        self.conv13 = ConvBlock(512, 512, (3, 3))
        
        self.upsample14 = nn.Upsample(scale_factor=2)
        self.conv15 = ConvBlock(512, 256, (3, 3))
        self.conv16 = ConvBlock(256, 256, (3, 3))
        self.conv17 = ConvBlock(256, 256, (3, 3))
        
        self.upsample18 = nn.Upsample(scale_factor=2)
        self.conv19 = ConvBlock(256, 128, (3, 3))
        self.conv20 = ConvBlock(128, 128, (3, 3))
        self.conv21 = ConvBlock(128, 128, (3, 3))
        
        self.upsample22 = nn.Upsample(scale_factor=2)
        self.conv23 = ConvBlock(128, 64, (3, 3))
        self.conv24 = ConvBlock(64, 64, (3, 3))
        self.conv25 = ConvBlock(64, 64, (3, 3))
        
        self.conv26 = ConvBlock(64, 1, (3, 3))
        
        
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxPool3(x)
        
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.maxPool6(x)
        
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.maxPool10(x)
        
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        
        x = self.upsample14(x)
        x = self.conv15(x)
        x = self.conv16(x)
        x = self.conv17(x)
        
        x = self.upsample18(x)
        x = self.conv19(x)
        x = self.conv20(x)
        x = self.conv21(x)
        
        x = self.upsample22(x)
        x = self.conv23(x)
        x = self.conv24(x)
        x = self.conv25(x)
        
        x = self.conv26(x)
        x = self.sigmoid(x)
        
        return x