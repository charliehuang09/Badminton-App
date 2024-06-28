import torch
from torch import nn
from torchvision.transforms import CenterCrop
import matplotlib.pyplot as plt
from torchsummary import summary
import config

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
        self.conv = nn.Conv2d(inchannels, outchannels, kernal, padding=1)
        self.activation = nn.ReLU()
        self.normalize = nn.BatchNorm2d(outchannels)
    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        x = self.normalize(x)
        return x
        
class Encoder(torch.nn.Module):
    def __init__(self, channels=9):
        super().__init__()
        
        self.activation = nn.ReLU()
        
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
        
        return x

class Decoder(torch.nn.Module):
    def __init__(self, channels=9):
        super().__init__()
        
        self.pad = nn.ZeroPad2d((0, 1, 0, 0))
        
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.conv2 = ConvBlock(1024, 512, (3, 3))
        self.conv3 = ConvBlock(512, 512, (3, 3))
        
        self.upsample4 = nn.Upsample(scale_factor=2)
        self.conv5 = ConvBlock(512, 256, (3, 3))
        self.conv6 = ConvBlock(256, 256, (3, 3))
        
        self.upsample7 = nn.Upsample(scale_factor=2)
        self.conv8 = ConvBlock(256, 128, (3, 3))
        self.conv9 = ConvBlock(128, 128, (3, 3))
        
        self.upsample10 = nn.Upsample(scale_factor=2)
        self.conv11 = ConvBlock(128, 64, (3, 3))
        self.conv12 = ConvBlock(64, 64, (3, 3))
        
        self.conv13 = ConvBlock(64, config.classes + 1, (3, 3))
        
    def forward(self, x):
        
        x = self.upsample4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        
        x = self.upsample7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        
        x = self.upsample10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        
        x = self.conv13(x)
        
        return x
    
class TrackNet(torch.nn.Module):
    def __init__(self, channels=9):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self._init_weights()

        
    def forward(self, x):
        
        batch_size = len(x)
        
        x = self.encoder(x)
        x = self.decoder(x)
        
        x = x.reshape(batch_size, config.classes + 1, -1)
        # x = self.softmax(x)
        x = x.reshape(batch_size, config.classes + 1, 640, 360)
        
        # x = self.sigmoid(x)
        
        return x
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.uniform_(module.weight, -0.05, 0.05)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)    

def main():
    model = TrackNet()
    summary(model, (9, 640, 360))

if __name__=='__main__':
    main()