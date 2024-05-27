import numpy as np
from torchvision.utils import make_grid
import torch
import random
import config
import os
from torch.utils.tensorboard import SummaryWriter
from model import Unet, TrackNet
from torchsummary import summary
import cv2
class Logger:
    def __init__(self, writer, writer_path):
        self.index = 0
        self.writer = writer
        self.writer_path = writer_path
        
        self.value = 0
        self.length = 0
    
    def add(self, input, length):
        input /= length
        self.value += input
        self.length += 1
        
    def get(self):
        return self.value / self.length 

    def write(self):
        value = self.value / self.length
        self.value = 0
        self.length = 0
        
        self.writer.add_scalar(self.writer_path, value, self.index)
        self.index += 1
        
        return value
        
def getHeatMap(input):
    img = np.empty((640, 360))
    for i in range(640):
        for j in range(360):
            img[i][j] = np.argmax(input[:, i, j])
    
    img = np.array([img])
    img = torch.from_numpy(img)
    img /= config.classes
    img *= 255
    return img
def writeTrainImage(writer, model, epoch, num_samples=4):
    img_idx = random.sample(range(len(os.listdir('data/train/imgs'))), num_samples)
    imgs = []
    for i in img_idx:
        imgs.append(torch.from_numpy(np.load(f'data/train/imgs/{i}.npy')))
    grid = make_grid(imgs)
    
    writer.add_image('train/X-Images', grid.cpu(), epoch)
    
    
    imgs = np.array(imgs)
    imgs = torch.from_numpy(imgs)
    imgs = imgs.to(config.device)
    
    outputs = model(imgs).detach().cpu()
    grid = []
    for output in outputs:
        grid.append(getHeatMap(output))
    grid = make_grid(grid)
    writer.add_image('train/Output-Images', grid, epoch)
    
    grid = []
    for i in img_idx:
        grid.append(torch.from_numpy(np.array([np.load(f'data/train/labels/{i}.npy')])))
    grid = make_grid(grid)
    writer.add_image('train/Label-Images', grid.cpu(), epoch)
    


def main():
    model = TrackNet().to(config.device)
    writer = SummaryWriter()
    writeTrainImage(writer, model, 0)

if __name__=='__main__':
    main()