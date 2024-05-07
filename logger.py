import numpy as np
from torchvision.utils import make_grid
import torch
import random
import config
import os
from torch.utils.tensorboard import SummaryWriter
from model import Unet
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
        
        
def writeTrainImage(writer, model, epoch):
    img_idx = random.sample(range(len(os.listdir('data/train/imgs'))), 8)
    imgs = []
    for i in img_idx:
        imgs.append(torch.from_numpy(np.load(f'data/train/imgs/{i}.npy')))
    
    grid = make_grid(imgs)
    
    img = np.array(grid)
    img = img.transpose()
    
    writer.add_image('train/X-Images', grid.cpu(), 0)
    
    
    imgs = np.array(imgs)
    imgs = torch.from_numpy(imgs)
    imgs = imgs.to(config.device)
    
    outputs = model(imgs).detach().cpu()
    outputs = outputs.repeat([1, 3, 1, 1])
    grid = make_grid(outputs)
    
    writer.add_image('train/Output-Images', grid.cpu(), epoch)
    
    imgs = []
    for i in img_idx:
        imgs.append(torch.from_numpy(np.array([np.load(f'data/train/labels/{i}.npy')])))
    grid = make_grid(imgs)
    writer.add_image('train/Label-Images', grid.cpu(), 0)
    


def main():
    model = Unet().to(config.device)
    writer = SummaryWriter()
    writeTrainImage(writer, model, 0)

if __name__=='__main__':
    main()