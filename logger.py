import numpy as np
from torchvision.utils import make_grid
import torch
import random
import config
from typing import Literal
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
        
def getHeatMap(img):
    img = np.array(img)
    img = np.argmax(img, axis=0)
    
    img = np.array(img, dtype=np.uint8)
    img = img[..., np.newaxis]
    
    circles = cv2.HoughCircles(img, method=cv2.HOUGH_GRADIENT, dp=1, minDist=100)
    
    if (circles is None):
        output = np.zeros((640, 360, 3))
        output = np.rollaxis(output, 2)
        output = torch.from_numpy(output)
        return output
        
    circles = np.uint16(np.around(circles))
    output = np.zeros((640, 360, 3))
    for i in circles[0,:]:
        cv2.circle(output,(i[0],i[1]),i[2],(0,255,0),2)
        cv2.circle(output,(i[0],i[1]),2,(0,0,255),3)
    
    output = np.rollaxis(output, 2)
    
    output = torch.from_numpy(output)
    
    return output
def writeImages(writer, model, epoch, type : Literal['train', 'valid'], num_samples=2):
    torch.cuda.empty_cache()
    img_idx = random.sample(range(len(os.listdir(f'data/{type}/imgs')) - 1), num_samples)
    imgs = []
    inputs = []
    for i in img_idx:
        imgs.append(torch.from_numpy(np.load(f'data/{type}/imgs/{i}.npy')[3:6]))
        inputs.append(torch.from_numpy(np.load(f'data/{type}/imgs/{i}.npy')))
        
    grid = make_grid(imgs)
    
    writer.add_image(f'{type}/X-Images', grid.cpu() / 255, epoch)
    
    inputs = np.array(inputs)
    inputs = torch.from_numpy(inputs)
    inputs = inputs.to(config.device)
    inputs /= 255
    
    outputs = model(inputs).detach().cpu()
    
    grid = []
    for output in outputs:
        img = np.array(output)
        img = np.argmax(img, axis=0).astype(np.float32)
        img = np.array([img])
        grid.append(torch.from_numpy(img))
    grid = make_grid(grid)
    writer.add_image(f'{type}/Output-Images', grid / 255, epoch)
    
    grid = []
    for output in outputs:
        grid.append(getHeatMap(output))
    grid = make_grid(grid)
    writer.add_image(f'{type}/Processed-Images', grid, epoch)
    
    grid = []
    for i in img_idx:
        grid.append(torch.from_numpy(np.array([np.load(f'data/{type}/labels/{i}.npy')])))
    grid = make_grid(grid)
    writer.add_image(f'{type}/Label-Images', grid.cpu(), epoch)

def main():
    model = TrackNet().to(config.device)
    writer = SummaryWriter()
    writeImages(writer, model, 0, 'train')
    writeImages(writer, model, 0, 'valid')

if __name__=='__main__':
    main()
