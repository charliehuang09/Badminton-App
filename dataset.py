from torch.utils.data import Dataset
import numpy as np
import os
import torch
from tqdm import tqdm, trange
from typing import Literal
from torch.utils.data import DataLoader
from linetimer import CodeTimer
from torchsummary import summary
import config

class Dataset(Dataset):
    def __init__(self, type : Literal['train', 'valid'], test=False):
        self.test = test
        self.x_path = os.path.join('data', type, 'imgs')
        self.y_path = os.path.join('data', type, 'labels')
        self.length = len(os.listdir(self.x_path)) - 1
        
        self.meshgrid, _, _ = np.meshgrid(np.linspace(0, 255, 256), np.linspace(0, 0, 640), np.linspace(0, 0, 360), indexing='ij')
        
        if (test):
            self.length = round(self.length / 100)
            
        print(f"Length: {self.length}")
    
    def getx(self):
        return self.x
    def gety(self):
        return self.y
    def __len__(self):
        return self.length
    def getY(self, input):
        output = np.empty((256, 640, 360))
        for i in range(256):
            output[i] = (self.meshgrid[i] == input)
        return output
    def __getitem__(self, index):
        x = torch.from_numpy(np.load(os.path.join(self.x_path, str(index) + '.npy'))) / 255
        y = torch.from_numpy(np.load(os.path.join(self.y_path, str(index) + '.npy')).astype(np.float32))
        return x, y
    
def main():
    train_dataset = Dataset("train")
    valid_dataset = Dataset("valid")
    
    train_dataset[0]
    valid_dataset[0]
    
    x, y = train_dataset[0]

    train_dataLoader = DataLoader(train_dataset, num_workers=config.num_workers, batch_size=config.batch_size)
    valid_dataLoader = DataLoader(valid_dataset, num_workers=config.num_workers, batch_size=config.batch_size)
    
    with CodeTimer():
        for i, batch in enumerate(tqdm(train_dataLoader)):
            x, y = batch
            x = x.to(config.device)
            y = y.to(config.device)
        for i, batch in enumerate(valid_dataLoader):
            x, y = batch
            x = x.to(config.device)
            y = y.to(config.device)


if __name__=='__main__':
    main()
