from torch.utils.data import Dataset
import numpy as np
import os
import torch
from tqdm import tqdm, trange
from typing import Literal

class Dataset(Dataset):
    def __init__(self, type : Literal['train', 'valid'], test=False):
        self.test = test
        self.x_path = os.path.join('data', type, 'imgs')
        self.y_path = os.path.join('data', type, 'labels')
        self.length = len(os.listdir(self.x_path))
        if (test):
            self.length = round(self.length / 10)
            
        print(f"Length: {self.length}")
    
    def getx(self):
        return self.x
    def gety(self):
        return self.y
    def __len__(self):
        return self.length
    def getY(self, input):
        print(input.shape)
        output = torch.zeros((256, 640, 360))
        print(output.shape)
        for i in range(640):
            for j in range(360):
                output[0, i, j] = 1
        return output
    def __getitem__(self, index):
        x = torch.from_numpy(np.load(os.path.join(self.x_path, str(index) + '.npy')))
        y = torch.from_numpy(np.load(os.path.join(self.y_path, str(index) + '.npy')))
        y = self.getY(y)
        return x, y
    
def main():
    train_dataset = Dataset("train")
    valid_dataset = Dataset("valid")


if __name__=='__main__':
    main()
