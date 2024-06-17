import torch
from torch import nn
from torch.nn import DataParallel
from dataset import Dataset
import config
from model import Unet, TrackNet
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
from torchsummary import summary
import time
from logger import Logger, writeImages
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam, SGD, Adadelta
from torch.nn import MSELoss, CrossEntropyLoss
def main():
    print(torch.seed())
    device = config.device
    torch.cuda.empty_cache()

    model = TrackNet()
    
    summary(model, (9, 640, 360))
    
    # model = DataParallel(model)
    model = model.to(device)
    
    train_dataset = Dataset('train', test=config.test)
    valid_dataset = Dataset('valid', test=config.test)
    
    train_dataLoader = DataLoader(train_dataset, num_workers=config.num_workers, batch_size=config.batch_size, shuffle=True)
    valid_dataLoader = DataLoader(valid_dataset, num_workers=config.num_workers, batch_size=config.batch_size, shuffle=True)
    
    writer = SummaryWriter()
    
    trainLossLogger = Logger(writer, 'train/Loss')
    validLossLogger = Logger(writer, 'valid/Loss')
    
    loss_fn = CrossEntropyLoss()
    
    # optimizer = Adam(model.parameters(), lr=config.lr)
    optimizer = Adadelta(model.parameters(), lr=config.lr)
    
    for epoch in trange(config.epochs):
        model.train()
        for i, batch in enumerate(train_dataLoader):
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            y = y.long()
            
            optimizer.zero_grad()
            outputs = model(x)
            outputs = outputs.reshape(outputs.shape[0], outputs.shape[1], -1)
            y = y.reshape(y.shape[0], -1)
            loss = loss_fn(outputs, y)
            loss.backward()
            optimizer.step()
            
            trainLossLogger.add(loss.item(), len(batch))
    
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(valid_dataLoader):
                x, y = batch
                x = x.to(device)
                y = y.to(device)
                y = y.long()
                
                outputs = model(x)
                outputs = outputs.reshape(outputs.shape[0], outputs.shape[1], -1)
                y = y.reshape(y.shape[0], -1)
                loss = loss_fn(outputs, y)
                
                validLossLogger.add(loss.item(), len(batch))
        
        writeImages(writer, model, epoch, 'train')
        writeImages(writer, model, epoch, 'valid')
        trainLossLogger.write()
        validLossLogger.write()
        
            

if __name__=='__main__':
    main()
