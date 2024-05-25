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
from logger import Logger, writeTrainImage
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam, SGD, Adadelta
from torch.nn import MSELoss, CrossEntropyLoss
def main():
    print(torch.seed())
    device = config.device
    torch.cuda.empty_cache()

    model = TrackNet()
    
    summary(model, (3, 640, 360))
    
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
            
            optimizer.zero_grad()
            outputs = model(x)
            loss = loss_fn(outputs.flatten(), y.flatten())
            loss.backward()
            optimizer.step()
            
            trainLossLogger.add(loss.item(), len(batch) * config.classes * 360 * 640)
        
        # for p in model.parameters():
        #         print(f'===========\ngradient\n----------\n{p.grad}')
    
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(valid_dataLoader):
                x, y = batch
                x = x.to(device)
                y = y.to(device)
                
                outputs = model(x)
                loss = loss_fn(outputs.flatten(), y.flatten())
                
                validLossLogger.add(loss.item(), len(batch) * config.classes * 360 * 640)
        
        writeTrainImage(writer, model, epoch)
        trainLossLogger.write()
        validLossLogger.write()
        
            

if __name__=='__main__':
    main()