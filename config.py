import torch

epochs = 150
lr = 1
batch_size = 2
num_workers = 64

classes = 255
device = torch.device('cuda')
test = False