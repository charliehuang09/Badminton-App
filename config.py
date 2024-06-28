import torch

epochs = 250
lr = 1
batch_size = 8
num_workers = 8
prefetch_factor=8

classes = 255
device = torch.device('cuda')
test = False