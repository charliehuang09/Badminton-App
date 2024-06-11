import torch

epochs = 500
lr = 1
batch_size = 2
num_workers = 2

classes = 255
device = torch.device('cuda')
test = False