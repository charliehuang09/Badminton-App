import torch

epochs = 500
lr = 1
batch_size = 4
num_workers = 4

classes = 255
device = torch.device('cuda')
test = False