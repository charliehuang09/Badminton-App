import torch

epochs = 500
lr = 1
batch_size = 6
num_workers = 6

classes = 255
device = torch.device('cuda')
test = False