import torch

epochs = 150
lr = 1
batch_size = 4
num_workers = 48

classes = 255
device = torch.device('cuda')
test = False