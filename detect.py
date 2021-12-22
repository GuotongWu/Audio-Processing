import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from dp_cnnwithatten import *
from argumentation import SoundDS

device = 'cpu'
lr = 0.0001
epoch = 1000
batch_size = 1
net = Net()


for each in os.listdir('./saved_model/'):
    testDataset = SoundDS('./test/')
    testLoader = DataLoader(testDataset, batch_size=batch_size)
    net.load_state_dict(torch.load('./saved_model/' + each))
    net = net.to(device=device)

    alls = 0
    correct = 0

    for x, y in testLoader:
        with torch.no_grad():
            alls += batch_size
            x, y = x.to(device), y.to(device)
            y_ = net(x)
            # print(y_.argmax(dim=1).cpu().numpy(), y.cpu().numpy())
            correct += (y_.argmax(dim=1).cpu().numpy() == y.cpu().numpy()).sum()

    print(each, correct / alls)
