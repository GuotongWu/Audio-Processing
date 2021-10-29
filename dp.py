import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class AudioDataset(Dataset):
    def __init__(self, pkl_path) -> None:
        self.pkl_path = pkl_path
        with open(self.pkl_path, 'rb') as f:
            data = pickle.load(f)
        self.raw_x = data['feature'][:-70]
        self.raw_y = data['label'][:-70]
            
    def __len__(self):
        return len(self.raw_x)
        
    def __getitem__(self, idx):
        return torch.from_numpy(self.raw_x[idx]).unsqueeze(dim=0).to(torch.float32), torch.tensor(self.raw_y[idx][0]).to(torch.long)

class AudioTestDataset(Dataset):
    def __init__(self, pkl_path) -> None:
        self.pkl_path = pkl_path
        with open(self.pkl_path, 'rb') as f:
            data = pickle.load(f)
        self.raw_x = data['feature'][-70:]
        self.raw_y = data['label'][-70:]
            
    def __len__(self):
        return len(self.raw_x)
        
    def __getitem__(self, idx):
        return torch.from_numpy(self.raw_x[idx]).unsqueeze(dim=0).to(torch.float32), torch.tensor(self.raw_y[idx][0]).to(torch.long)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # (20, 570)
        # trans to (114, 100)

        # (1, 114, 100)
        self.Conv0 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)

        self.ReLU0 = nn.ReLU()

        # (8, 57, 50)
        self.Pooling0 = nn.MaxPool2d(kernel_size=2, stride=2)

        # (32, 20, 18)
        self.Conv1 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=5, stride=3, padding=3)

        self.ReLU1 = nn.ReLU()

        # (32, 5, 4)
        self.Pooling1 = nn.MaxPool2d(kernel_size=4, stride=4)

        self.Flatten = nn.Flatten(start_dim=1, end_dim=-1)

        self.FC0 = nn.Linear(in_features=32*5*4, out_features=128)

        self.ReLU_0 = nn.ReLU()

        self.FC1 = nn.Linear(in_features=128, out_features=32)
        
        self.ReLU_1 = nn.ReLU()

        self.FC2 = nn.Linear(in_features=32, out_features=9)

        self.Softmax = nn.Softmax()

        self.Sequence = nn.Sequential(
            self.Conv0,
            self.ReLU0,
            self.Pooling0,
            self.Conv1,
            self.ReLU1,
            self.Pooling1,
            self.Flatten,
            self.FC0,
            self.ReLU_0,
            self.FC1,
            self.ReLU_1,
            self.FC2,
            self.Softmax
        )


    def forward(self, x:torch.Tensor):
        x = x.reshape((-1, 1, 114, 100))
        return self.Sequence(x)


if __name__ == '__main__':
    device = 0
    lr = 0.0001
    epoch = 1000
    batch_size = 16
    net = Net().to(device=device)


    optimiser = torch.optim.Adam(net.parameters(), lr=lr)
    loss_func = torch.nn.CrossEntropyLoss()

    dataset = AudioDataset('audio.pkl')
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    for _ in range(epoch):
        for x, y in loader:
            x, y = x.to(device=device), y.to(device=device)
            y_ = net(x)
            loss = loss_func(y_, y)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        print(_, '|' ,epoch,"  " ,(y_.argmax(dim=1) == y).to(torch.float32).mean())

            
    dataset = AudioTestDataset('audio.pkl')
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=0)

    alls = 0
    correct = 0
    for x, y in loader:
        alls += 1
        x, y = x.to(device=device), y.to(device=device)
        y_ = net(x)
        print(y_.argmax().item(), y.item())
        if(y_.argmax().item() == y.item()):
            correct += 1
    print(correct / alls)