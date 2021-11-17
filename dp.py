import pickle
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class AudioDataset(Dataset):
    def __init__(self, pkl_path, type, shuffle_indices) -> None:
        self.type = type
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        if type == 'train':
            indices = shuffle_indices[:len(shuffle_indices)//10*8]
        else:
            indices = shuffle_indices[len(shuffle_indices)//10*8:]

        self.raw_x = torch.FloatTensor(data['feature'])[indices]
        self.raw_y = torch.LongTensor(data['label'])[indices]
            
    def __len__(self):
        return len(self.raw_x)
        
    def __getitem__(self, idx):
        return self.raw_x[idx].unsqueeze(dim=0), self.raw_y[idx][0]

class AudioTestDataset(Dataset):
    def __init__(self, pkl_path) -> None:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        self.raw_x = data['feature']
        self.raw_y = data['label']

    def __len__(self):
        return len(self.raw_x)
        
    def __getitem__(self, idx):
        return torch.from_numpy(self.raw_x[idx]).unsqueeze(dim=0).to(torch.float32), torch.tensor(self.raw_y[idx][0]).to(torch.long)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # (1, 20, 52)
        self.Conv0 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.BatchNorm0 = nn.BatchNorm2d(num_features=8)
        self.ReLU0 = nn.ReLU()
        self.Pooling0 = nn.MaxPool2d(kernel_size=2, stride=2)  # (8, 10, 26)

        self.Conv1 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.BatchNorm1 = nn.BatchNorm2d(num_features=32)
        self.ReLU1 = nn.ReLU()
        self.Pooling1 = nn.MaxPool2d(kernel_size=2, stride=2)  # (32, 5, 13)

        self.Flatten = nn.Flatten(start_dim=1, end_dim=-1)

        self.FC0 = nn.Linear(in_features=32*5*13, out_features=256)
        self.ReLU_0 = nn.ReLU()

        self.FC1 = nn.Linear(in_features=256, out_features=64)
        self.ReLU_1 = nn.ReLU()

        self.FC2 = nn.Linear(in_features=64, out_features=9)
        self.Softmax = nn.Softmax()

        self.Sequence = nn.Sequential(
            self.Conv0,
            self.BatchNorm0,
            self.ReLU0,
            self.Pooling0,
            self.Conv1,
            self.BatchNorm1,
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

        self.lossfunc = torch.nn.CrossEntropyLoss()

    def forward(self, x:torch.Tensor):
        return self.Sequence(x)

    def cal_loss(self, pred, y, regulation_lambda):
        regulation_loss = 0
        for param in net.parameters():
            regulation_loss += torch.sum(param**2)
        return self.lossfunc(pred, y) + regulation_lambda*regulation_loss

def dev(dev_loader, net, device, regulation_lambda):
    net.eval()
    total_loss = 0
    total_acc = 0

    for x, y in dev_loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            pred = net(x)
            loss = net.cal_loss(pred, y, regulation_lambda).item()
            total_loss += loss * len(x)
            total_acc += (pred.argmax(dim=1) == y).to(torch.float32).mean().item() * len(x)

    return  total_loss/len(dev_loader.dataset), total_acc/len(dev_loader.dataset)

if __name__ == '__main__':
    device = 0
    lr = 0.0001
    epoch = 5000
    batch_size = 8
    regulation_lambda = 0.0001
    early_stop = 500

    net = Net().to(device=device)

    optimiser = torch.optim.Adam(net.parameters(), lr=lr) 

    shuffle_indices = indices = np.arange(279)
    np.random.shuffle(indices)

    train_dataset = AudioDataset('train_padding.pkl', 'train', shuffle_indices)
    dev_dataset = AudioDataset('train_padding.pkl', 'dev', shuffle_indices)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    dev_loader = DataLoader(dataset=dev_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    train_loss = []
    train_acc = []
    dev_loss = []
    dev_acc = []
    min_loss = 10000
    early_stop_cnt = 0

    for _ in range(epoch):
        for x, y in train_loader:
            net.train()
            x, y = x.to(device=device), y.to(device=device)
            y_ = net(x)

            loss = net.cal_loss(y_, y, regulation_lambda)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            train_loss.append(loss.item())
            train_acc.append((y_.argmax(dim=1) == y).to(torch.float32).mean().item())
        loss, acc = dev(dev_loader, net, device, regulation_lambda)
        dev_loss.append(loss)
        dev_acc.append(acc)

        if loss < min_loss:
            min_loss = loss        
            torch.save(net.state_dict(), 'net.pt')
            print('Saving Model: ', _, '|' ,epoch,"  " ,acc , '  ', loss)

            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        if early_stop_cnt > early_stop:
            break        

    dataset = AudioTestDataset('test_padding.pkl')
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=0)

    del net
    net = Net().to(device)
    ckpt = torch.load('net.pt', map_location='cpu')  # Load your best model
    net.load_state_dict(ckpt)

    alls = 0
    correct = 0
    for x, y in loader:
        alls += 1
        x, y = x.to(device=device), y.to(device=device)
        with torch.no_grad():
            y_ = net(x)
        print(y_.argmax().item(), y.item())
        if(y_.argmax().item() == y.item()):
            correct += 1
    print(correct / alls)

    import matplotlib.pyplot as plt
    plt.plot(np.arange(len(dev_loss)), dev_loss)
    plt.show()