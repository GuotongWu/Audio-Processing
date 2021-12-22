import os
import pickle
import torchaudio
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from argumentation import SoundDS

class AudioDataset(Dataset):
    def __init__(self, data_dir, sr=44100) -> None:
        self.data_dir = data_dir
        self.files = os.listdir(self.data_dir)
        self.sr = sr
            
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        sig, sr = torchaudio.load(os.path.join(self.data_dir, self.files[idx]))
        sig = torchaudio.transforms.Resample(sr, self.sr)(sig.mean(dim=0))
        sig = torchaudio.transforms.MFCC(self.sr, n_mfcc=26, melkwargs={"n_fft": 2048, "hop_length": 512, "power": 2})(sig)
        order_id = int(self.files[idx][:-4].split('-')[2])
        user_id = int(self.files[idx][:-4].split('-')[0])
        return sig.mean(dim=-1).unsqueeze(dim=1), order_id, user_id

class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return torch.transpose(input=x, dim0=self.dim0, dim1=self.dim1)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # (1, 26)
        self.BatchNorm = nn.BatchNorm1d(num_features=26)
        self.Flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.FC0 = nn.Linear(in_features=26, out_features=256)
        self.GELU_0 = nn.GELU()

        # self.FC1 = nn.Linear(in_features=256, out_features=64)
        # self.GELU_1 = nn.GELU()

        self.FC2 = nn.Linear(in_features=256, out_features=9)
        self.Softmax = nn.Softmax(dim=-1)

        self.Sequence = nn.Sequential(
            self.BatchNorm,
            # self.Transformer,
            self.Flatten,
            self.FC0,
            self.GELU_0,
            # self.FC1,
            # self.GELU_1,
            self.FC2,
            self.Softmax
        )

        self.lossfunc = torch.nn.CrossEntropyLoss()

    def forward(self, x:torch.Tensor):
        # x = x.reshape((-1, 1, 40, 32))
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

    for x, y, z in dev_loader:
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
    batch_size = 1024
    regulation_lambda = 0.0000
    early_stop = 100

    
    net = Net().to(device=device)

    optimiser = torch.optim.Adam(net.parameters(), lr=lr) 


    train_dataset = AudioDataset('src')
    # train_dataset = AudioDataset('./pkl/all_train_padding.pkl', 'train', shuffle_indices)
    # dev_dataset = AudioDataset('./pkl/all_train_padding.pkl', 'dev', shuffle_indices)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    # dev_loader = DataLoader(dataset=dev_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    test_dataset = AudioDataset('test')
    # test_dataset = AudioTestDataset('./pkl/all_test_padding.pkl')
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0)

    train_loss = []
    train_acc = []
    dev_loss = []
    dev_acc = []
    min_loss = 10000
    early_stop_cnt = 0
    best_acc = 0

    for _ in range(epoch):
        for x, y, z in train_loader:
            net.train()
            x, y = x.to(device=device), y.to(device=device)
            y_ = net(x)

            loss = net.cal_loss(y_, y, regulation_lambda)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            train_loss.append(loss.item())
            train_acc.append((y_.argmax(dim=1) == y).to(torch.float32).mean().item())
        loss, acc = dev(test_loader, net, device, regulation_lambda)
        dev_loss.append(loss)
        dev_acc.append(acc)

        if loss < min_loss:
            min_loss = loss
            if _ % 10 == 0:        
                torch.save(net.state_dict(), './saved_model/net_cnn_atten%d.pt' % _)
            print('Saving Model: ', _, '|' ,epoch,"  " ,'acc:',acc , '  ', 'loss:', loss, " ", early_stop_cnt)

            if best_acc <= acc:
                torch.save(net.state_dict(), './saved_model/net_cnn_atten at %d.pt' % (acc*10000))
                best_acc = acc

            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        if early_stop_cnt >= early_stop:
            break        

    import matplotlib.pyplot as plt
    plt.plot(np.arange(len(train_loss)),train_loss)
    plt.show()

    del net
    
    net = Net().to(device)
    ckpt = torch.load('./saved_model/net.pt', map_location='cpu')  # Load your best model
    net.load_state_dict(ckpt)

    

    alls = 0
    correct = 0
    for x, y in test_loader:
        alls += 1
        x, y = x.to(device=device), y.to(device=device)
        with torch.no_grad():
            y_ = net(x)
        print(y_.argmax().item(), y.item())
        if(y_.argmax().item() == y.item()):
            correct += 1
    print(correct / alls)