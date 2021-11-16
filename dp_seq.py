import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt


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
        return torch.from_numpy(self.raw_x[idx]).to(torch.float32), torch.tensor(self.raw_y[idx][0]).to(torch.long)

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
        return torch.from_numpy(self.raw_x[idx]).to(torch.float32), torch.tensor(self.raw_y[idx][0]).to(torch.long)

class NetBlock(nn.Module):
    def __init__(self, in_channels:int=20, heads_num:int=4, hidden_nodes:int=1024):
        super(NetBlock, self).__init__()
        self.in_channels = in_channels
        self.heads_num = heads_num
        self.hidden_nodes = hidden_nodes
        self.head_channels = self.in_channels // self.heads_num
        self.scale = self.head_channels ** -0.5

        # (in_channels, seq_len)
        # transpose

        # (seq_len, in_channels)
        self.qkv_linear = torch.nn.Linear(in_features=self.in_channels, out_features=self.in_channels * 3)
        self.layernorm0 = torch.nn.LayerNorm(normalized_shape=self.in_channels)

        self.multihead_attention = torch.nn.MultiheadAttention(
            embed_dim=self.in_channels,
            num_heads=self.heads_num,
            # batch_first=True
        )

        self.layernorm1 = torch.nn.LayerNorm(normalized_shape=self.in_channels)
        self.linear0 = torch.nn.Linear(in_features=self.in_channels, out_features=self.hidden_nodes)
        self.activation = torch.nn.ReLU()
        self.linear1 = torch.nn.Linear(in_features=self.hidden_nodes, out_features=self.in_channels)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # x (seq_len, in_channels)
        x_norm0 = self.layernorm0(x)

        qkv = self.qkv_linear(x_norm0)

        # qkv (seq_len, in_channels*3)
        q, k, v = torch.reshape(input=qkv, shape=(3, *x.shape))

        # q, k, v(seq_len, in_channels)
        attn_output, attn_output_weights = self.multihead_attention(q, k, v)

        x = x + attn_output

        x_norm1 = self.layernorm1(x)

        x_norm1 = self.linear0(x_norm1)

        x_norm1 = self.activation(x_norm1)

        x_norm1 = self.linear1(x_norm1)

        x = x + x_norm1

        return x


class Net(torch.nn.Module):
    def __init__(self, in_channels:int=20, seq_len:int=570, heads_num:int=4, hidden_nodes:int=1024, block_num:int=6):
        super().__init__()
        self.in_channels = in_channels
        self.heads_num = heads_num
        self.hidden_nodes = hidden_nodes
        self.seq_len = seq_len
        self.block_num = block_num

        self.cls_token = torch.nn.Parameter(torch.rand(1, 1, self.in_channels))
        # self.positional_embedding = torch.nn.Parameter(torch.rand(1, self.seq_len + 1, self.in_channels))
        self.block_list = torch.nn.Sequential(*[
            NetBlock(
                in_channels=self.in_channels,
                heads_num=self.heads_num,
                hidden_nodes=self.hidden_nodes
            ) for i in range(self.block_num) 
        ])
        self.layernorm = torch.nn.LayerNorm(normalized_shape=self.in_channels)
        self.output_linear = torch.nn.Linear(in_features=self.in_channels, out_features=9)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x:torch.Tensor):
        # x (batches, in_channels, seq_len)
        x = x.transpose(-1, -2)

        # x (batches, seq_len, in_channels)
        cls_token = self.cls_token.expand(x.shape[0], 1, self.in_channels)
        x = torch.cat((cls_token, x), dim=1)

        # x = x + self.positional_embedding

        x = self.block_list(x)
        x = self.layernorm(x)
        cls_final = x[:, 0, :]
        x = self.output_linear(cls_final)
        # x = self.softmax(x)
        return x


if __name__ == '__main__':
    device = 0
    lr = 0.1
    epoch = 200
    batch_size = 1
    regulation_lambda = 0.0000000001
    net = Net().to(device=device)
    losses = []


    optimiser = torch.optim.SGD(net.parameters(), lr=lr)
    
    loss_func = torch.nn.CrossEntropyLoss()

    dataset = AudioDataset('audio.pkl')
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    for _ in range(epoch):
        for x, y in loader:
            x, y = x.to(device=device), y.to(device=device)
            y_ = net(x)

            # loss_reg = torch.tensor([0.0]).to(device=device)
            # for para in net.parameters():
                # loss_reg += torch.sum(para**2) * regulation_lambda

            loss = loss_func(y_, y)
            # loss += loss_reg
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            print(_, '|' ,epoch,"  " ,(y_.argmax(dim=1) == y).to(torch.float32).mean(), loss)
            losses.append(loss.detach().cpu().item())

    plt.plot(list(range(len(losses))), losses)
    plt.show()
    
    dataset = AudioTestDataset('test.pkl')
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