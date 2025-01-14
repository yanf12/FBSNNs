


import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.fc1 = nn.Linear(in_channels, out_channels)
        self.fc2 = nn.Linear(out_channels, out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.fc1(x))
        out = self.fc2(out)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, in_channels, num_classes, num_layers, width):
        super(ResNet, self).__init__()
        self.width = width
        self.fc1 = nn.Linear(in_channels, self.width)
        self.relu = nn.ReLU()


        self.resblocks = nn.ModuleList()
        for _ in range(num_layers):
            self.resblocks.append(ResBlock(self.width, self.width))

        self.fc2 = nn.Linear(self.width, num_classes)

    def forward(self, x):
        x = self.relu(self.fc1(x))

        for resblock in self.resblocks:
            x = resblock(x)

        x = self.fc2(x)
        return x


class neural_net(nn.Module):
    def __init__(self, pathbatch=100, n_dim=100 + 1, n_output=1, num_layers=3, width=256):
        super(neural_net, self).__init__()
        self.pathbatch = pathbatch
        self.num_layers = num_layers
        self.width = width

        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(n_dim, self.width))
        for _ in range(num_layers - 1):
            self.fc_layers.append(nn.Linear(self.width, self.width))
        self.out = nn.Linear(self.width, n_output)

        self.activation = torch.sin

        with torch.no_grad():
            for layer in self.fc_layers:
                torch.nn.init.xavier_uniform(layer.weight)

    def forward(self, state, train=False):
        for i in range(self.num_layers):
            state = self.activation(self.fc_layers[i](state))
        fn_u = self.out(state)
        return fn_u


import numpy as np

import numpy as np

class errormeasure:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    def calculate_mse(self):
        return np.mean((self.y_true - self.y_pred) ** 2)

    def calculate_mape(self):
        return np.mean(np.abs((self.y_true - self.y_pred) / self.y_true)) * 100

    def calculate_mae(self):
        return np.mean(np.abs(self.y_true - self.y_pred))

