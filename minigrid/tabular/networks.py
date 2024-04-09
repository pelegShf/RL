import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class network(nn.Module):
    def __init__(self):
        super(network,self).__init__()


        self.conv1 = nn.Conv2d(1, 16, kernel_size=(8, 8), stride=4)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(16, 32, kernel_size=(4, 4), stride=2)
        self.act2 = nn.ReLU()

        self.flat = nn.Flatten()

        self.fc1 = nn.Linear(2592, 256)  # Corrected input size
        self.act3 = nn.ReLU()
        self.fc2 = nn.Linear(256, 3)
        self.act4 = nn.ReLU()

        # self.apply(self.init_weights)


    def init_weights(self, m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)

        x = self.act1(x)
        # print(x.shape)

        x = self.conv2(x)
        # print(x.shape)

        x = self.act2(x)
        x = self.flat(x)

        # x = x.view(x.size(0), -1)


        x = self.fc1(x)
        x = self.act3(x)
        x = self.fc2(x)

        return x
