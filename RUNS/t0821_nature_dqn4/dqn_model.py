import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class DQN(nn.Module):
    def __init__(self, in_channels=4, num_actions=18, in_h=84, in_w=84):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # self.conv_layers = nn.Sequential(
        #     self.conv1, nn.ReLU()
        #     ...
        # )
        # 7 x 7 <== 84 x 84, after conv 1/2/3
        dummy = Variable(torch.FloatTensor(1, in_channels, in_h, in_w), volatile=True)
        dummy_out = self.conv3(self.conv2(self.conv1(dummy)))
        self.out_feat_num = np.prod(dummy_out.size()[1:])

        self.fc4 = nn.Linear(self.out_feat_num, 512)
        self.fc5 = nn.Linear(512, num_actions)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4( x.view(x.size(0), -1) ))
        y = self.fc5(x)
        return y


    def save(self, fname):
        torch.save(self.state_dict(), fname)


    def load(self, fname):
        self.load_state_dict(torch.load(fname))

