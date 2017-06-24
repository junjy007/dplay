import numpy as np
import os
import gym
import torch
import torch.nn as nn
from torch.autograd import Variable

USE_CUDA = torch.cuda.is_available()


def get_checkpoint_file(d, fid):
    return os.path.join(d, 'checkpoint{:d}'.format(fid))


def make_game():
    return gym.make('Pong-v0')


def to_tensor_f32(x):
    t_ = torch.from_numpy(np.float32(np.ascontiguousarray(x)))
    if USE_CUDA:
        t_ = t_.cuda()
    return t_


def to_tensor_int(x):
    # noinspection PyArgumentList
    t_ = torch.LongTensor(x)
    if USE_CUDA:
        t_ = t_.cuda()
    return t_


def to_numpy(t):
    if USE_CUDA:
        x = t.cpu().numpy()
    else:
        x = t.numpy()
    return x


# hyperparameters
H = 200  # number of hidden layer neurons
D = 80 * 80  # input dimensionality: 80x80 grid


# noinspection PyPep8Naming,PyPep8Naming
def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()


class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(D, H),
            nn.ReLU(inplace=True),
            nn.Linear(H, 2),
            nn.LogSoftmax()
        )
        if USE_CUDA:
            self.cuda()

    def forward(self, x):
        return self.net(x)

    def save(self, fname):
        torch.save(self.state_dict(), fname)

    def load(self, fname):
        self.load_state_dict(torch.load(fname))


class Policy:
    def __init__(self, cpdir, cpid):
        self.x0 = None
        self.net = PolicyNet()
        self.net.load(get_checkpoint_file(cpdir, cpid))
        self.rng = np.random.RandomState(42)

    def __call__(self, s):
        cur_x = prepro(s)
        x = cur_x - self.x0 if self.x0 is not None else np.zeros(D)
        self.x0 = cur_x

        # forward the policy network and sample an action from the returned probability
        x_tv = Variable(to_tensor_f32(x), requires_grad=False).unsqueeze(0)
        action_prob_log_tv = self.net.forward(x_tv)
        action_prob_log = to_numpy(action_prob_log_tv.data)[0]
        aprob = np.exp(action_prob_log)
        y = self.rng.choice(2, p=aprob)
        button_action = 2 if y == 0 else 3
        return button_action


def make_policy(cpdir, cpid):
    pol = Policy(cpdir, cpid)
    return pol
