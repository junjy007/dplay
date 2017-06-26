"""
Deep Q learning
- 2 convolutional layers
- without secondary net for TD

24 June 2017
"""
import gym
import numpy as np
import torch.nn as nn
import torch
from collections import deque
from torch.autograd import Variable
from dplay_utils.tensordata import to_tensor_f32, to_numpy, to_tensor_int, does_use_cuda
from experience_managers.preprocessors import GymAtariFramePreprocessor_Stacker

opts = {
    "dqn":{
        "discount": 0.99
    },
    "train": {
        "lr": 1e-2,
        "sync_nets_every_n_steps": 100,
        "batch_size": 50,
        "train_every_n_steps": 100,
        "save_every_n_train_steps": 1000
    },
    "net": {
        "input_channels": 4,
        "input_size": {
            "height": None,
            "width": None
        },
        "output_num": None,
        "convs": [
            {"kernel_size":3, "conv_kernels": 32, "pool_factor": 2, "relu": True},
            {"kernel_size":3, "conv_kernels": 64, "pool_factor": 2, "relu": True}
        ]
    },
    "explore": {
        "init_randomness": 0.95,
        "final_randomness": 0.01,
        "explore_steps": 10000,
        "record_size": 1000000,
        "minimum_records": int(10000)
    }
}

class QNet(nn.Module):
    def __init__(self, opts):
        super(QNet, self).__init__()

        self.conv_layers = []
        self.input_channels = opts['input_channels']
        in_kernels = self.input_channels

        for cf in opts['convs']:
            ks = cf['kernel_size']
            kn = cf['conv_kernels']

            lay_ = []
            lay_.append(nn.Conv2d(in_channels=in_kernels,
                                  out_channels=kn,
                                  kernel_size=ks, padding=(ks - 1) / 2))
            if cf['relu']:
                lay_.append(nn.ReLU())
            lay_.append(nn.MaxPool2d(kernel_size=cf['pool_factor']))
            self.conv_layers.append(nn.Sequential(*lay_))
            in_kernels = kn

        self.feature = nn.Sequential(*self.conv_layers)
        if does_use_cuda():
            self.feature.cuda()

        self.num_features = None
        nf = self.get_feature_num(opts['input_size'])
        self.dec = nn.Sequential(
            nn.Linear(in_features=nf, out_features=256),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=opts["output_num"])
        )

        if does_use_cuda():
            self.cuda()

    def get_feature_num(self, image_size=None):
        """
        :param image_size: info about state variable of images, see Preprocessor and ExperienceMemory
            ['height/width']
        """
        if self.num_features is None:
            assert not (image_size is None), "Image size must be given in the first time"
            dummy_input_t = torch.rand(1, self.input_channels,
                                       image_size['height'], image_size['width'])
            if does_use_cuda():
                dummy_input_t = dummy_input_t.cuda()
            dummy_input = Variable(dummy_input_t, requires_grad=False)
            dummy_feature = self.feature(dummy_input)
            nfeat = np.prod(dummy_feature.size()[1:])
            self.num_features = nfeat
        return self.num_features

    def forward(self, x):
        y = self.feature(x)
        y = y.view(-1, self.get_feature_num())
        y = self.dec(y)
        return y

    def save(self, fname):
        torch.save(self.state_dict(), fname)

    def load(self, fname):
        self.load_state_dict(torch.load(fname))

class ExplorePolicy:
    def __init__(self, action_num, init_explore, explore_steps, final_explore=0.0):
        self.explore_decay = (init_explore - final_explore) / float(explore_steps)
        self.rng = np.random.RandomState(42)
        self.explore_prob = init_explore
        self.min_explore_prob = final_explore
        self.action_num = action_num

    def __call__(self, action_val):
        assert action_val.size == self.action_num
        if self.rng.rand() > self.explore_prob:
            return np.argmax(action_val)
        else:
            return self.rng.randint(self.action_num)

    def reduce_explore(self):
        if self.explore_prob >= self.min_explore_prob + self.explore_decay:
            self.explore_prob -= self.explore_decay

class QLearn:
    def __init__(self, net):
        """
        :param net:
        :type net: nn.Module
        """
        self.net = net
        self.optim = torch.optim.Adagrad(net.parameters(), lr=opts['train']['lr'])

    def step(self, batch):
        s0 = Variable(to_tensor_f32(batch['states']), requires_grad=False)
        s1 = Variable(to_tensor_f32(batch['states_new']), requires_grad=False)
        a  = Variable(to_tensor_int(batch['actions']), requires_grad=False).unsqueeze(1)
        r = Variable(to_tensor_f32(batch['rewards']), requires_grad=False).unsqueeze(1)
        av0 = self.net(s0)
        av0 = torch.gather(av0, 1, a)
        av1 = self.net(s1)
        av1 = torch.max(av1, dim=1)[0]

        self.loss = torch.mean(((r + av1) - av0) ** 2)
        self.optim.zero_grad()
        self.loss.backward()
        self.optim.step()
        return to_numpy(self.loss.data)

class Records:
    def __init__(self, record_size):
        self.rec={
            'states': deque(maxlen=record_size),
            'states_new': deque(maxlen=record_size),
            'actions': deque(maxlen=record_size),
            'rewards': deque(maxlen=record_size)
        }
        self.rng = np.random.RandomState(42)

    def add(self, s0, a, r, s1):
        self.rec['states'].append(s0)
        self.rec['states_new'].append(s1)
        self.rec['actions'].append(a)
        self.rec['rewards'].append(r)

    def get_batch(self, batch_size):
        n = len(self.rec['states'])
        ind = self.rng.randint(0, n, batch_size)   # don't consider the tiny chance of duplicate
        batch = {}
        for k in self.rec.keys():
            batch[k] = np.ascontiguousarray([self.rec[k][i_] for i_ in ind])
        return batch

env = gym.make('Pong-v0')
ACTION_NUM = env.action_space.n
s = env.reset()
preproc = GymAtariFramePreprocessor_Stacker(stack_frames=opts['net']['input_channels'])
s = preproc(s)
opts['net']['input_size']['height'] = s.shape[1]  # preprocessing returns (1) x channels x H x W
opts['net']['input_size']['width']  = s.shape[2]
opts['net']['output_num']  = ACTION_NUM
qnet = QNet(opts['net'])

policy = ExplorePolicy(action_num=ACTION_NUM,
                       init_explore=opts['explore']['init_randomness'],
                       explore_steps=opts['explore']['explore_steps'],
                       final_explore=opts['explore']['final_randomness'])

records = Records(record_size=opts['explore']['record_size'])

dc_ = opts['dqn']['discount']
T = 0
ql = QLearn(qnet)

recent_reward = 0.0
recent_reward_lt = 0.0
train_steps = 0
while True:
    avals = to_numpy(qnet.forward(Variable(to_tensor_f32(s), requires_grad=False).unsqueeze(0)).data).squeeze()

    a = policy(avals)
    s_new, r, term, _ = env.step(a)
    s_new = preproc(s_new)

    records.add(s, a, r, s_new)

    recent_reward += r
    if term:
        s = preproc(env.reset())
        recent_reward_lt = recent_reward * 0.01 + recent_reward_lt*0.99
        print "train steps {}: {} / {} (lt)".format(train_steps, recent_reward, recent_reward_lt)
        recent_reward = 0.0
    else:
        s = s_new
    T += 1

    if T > opts['explore']['minimum_records']:
        if T % opts['train']['train_every_n_steps'] == 0:
            b = records.get_batch(opts['train']['batch_size'])
            ql.step(b)

            train_steps += 1
            if train_steps % opts['train']['save_every_n_train_steps'] == 0:
                qnet.save('checkpoints/checkpoint{:06d}.torchnet'.format(train_steps))



