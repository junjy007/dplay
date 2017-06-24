"""
Verify if my framework works -- start from "manual" implmentation, replace
components with our framework ones, until get where the difference comes from
"""
import numpy as np
import os
import gym
import torch
import torch.nn as nn
from torch.autograd import Variable
import json
import matplotlib.pyplot as plt

SAVE_DIR = 'checkpoints_aio'
if not os.path.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)
USE_CUDA = torch.cuda.is_available()
if not os.path.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)

infofile = os.path.join(SAVE_DIR, 'info.json')


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
batch_size = 10  # every how many episodes to do a param update?
learning_rate = 1e-5
gamma = 0.99  # discount factor for reward
decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2
resume = False  # resume from previous checkpoint?
render = False

# model initialization
D = 80 * 80  # input dimensionality: 80x80 grid
# if resume: else:

# grad_buffer = {k: np.zeros_like(v) for k, v in model.iteritems()}  # update buffers that add up gradients over a batch
# rmsprop_cache = {k: np.zeros_like(v) for k, v in model.iteritems()}  # rmsprop memory


def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    I = I[np.newaxis, :, :]
    return I.astype(np.float) # I.astype(np.float).ravel()


# noinspection PyPep8
def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        if r[t] != 0: running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

encoder_opts = {
    'input_channels': 1,
    'convs': [
        {'kernel_size':3, 'conv_kernels': 32, 'pool_factor': 2, 'relu': True},
        {'kernel_size':3, 'conv_kernels': 64, 'pool_factor': 2, 'relu': True}]
}
rng = np.random.RandomState(0)
from networks.encoders.conv_encoders import DeepConvEncoder
enc = DeepConvEncoder(encoder_opts)
print enc
from networks.decoders.policy_decoders import Decoder
dummy_in = prepro(rng.randint(0, 256, (210, 160, 3)))
feat_num = enc.get_feature_num({'height': dummy_in.shape[1], 'width': dummy_in.shape[2]})
dec = Decoder({"output_num": 2, "fc1_hidden_unit_num": 256, "input_num": feat_num})


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

def plot_performance(rs, ls, savepath, max_records=500):
    rs = np.asarray(rs)
    ls = np.asarray(ls)
    n = rs.shape[0]
    assert n == ls.shape[0]
    if n > max_records:
        idx = np.linspace(0, n - 1, max_records)
        idx_n = idx.astype(np.int)
        rs = rs[idx_n]
        ls = ls[idx_n]
    else:
        idx = np.arange(n)

    figs = [1, 2]
    data = [rs, ls]
    titles = ['Average Episode Reward', 'Average Episode Length']
    fnames = [os.path.join(savepath, s + '.png') for s in ['rewards', 'epilen']]

    for fig, d, tl, fn in zip(figs, data, titles, fnames):
        plt.figure(fig)
        plt.clf()
        plt.plot(d, 'b-')
        if n > 20:
            plt.xticks(np.linspace(0, n-1, 10))
        plt.title(tl)
        plt.savefig(fn)


env = gym.make("Pong-v0")
observation = env.reset()
prev_x = None  # used in computing the difference frame
xs, ys, drs = [], [], []
running_reward = None
reward_sum = 0
rng = np.random.RandomState(42)
loss = nn.NLLLoss(size_average=False)
recent_reward = 0
train_every_n_steps = 10
from networks.nets import RLNet
policy_net = RLNet(enc, dec)
print policy_net
T = 0
if os.path.exists(infofile):
    with open(infofile, 'r') as f:
        records = json.load(f)
    T = records['T']
    enc.load(records['latest_filename'] + '.encoder')
    dec.load(records['latest_filename'] + '.decoder')
else:
    records = {'latest_filename':'', 'reward_history':[], 'episode_length_history':[], 'T':0}

optimiser = torch.optim.RMSprop(policy_net.parameters(), lr=learning_rate, alpha=0.99)
optimiser.zero_grad()

class Hook:
    def __init__(self):
        self.n_ = 0
    def __call__(self, m, input, output):
        if self.n_ % 10000 == 0:
            print output.data
        self.n_ += 1

print_handle = dec._fc2.register_forward_hook(Hook())


while True:
    if render:
        env.render()

    # preprocess the observation, set input to network to be difference image
    cur_x = prepro(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros((1, 80, 80))
    prev_x = cur_x

    # forward the policy network and sample an action from the returned probability
    x_tv = Variable(to_tensor_f32(x), requires_grad=False).unsqueeze(0)
    action_prob_log_tv = policy_net.forward(x_tv)
    action_prob_log = to_numpy(action_prob_log_tv.data)[0]
    aprob = np.exp(action_prob_log)
    y = rng.choice(2, p=aprob)

    xs.append(x)  # observation
    ys.append(y)

    button_action = 2 if y==0 else 3
    observation, reward, done, info = env.step(button_action)
    drs.append(reward)  # record reward (has to be done after we call step() to get reward for previous action)

    if done:
        T += 1
        observation = env.reset()
        epx = np.stack(xs)
        epy = np.vstack(ys)
        epr = np.vstack(drs)
        recent_reward += epr.sum()
        xs, ys, drs = [], [], []
        epx_tv = Variable(to_tensor_f32(epx), requires_grad=False)
        epy_tv = Variable(to_tensor_int(epy), requires_grad=False)
        discounted_epr = discount_rewards(epr)
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)
        epr_tv = Variable(to_tensor_f32(discounted_epr), requires_grad=False)

        n_trails = epr_tv.size()[0]

        loss_value = 0.0
        for ti in range(n_trails):
            logP_i = policy_net(epx_tv[ti].unsqueeze(0))
            L_i = loss(logP_i, epy_tv[ti]) * epr_tv[ti] / float(n_trails * train_every_n_steps)
            L_i.backward()  # this will accumulate gradients in all network parameters
            # ** with applying weight = advantages[ti] **
            loss_value += L_i.data[0]  # L_i is now a singleton tensor


        if T % train_every_n_steps == 0:
            optimiser.step()
            optimiser.zero_grad()
            recent_reward /= float(train_every_n_steps)
            print "Train step {}, recent rewards: {}".format(T/train_every_n_steps, recent_reward)
            records['reward_history'].append(recent_reward)
            records['episode_length_history'].append(n_trails)
            recent_reward = 0


        print "Endgame {}, Finish {} trails".format(T, n_trails),
        print "Reward {}".format(epr.sum()),
        print "Last aprob: {}".format(aprob)

    if done and T % 100 == 0:
        records['latest_filename'] = os.path.join(SAVE_DIR, 'checkpoint{}'.format(T))
        records['T'] = T

        enc.save(records['latest_filename'] + '.encoder')
        dec.save(records['latest_filename'] + '.decoder')
        plot_performance(records['reward_history'], records['episode_length_history'], SAVE_DIR)
        with open(infofile, 'w') as f:
            json.dump(records, f)
