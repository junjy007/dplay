"""
29 June 2017
When a round ends, the next state is meaningless. So we add back the "selection"
part of Q-learn:

target[non-terminal] = target-net(next_state).max + r[non-terminal]
target[terminal] = r[terminal]
"""
import cv2
import gym
import time
import math
import random
import numpy as np
import json
import matplotlib.pyplot as plt
from collections import namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
from dplay_utils.tensordata import *
# noinspection PyPep8Naming
import torch.nn.functional as F
from torch.autograd import Variable
from collections import deque

######################################################################
# CLASSES AND TYPES
######################################################################
# FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
# LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
# ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
# Tensor = FloatTensor

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

hparam = {
    "discount": 0.99,
    "optim": {
        "optimiser": "RMSprop",
        "opts": {"lr": 0.00025, "momentum": 0.95}
    },
    "sync_model_every_n_steps": 10000,
    "max_train_steps": int(5e+7),
    "relay_memory_size": int(2e5),
    "batch_size": 128,
    "eps_start": 1.0,
    "eps_end": 0.1,
    "eps_decay": int(1e+6),
    "act_n_train_every_n_frames": 4,
    "save_every_n_steps": 50000,
    "demo_every_n_episodes": 100,
    "state_frame_size": (84, 84)
}


# DEPREC
class ReplayMemory2(object):
    """
    In-memory cache
    """

    def __init__(self, capacity, batch_size, s0):
        self.capacity = capacity
        self.batch_size = batch_size
        self.memory = []
        self.position = 0
        self.states = np.zeros((capacity,) + s0.shape[1:], dtype=np.uint8)
        self.next_states = np.zeros((capacity,) + s0.shape[1:], dtype=np.uint8)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int)
        self.state_batch = np.zeros((batch_size,) + s0.shape[1:], dtype=np.float32)
        self.next_state_batch = np.zeros((batch_size,) + s0.shape[1:], dtype=np.float32)
        self.reward_batch = np.zeros(batch_size, dtype=np.float32)
        self.action_batch = np.zeros(batch_size, dtype=np.int)
        self.rng = np.random.RandomState(42)
        self.full = False

    def push(self, s0, a, r, s1):
        self.states[self.position, ...] = s0[0, ...]
        self.next_states[self.position, ...] = s1[0, ...]
        self.rewards[self.position] = r
        self.actions[self.position] = a

        self.position += 1
        if self.position == self.capacity:
            self.full = True
            self.position = 0

    def sample(self, batch_size):
        assert batch_size == self.batch_size
        l = self.capacity if self.full else self.position
        ii = self.rng.randint(0, l, (self.batch_size,))
        for ci, i in enumerate(ii):
            self.state_batch[ci] = self.states[i].astype(np.float32) / 255.0
            self.next_state_batch[ci] = self.next_states[i].astype(np.float32) / 255.0
            self.action_batch[ci] = self.actions[i]
            self.reward_batch[ci] = self.rewards[i]
        return self.state_batch, self.action_batch, self.reward_batch, self.next_state_batch

    def __len__(self):
        return self.capacity if self.full else self.position


class ReplayMemory(object):
    """
    Normal cache
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.states = []
        self.next_states = []
        self.rewards = []
        self.actions = []
        self.continues = []
        self.state_batch = []
        self.next_state_batch = []
        self.reward_batch = []
        self.action_batch = []
        self.cont_batch = []
        self.rng = np.random.RandomState(42)
        self.full = False

    def push(self, s0, a, r, s1, c):
        if not self.full:
            self.states.append(s0)
            self.next_states.append(s1)
            self.rewards.append(r)
            self.actions.append(a)
            self.continues.append(c)
        else:
            self.states[self.position] = s0
            self.next_states[self.position] = s1
            self.rewards[self.position] = r
            self.actions[self.position] = a
            self.continues[self.position] = c

        self.position += 1
        if self.position == self.capacity:
            self.full = True
            self.position = 0

    def sample(self, batch_size):
        l = self.capacity if self.full else self.position
        ii = self.rng.randint(0, l, (batch_size,))
        if len(self.state_batch) >= batch_size:
            for ci, i in enumerate(ii):
                self.state_batch[ci] = self.states[i]
                self.next_state_batch[ci] = self.next_states[i]
                self.action_batch[ci] = self.actions[i]
                self.reward_batch[ci] = self.rewards[i]
                self.cont_batch[ci] = self.continues[i]
        else:
            self.state_batch = [self.states[i] for i in ii]
            self.next_state_batch = [self.next_states[i] for i in ii]
            self.reward_batch = [self.rewards[i] for i in ii]
            self.action_batch = [self.actions[i] for i in ii]
            self.cont_batch = [self.continues[i] for i in ii]

        return np.concatenate(self.state_batch, axis=0).astype(np.float32) / 255.0, \
               np.ascontiguousarray(self.action_batch, dtype=np.int), \
               np.ascontiguousarray(self.reward_batch, dtype=np.float32), \
               np.concatenate(self.next_state_batch, axis=0).astype(np.float32) / 255.0, \
               np.ascontiguousarray(self.cont_batch, dtype=np.uint8)

    def __len__(self):
        return len(self.states)


class DQN(nn.Module):
    def __init__(self, in_shape):
        super(DQN, self).__init__()
        self.in_shape = in_shape
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        # self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        # self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # self.bn3 = nn.BatchNorm2d(32)
        self.feat = nn.Sequential(
            self.conv1, nn.ReLU(inplace=True),
            self.conv2, nn.ReLU(inplace=True),
            self.conv3, nn.ReLU(inplace=True)
        )
        # self.feat = nn.Sequential(
        #     self.conv1, nn.ReLU(inplace=True),
        #     self.conv2, nn.ReLU(inplace=True),
        # )
        print "Input {}".format(in_shape)
        dfeat = self.feat(Variable(torch.rand(*in_shape)))
        print "Feat {}".format(dfeat.size())
        nfeat = np.prod(dfeat.size()[1:])
        self.head = nn.Sequential(
            nn.Linear(nfeat, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 2)
        )
        if does_use_cuda():
            self.cuda()

    def forward(self, x):
        x = self.feat(x)
        return self.head(x.view(x.size(0), -1))

    def save(self, fname):
        torch.save(self.state_dict(), fname)

    def load(self, fname):
        self.load_state_dict(torch.load(fname))

    def clone_to(self, another):
        assert self.in_shape == another.in_shape, \
            "Try to clone model processing {} to one processing {}".format(
                self.in_shape, another.in_shape)
        for tp, sp in zip(another.parameters(), self.parameters()):
            tp.data[...] = sp.data
        another.zero_grad()
        return another


# noinspection PyShadowingNames
def get_screen(env):
    screen = env.render(mode='rgb_array')
    screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
    screen = cv2.resize(screen, dsize=hparam['state_frame_size'])
    return screen

def get_screen2(s):
    screen = cv2.cvtColor(s, cv2.COLOR_RGB2GRAY)
    screen = cv2.resize(screen, dsize=hparam['state_frame_size'])
    return screen


def env_step(env, act, m):
    accum_state = []
    accum_reward = 0.0
    term = None
    for i in range(m):
        if act is not None:
            _, r, term, _ = env.step(act)
            s = get_screen(env)
        else:
            s = get_screen(env)
            r = 0
            term = True
        accum_state.append(s)
        accum_reward += r

        if term:  # if terminated, then stops acting
            act = None

    ss = np.stack(accum_state)[np.newaxis, ...]

    return ss, accum_reward, term

def env_step2(env, act, m):
    accum_state = []
    accum_reward = 0.0
    term = None
    s_ = None
    for i in range(m):
        if act is not None:
            s, r, term, _ = env.step(act)
            s = get_screen2(s)
        else:
            if s_ is None:
                s_, r, term, _ = env.step(0)
            s = get_screen2(s_)
            r = 0
            term = True
        accum_state.append(s)
        accum_reward += r

        if term:  # if terminated, then stops acting
            act = None

    ss = np.stack(accum_state)[np.newaxis, ...]

    return ss, accum_reward, term

# noinspection PyPep8Naming,PyShadowingNames
def make_DQN_net(env):
    """
    :param env: environment
    """
    s0 = env_step(env, None, hparam['act_n_train_every_n_frames'])[0]
    model = DQN(s0.shape)
    return model


def save(net, rec_r, rec_d, steps):
    conf_fname = 'save/latest.json'
    cp_fname = 'save/cp{:06d}.torchmodel'.format(steps)
    with open(conf_fname, 'w') as f:
        json.dump({'latest_checkpoint': cp_fname,
                   'record_rewards': rec_r,
                   'record_durations': rec_d,
                   'steps': steps}, f)
    net.save(cp_fname)


def load(net):
    conf_fname = 'save/latest.json'
    try:
        with open(conf_fname, 'r') as f:
            c = json.load(f)
        fn = c['latest_checkpoint']
        net.load(fn)
    except:
        return None

    return c


# noinspection PyPep8,PyShadowingNames
class Policy:
    def __init__(self, opts, steps=0):
        self.de = (opts['eps_start'] - opts['eps_end']) / float(opts['eps_decay'])
        self.rng = np.random.RandomState(42)
        self.eps = opts['eps_start'] - self.de * steps
        self.eps_end = opts['eps_end']

    # noinspection PyArgumentList
    def __call__(self, model, state):
        if self.rng.rand() > self.eps:
            state_t = to_tensor_f32(state) / 255.0
            state_t = Variable(state_t, volatile=True)
            a_ = model(state_t).data.max(1)[1]
            a_ = a_[0, 0]
        else:
            a_ = self.rng.randint(2)
        if self.eps > self.eps_end:
            self.eps -= self.de
        return a_


# DEPREC
def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = hparam['eps_end'] + (hparam['eps_start'] - hparam['eps_end']) * \
                                        math.exp(-1. * steps_done / hparam['eps_decay'])
    steps_done += 1
    if sample > eps_threshold:
        a_ = model.forward(
            Variable(state, volatile=True).type(FloatTensor)).data.max(1)[1]
    else:
        a_ = LongTensor([[random.randrange(2)]])
    return a_


def plot_records(rec, fname):
    plt.figure(2)
    plt.clf()

    plt.title('Training...')
    plt.xlabel('Episodes')
    plt.ylabel(fname)
    if len(rec) > 200:
        s = len(rec) // 100
    else:
        s = 1
    xx = range(0, len(rec), s)
    yy = rec[::s]
    plt.plot(xx, yy)
    plt.savefig(fname + '.png')
    # Take 100 episode averages and plot them too
    # if len(durations_t) >= 100:
    #     means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
    #     means = torch.cat((torch.zeros(99), means))
    #     plt.plot(means.numpy())

    # plt.pause(0.001)  # pause a bit so that plots are updated
    # if is_ipython:
    #     display.clear_output(wait=True)
    #     display.display(plt.gcf())


######################################################################
# Training loop
# ^^^^^^^^^^^^^

# noinspection PyShadowingNames
def optimize_model(model, model_a, optimiser, memory):
    if len(memory) < hparam['batch_size'] * 10:
        return

    def fn(x, vl=False):
        y = Variable(torch.from_numpy(x), volatile=vl)
        return y.cuda() if does_use_cuda() else y

    state_batch, action_batch, reward_batch, next_state_batch, cont_batch = map(
        fn, memory.sample(hparam['batch_size']), (False, False, False, True, False))
    action_batch = action_batch.unsqueeze(1)  # batch_size x 1, to be used for indexing

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = model(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = model_a(next_state_batch).max(1)[0]
    next_state_values[1-cont_batch] = 0
    next_state_values.volatile = False

    # Now, we don't want to mess up the loss with a volatile flag, so let's
    # clear it. After this, we'll just end up with a Variable that has
    # requires_grad=False
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * hparam['discount']) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    # Optimize the model
    optimiser.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)
    optimiser.step()


######################################################################
#
# Below, you can find the main training loop. At the beginning we reset
# the environment and initialize the ``state`` variable. Then, we sample
# an action, execute it, observe the next screen and the reward (always
# 1), and optimize our model once. When the episode ends (our model
# fails), we restart the loop.
#
# Below, `num_episodes` is set small. You should download
# the notebook and run lot more epsiodes.

if __name__ == '__main__':
    env = gym.make('Pong-v0').unwrapped
    #env = gym.make('CartPole-v0').unwrapped
    env.reset()

    env_step_fn = env_step

    state, _, _ = env_step_fn(env, None, hparam['act_n_train_every_n_frames'])
    model = DQN(state.shape)
    model_a = DQN(state.shape)
    optimiser = optim.RMSprop(model.parameters(), **hparam['optim']['opts'])
    memory = ReplayMemory(hparam['relay_memory_size'])

    steps_done = 0
    recent_reward = 0.0
    rec_reward = []
    rec_duration = []
    rrlt = None
    sv = load(model)
    if sv is not None:
        rec_reward = sv['record_rewards']
        rec_duration = sv['record_durations']
        steps_done = sv['steps']

    policy = Policy(hparam, steps_done)
    last_steps_done = steps_done
    model.clone_to(model_a)
    timecost = {
        'data': 0.0,
        'forward': 0.0,
        'backward-step': 0.0,
        'collect-mem': 0.0
    }
    do_render = False
    while steps_done < hparam['max_train_steps']:
        if do_render:
            env.render()
            time.sleep(0.03)
        action = policy(model, state)
        action_button = 2 if action == 0 else 3
        #action_button = 0 if action == 0 else 1
        new_state, reward, term = env_step_fn(env, action_button, hparam['act_n_train_every_n_frames'])
        # For pong, if reward != 0, a round has ended and we have a winner.
        does_round_cont = (reward == 0)

        memory.push(state, action, reward, new_state, does_round_cont)



        optimize_model(model, model_a, optimiser, memory)
        if steps_done % hparam['sync_model_every_n_steps'] == 0:
            print "Sync ..."
            model.clone_to(model_a)

        if term:
            env.reset()
            state, _, _ = env_step_fn(env, None, hparam['act_n_train_every_n_frames'])
        else:
            state = new_state

        steps_done += 1
        recent_reward += reward

        if term:
            rec_reward.append(recent_reward)
            rec_duration.append(steps_done - last_steps_done)
            # plot_records(rec_reward, "save/Reward")
            # plot_records(rec_duration, "save/Duration")
            rrlt = recent_reward if rrlt is None else recent_reward * 0.01 + rrlt * 0.99
            print "Episodes {} Recent/Total-Train-Steps {} / {}, R {}, R(lt) {:.2f}, Memory {}".format(
                len(rec_duration), rec_duration[-1], steps_done, recent_reward, rrlt, len(memory))
            # print "Time cost",
            # for k_ in timecost:
            #     print "{}:{:.6f}, ".format(k_, timecost[k_]),
            #     timecost[k_] = 0.0
            # print
            recent_reward = 0.0
            last_steps_done = steps_done
            do_render = (len(rec_duration) % hparam['demo_every_n_episodes'] == 0)

        if steps_done % hparam['save_every_n_steps'] == 0:
            save(model, rec_reward, rec_duration, steps_done)

    print 'Complete'
    # env.render(close=True)
    # env.close()
