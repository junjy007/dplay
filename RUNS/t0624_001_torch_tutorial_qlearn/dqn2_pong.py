import gym
import math
import random
import numpy as np
import json
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from copy import deepcopy
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
# noinspection PyPep8Naming
import torch.nn.functional as F
from torch.autograd import Variable
# noinspection PyPep8Naming
import torchvision.transforms as T

env = gym.make('Pong-v0').unwrapped

######################################################################
# CLASSES AND TYPES
######################################################################

# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, dummy):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=4, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=16, stride=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.feat = nn.Sequential(
            self.conv1, self.bn1, nn.ReLU(inplace=True),
            self.conv2, self.bn2, nn.ReLU(inplace=True),
            self.conv3, self.bn3, nn.ReLU(inplace=True)
        )
        # self.feat = nn.Sequential(
        #     self.conv1, nn.ReLU(inplace=True),
        #     self.conv2, nn.ReLU(inplace=True),
        # )
        print "Input {}".format(dummy.size())
        dfeat = self.feat(Variable(torch.rand(*dummy.size())))
        print "Feat {}".format(dfeat.size())
        nfeat = np.prod(dfeat.size()[1:])
        self.head = nn.Linear(nfeat, 2)

    def forward(self, x):
        x = self.feat(x)
        return self.head(x.view(x.size(0), -1))

    def save(self, fname):
        torch.save(self.state_dict(), fname)

    def load(self, fname):
        self.load_state_dict(torch.load(fname))

resize = T.Compose([T.ToPILImage(),
                    T.Scale(80, interpolation=Image.NEAREST),
                    T.ToTensor()])


# noinspection PyShadowingNames
def get_screen(env):
    screen = env.render(mode='rgb_array')[35:195,:, 0][:,:,np.newaxis]
    screen[screen==109] = 0
    screen[screen==144] = 0
    screen[screen!=0] = 1
    return resize(screen).unsqueeze(0).type(Tensor)


# noinspection PyPep8Naming,PyShadowingNames
def make_DQN_net(env):
    """
    :param env: environment
    """
    screen0 = get_screen(env)
    screen1 = get_screen(env)
    s = screen1 - screen0
    return DQN(s)


def save(ep_num, net, rec_r, rec_d, steps):
    conf_fname = 'save/latest.json'
    cp_fname = 'save/cp{:06d}.torchmodel'.format(ep_num)
    with open(conf_fname, 'w') as f:
        json.dump({'latest_checkpoint': cp_fname,
                   'ep_num': ep_num,
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



######################################################################
# Prepare training.
######################################################################
env.reset()
# plt.figure()
# plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(),
#           interpolation='none')
# plt.title('Example extracted screen')
# plt.show()


######################################################################
# Training

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 50000
SAVE_EVERY_N_EPISODES = 1000

model = make_DQN_net(env)

if use_cuda:
    model.cuda()

optimizer = optim.RMSprop(model.parameters())
memory = ReplayMemory(20000)
steps_done = 0


# noinspection PyPep8,PyShadowingNames
def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                              math.exp(-1. * steps_done / EPS_DECAY)
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
    if len(rec)>200:
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

last_sync = 0


# noinspection PyShadowingNames
def optimize_model():
    global last_sync
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = ByteTensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)))

    # We don't want to backprop through the expected action values and volatile
    # will save us on temporarily changing the model parameters'
    # requires_grad to False!
    non_final_next_states = Variable(torch.cat([s for s in batch.next_state
                                                if s is not None]),
                                     volatile=True)
    state_batch = Variable(torch.cat(batch.state))
    action_batch = Variable(torch.cat(batch.action))
    reward_batch = Variable(torch.cat(batch.reward))

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = model(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = Variable(torch.zeros(BATCH_SIZE).type(Tensor))
    next_state_values[non_final_mask] = model(non_final_next_states).max(1)[0]
    # Now, we don't want to mess up the loss with a volatile flag, so let's
    # clear it. After this, we'll just end up with a Variable that has
    # requires_grad=False
    next_state_values.volatile = False
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


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

num_episodes = 20000000
recent_reward = 0.0
rec_reward = []
rec_duration = []
rrlt = None
start_ep = 0
sv = load(model)
if sv is not None:
    rec_reward = sv['record_rewards']
    rec_duration = sv['record_durations']
    start_ep = sv['ep_num']
    steps_done = sv['steps']

for i_episode in range(start_ep, num_episodes):
    # Initialize the environment and state
    env.reset()
    last_screen = get_screen(env)
    current_screen = get_screen(env)
    # plt.imshow(current_screen.cpu().numpy()[0, 0])
    # plt.show()
    # exit(1)
    state = current_screen - last_screen
    for t in count():
        # Select and perform an action
        action = select_action(state)
        _, reward, done, _ = env.step(2 if action[0, 0]==0 else 3)
        recent_reward += reward
        reward = Tensor([reward])

        # Observe new state
        last_screen = current_screen
        current_screen = get_screen(env)
        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model()

        if done:
            rec_reward.append(recent_reward)
            rec_duration.append(t+1)
            plot_records(rec_reward, "Reward")
            plot_records(rec_duration, "Duration")
            rrlt = recent_reward if rrlt is None else recent_reward*0.01 + rrlt*0.99
            print "Epi {}, steps {}, R {}, R(lt) {:.2f}".format(i_episode, steps_done, recent_reward, rrlt)
            recent_reward = 0.0
            if i_episode % SAVE_EVERY_N_EPISODES == 0:
                save(i_episode, model, rec_reward, rec_duration, steps_done)
            break

print('Complete')
env.render(close=True)
env.close()
