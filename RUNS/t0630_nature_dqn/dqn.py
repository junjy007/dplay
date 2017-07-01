import cv2
import gym
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import json

USE_CUDA = torch.cuda.is_available()
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor


def show_state(s):
    a = np.concatenate((s[0], s[2]))
    b = np.concatenate((s[1], s[3]))
    c = np.concatenate((a, b), axis=1)
    cv2.imshow("tmp", c)
    cv2.waitKey(30)


# noinspection PyUnresolvedReferences
class ReplayMemory:
    """
    We store frames in raw format, this way, we can put a large memory
    into GPU
    """

    def __init__(self, capacity, frm_height, frm_width,
                 batch_size=32, frames_per_state=4):
        self.sample_shape = (frames_per_state, frm_height, frm_width)
        self.obs_ = ByteTensor(capacity, *self.sample_shape)
        self.act_ = ByteTensor(capacity)
        self.reward_ = FloatTensor(capacity)
        self.term_ = np.asarray(np.zeros(capacity), dtype=np.bool)
        self.position = 0
        self.is_full = False
        self.capacity = capacity
        self.frm_height = frm_height
        self.frm_width = frm_width
        self.batch_size = batch_size
        self.frames_per_state = frames_per_state
        self.state_batch = FloatTensor(batch_size, *self.sample_shape)
        self.next_state_batch = FloatTensor(batch_size, *self.sample_shape)
        self.reward_batch = FloatTensor(batch_size)
        self.action_batch = LongTensor(batch_size)
        self.term_batch = ByteTensor(batch_size)
        self.rng = np.random.RandomState(42)

    def assert_input(self, s):
        """
        :param s:
        :type s: np.ndarray
        :return:
        """
        assert type(s) is np.ndarray
        assert s.dtype == np.uint8
        assert s.ndim == 3
        assert tuple(s.shape) == self.sample_shape

    def push(self, s0, a, r, s1, is_term, is_first_step=False):
        """
        :param s0: one preprocessed single-channel frame
        :type s0: np.ndarray, 1 x H x W, uint8
        :param a:
        :param r:
        :param s1:
        :param is_term: if terminated
        :param is_first_step: if this is the first step in an episode,
          if yes, both s0 and s1 must be stored. Otherwise, only s1
          needs to be saved
        :return:
        """
        self.assert_input(s0)
        self.assert_input(s1)
        if is_first_step:
            self.obs_[self.position, ...] = torch.from_numpy(s0)
            self.inc_position()

        self.obs_[self.position, ...] = torch.from_numpy(s1)
        self.act_[self.position] = a
        self.reward_[self.position] = r
        self.term_[self.position] = is_term
        self.inc_position()

    def inc_position(self):
        self.position += 1
        if self.position >= self.capacity:
            self.position = 0
            self.is_full = True

    def __len__(self):
        return self.capacity if self.is_full else self.position

    def make_sample_(self, bi, ci):
        """
        Take a sample from cache, do transform, save to batch
        :param bi: index in batch
        :param ci: index in cache

        Cache Structure:
          ...                T=True
          S_0   N/A   N/A    T_0     <-- new episode
          ...
          S_t   a_t-1 r_t    T_t
          S_t+1 a_t   r_t+1  T_t+1
          S_t+2 a_t+1 r_t+2  T_t+2
         *S_t+3 a_t+2 r_t+3  T_t+3
          S_t+4 a_t+3 r_t+4  T_t+4
          ...

        if ci-->*, T_t+3 must be false, the experience reads
          "At S_t+3, do a_t+3, get reward r_t+4, reach S_t+4, done? T_t+4"

        :return:

        NB,
        k: frames_per_batch, N: memory size, then
            k < ci < N-1
        we don't take care of rolling over the edge of the
        cycle list
        """
        if self.term_[ci]:
            return False  # No experience from a terminal state
        self.state_batch[bi] = self.obs_[ci].float() / 255.0
        self.next_state_batch[bi] = self.obs_[ci + 1].float() / 255.0
        self.action_batch[bi] = self.act_[ci + 1]
        self.reward_batch[bi] = self.reward_[ci + 1]
        self.term_batch[bi] = int(self.term_[ci + 1])
        return True

    def make_batch(self, batch_size=32):
        """
        :param batch_size: say, N samples
        :return:
          state_batch: N x frames-per-state
        """
        assert batch_size == self.batch_size
        for bi in range(batch_size):
            sample_done = False
            while not sample_done:
                sample_done = self.make_sample_(bi, self.rng.randint(0, len(self) - 1))

        return self.state_batch, self.action_batch, \
               self.reward_batch, self.next_state_batch, self.term_batch


class Preprocessor:
    def __init__(self, frm_height, frm_width):
        self.frm_height = frm_height
        self.frm_width = frm_width
        self.previous_frame = None

    def reset(self):
        self.previous_frame = None

    def proc(self, raw_frame):
        frm = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2GRAY)
        frm = cv2.resize(frm, (self.frm_height, self.frm_width))
        if self.previous_frame is None:
            self.previous_frame = frm
            frm_out = frm
        else:
            frm_out = np.maximum(self.previous_frame, frm)
            self.previous_frame = frm

        return frm_out

    def __call__(self, raw_frame):
        return self.proc(raw_frame)


class Env:
    def __init__(self, preprocessor, repeat_action):
        self.env = gym.make('Pong-v0').unwrapped
        self.preproc = preprocessor
        self.state = None
        self.repeat_action = repeat_action
        self.reset()

    def reset(self):
        self.preproc.reset()
        self.env.reset()
        raw_frm = self.env.render(mode='rgb_array')
        f = self.preproc(raw_frm)
        self.state = np.stack([f for _ in range(self.repeat_action)])
        return self.state

    def step(self, a):
        term = False
        rr = 0.0
        for i in range(self.repeat_action):
            if not term:
                _, r, term, _ = self.env.step(a)
                rr += r

            raw_frm = self.env.render(mode='rgb_array')
            f = self.preproc(raw_frm)
            self.state[i, ...] = f
        return self.state, rr, term


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
        print "Input {}".format(in_shape)
        dfeat = self.feat(Variable(torch.rand(*in_shape)))
        print "Feat {}".format(dfeat.size())
        nfeat = np.prod(dfeat.size()[1:])
        self.head = nn.Sequential(
            nn.Linear(nfeat, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 2)
        )
        if USE_CUDA:
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


class Policy:
    def __init__(self, model, action_num, start_eps, final_eps, eps_decay):
        self.eps = start_eps
        self.final_eps = final_eps
        self.eps_decay = eps_decay
        self.rng = np.random.RandomState(42)
        self.model = model
        self.action_num = action_num

    def __call__(self, state):
        if self.rng.rand() <  self.eps:
            a = self.rng.randint(self.action_num)
        else:
            state_t = torch.from_numpy(state).type(FloatTensor) / 255.0
            s = Variable(state_t.unsqueeze(0), volatile=True)
            v = self.model(s)
            max_ind = v.max(dim=1)[1]
            a = max_ind.data[0,0]
        if self.eps > self.final_eps:
            self.eps -= self.eps_decay
        return a


def learn_step(model, model_, memory, optim, loss_fn, batch_size=32, discount=0.99):
    state_batch, action_batch, reward_batch, next_state_batch, term_batch = \
        memory.make_batch(batch_size)

    pred_next_state_val = model_(Variable(next_state_batch, volatile=True)).max(dim=1)[0] * discount
    target_reward = Variable(torch.zeros(batch_size).type(FloatTensor))
    target_reward.data[...] = reward_batch
    target_reward[1-term_batch] += pred_next_state_val[1-term_batch]
    target_reward.volatile = False  # don't propagate this flag to loss

    aind = Variable(action_batch).unsqueeze(dim=1)
    pred_reward = model(Variable(state_batch)).gather(1, aind)

    loss = loss_fn(pred_reward, target_reward)
    optim.zero_grad()
    loss.backward()
    # clamp
    optim.step()
    return loss.data[0]



if __name__ == '__main__':
    F_HEIGHT = F_WIDTH = 84
    FRAMES_PER_STATE = 4
    BATCH_SIZE=32
    LEARNING_RATE=0.00025
    MOMENTUM=0.95
    CAPACITY= 250000
    MIN_MEM = 20000
    EPS_START = 1.0
    EPS_FINAL = 0.1
    EPS_DECAY = (EPS_START - EPS_FINAL) / 1e+6
    STEP_START = 0
    RECORD_EVERY_N_STEPS = 10000
    SAVE_EVERY_N_STEPS   = 50000
    SYNC_EVERY_N_STEPS   = 10000

    preproc = Preprocessor(F_HEIGHT, F_WIDTH)
    env = Env(preproc, FRAMES_PER_STATE)
    mem = ReplayMemory(CAPACITY, F_HEIGHT, F_WIDTH, BATCH_SIZE, FRAMES_PER_STATE)
    model = DQN((BATCH_SIZE, FRAMES_PER_STATE, F_HEIGHT, F_WIDTH))
    loss_history = []
    reward_history = []
    duration_history = []

    try:
        with open('save/latest.json', 'r') as f:
            status = json.load(f)
            EPS_START = status['eps']
            STEP_START = status['steps']
            model.load(status['latest_checkpoint'])
            loss_history = status['loss_history']
            reward_history = status['reward_history']
            duration_history = status['duration_history']
    except:
        pass


    policy = Policy(model, 2, EPS_START, EPS_FINAL, EPS_DECAY)
    model_ = DQN((BATCH_SIZE, FRAMES_PER_STATE, F_HEIGHT, F_WIDTH))
    model.clone_to(model_)

    # loss_fn = torch.nn.SmoothL1Loss()
    loss_fn = torch.nn.MSELoss()
    optim = torch.optim.RMSprop(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    state = env.reset()
    is_first_step = True
    game_r = []
    recent_losses = []
    gr_ = 0
    for i in range(STEP_START,  50000000):
        action = policy(state)
        action_button = 2 if action==0 else 3
        next_state, r, term = env.step(action_button)
        round_done = (r != 0)  # for Pong
        mem.push(state, action, r, next_state, round_done, is_first_step)
        gr_ += r
        is_first_step = False

        if term:
            state = env.reset()
            game_r.append(gr_)
            gr_ = 0
        else:
            state = next_state

        if round_done or term:
            is_first_step = True

        # learning
        if i % 1000 ==0:
            print '.',

        if len(mem) > MIN_MEM:
            ls = learn_step(model, model_, mem, optim, loss_fn)
            recent_losses.append(ls)

            if i % SYNC_EVERY_N_STEPS == 0:
                model.clone_to(model_)
                print "Synced"

            if i % RECORD_EVERY_N_STEPS == 0:
                rr_ = np.mean(game_r)
                n = len(game_r)
                dur = float(RECORD_EVERY_N_STEPS) / n
                ll_ = np.mean(recent_losses)
                print "\rStep {}, Loss {}, Reward {}, Avg Dur {}".format(i, ll_, rr_, dur)
                loss_history.append(ll_)
                reward_history.append(rr_)
                duration_history.append(dur)
                recent_losses = []
                game_r = []

            if i % SAVE_EVERY_N_STEPS == 0:
                latest_checkpoint = 'save/cp{:d}.torchmodel'.format(i)
                model.save(latest_checkpoint)
                with open('save/latest.json', 'w') as f:
                    json.dump({
                        'latest_checkpoint':latest_checkpoint,
                        'eps':policy.eps,
                        'steps':i+1,
                        'loss_history':loss_history,
                        'reward_history':reward_history,
                        'duration_history': duration_history
                    }, f)
                print "Checkpoint saved to {}".format(latest_checkpoint)


























