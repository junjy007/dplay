import os
import torch
import torch.nn as nn
import json
from torch.autograd import Variable
import numpy as np
import imp  # Python 2
from collections import deque
import gym
USE_CUDA = torch.cuda.is_available()

REL_PROJ_PATH = 'projects/dplay'
FULL_PROJ_PATH = os.path.join(os.environ['HOME'], REL_PROJ_PATH)

def to_tensor_f32(x):
    t_ = torch.from_numpy(np.float32(np.ascontiguousarray(x)))
    if USE_CUDA:
        t_ = t_.cuda()
    return t_

def to_tensor_int(x):
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

def running_val(running_v, v):
    a = 0.99
    return v if running_v is None \
        else running_v * a + v * (1.0 - a)

# FRAMEWORK_PREPROCESSOR: Stack-3-frames
class Preprocessor:
    """
    Raw pixel to numpy array as a "single state observation", which will be
    dealt with by experience memory.
    
    This object will compare two frames, take the difference
    """
    IM_WIDTH = 80
    IM_HEIGHT = 80
    def __init__(self):
        self.previous_frame_2 = None
        self.previous_frame_1 = None
        
    def __call__(self, I):
        return self.process(I)

    # noinspection PyPep8Naming,PyMethodMayBeStatic
    def process(self, I):
        """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
        I = I[35:195]  # crop
        I = I[::2, ::2, 0]  # downsample by factor of 2
        I[I == 144] = 0  # erase background (background type 1)
        I[I == 109] = 0  # erase background (background type 2)
        I[I != 0] = 1  # everything else (paddles, ball) just set to 1
        I = np.float32(I)
        if self.previous_frame_1 is None:
            s = np.stack((I, I, I))
            self.previous_frame_1 = I
            self.previous_frame_2 = I
        else:
            s = np.stack((self.previous_frame_2, self.previous_frame_1, I))
            self.previous_frame_2 = self.previous_frame_1
            self.previous_frame_1 = I
        return s

# FRAMEWORK_ENVIRONMENT: Pong-v0
class Environment:
    def __init__(self):
        self._env = gym.make('Pong-v0')
        self._stopped = False

    def reset(self):
        self._stopped = False
        return self._env.reset()
    
    def step(self, act):
        if not self._stopped:
            s, r, t, info = self._env.step(act)
            if r!=0:
                t = True
            if t:
                self._stopped = True
        else:
            s = r = info = None
        return s, r, self._stopped, info
    
    def render(self):
        return self._env.render()
    

class ExperienceMemory:
    """
    At a time step t, this memory manager expects (args to add_experience):
        S_{t-1}, A_{t-1}, R_t, Is_Terminal_{t}, Agent_Response_To(S_{t-1})
    i.e. the arguments (numbers) are
                t-1  | t
    - STATE:      1  |
    - ACTION:     2  |
    - REWARD:        | 3
    - TERMINAL:      | 4
    - RESP:         =5=>
    and
    - ADVANTAGE_{t}: = Reward_t + Reward_{t+1}*discount + ...
      will be filled when episode ends

    NB   S_t (the current state) will be added in the next step. The last state of an episode is
      NOT recorded (no action will be taken, after all!)
    NB-1 All info is not used in all learning algorithms
    NB-2 RESP will be saved to time step t. This is to align with the supervision information
         that will finally arrive to train the agent. E.g. in Q-learning, the agent would
         try to evaluate all actions given state S_{t-1}, the evaluation will be comapred
         to reward received at time {t}.
    """

    def __init__(self, capacity, discount):
        self.capacity = capacity
        self.discount = discount

        self.experience = {
            'states': deque(),
            'actions': deque(),
            'rewards': deque(),
            'advantages': deque(),
            'term_status': deque(),
            'prev_responses': deque(),
            'episode_id': deque()
        }
        self.first_step_in_episode = True

    def add_experience(self, episode_id, state, action, reward, is_terminal,
                       prev_resp=None, do_compute_advantage=True):
        """
        :param episode_id: do NOT record the episode ID in this object, which can cause inconsistence
          when save/load training sessions.
        :param state: See class doc, the last step state.
        :type state: np.ndarray
        :param action:
        :type action: int
        :param reward:
        :param is_terminal:
        :param prev_resp:
        :param do_compute_advantage: if set, I will compute discounted advantage (see
          class doc) when an episode ends.
        :return:
        """

        # if self.first_step_in_episode:
        #     self.first_step_in_episode = False
        #     self.experience['states'].append(state)
        #     assert new_state is None
        #     assert action is None
        #     assert reward == 0
        #     assert not is_terminal
        #     assert prev_resp is None
        # else:
        #    # eventually, "actions" has one less element than other records

        self.experience['states'].append(state)
        self.experience['actions'].append(action)
        self.experience['rewards'].append(reward)
        self.experience['term_status'].append(is_terminal)
        self.experience['advantages'].append(0)
        self.experience['prev_responses'].append(prev_resp)
        self.experience['episode_id'].append(episode_id)

        N = len(self.experience['states'])
        if N > self.capacity:
            for k_ in self.experience:
                self.experience[k_].popleft()
            N -= 1

        if is_terminal and do_compute_advantage:
            self.experience['advantages'][-1] = future_reward = reward
            this_episode = self.experience['episode_id'][-1]
            for t in range(-2, - N - 1, -1):
                if self.experience['episode_id'][t] != this_episode:
                    break  # have reached the previous episode
                self.experience['advantages'][t] = \
                    self.experience['rewards'][t] + future_reward * self.discount
                future_reward = self.experience['advantages'][t]

    def get_training_batch(self, num_steps=None, episodes=None):
        """
        Specific to the RL algorithm. This implementation is for Policy Gradient using a single episode
        """
        s_ = not (num_steps is None)
        e_ = not (episodes is None)
        assert (s_ and not e_) or (not s_ and e_), \
            "Either the number of steps or the episode id list must be given"

        # implement only using episode
        states = []
        actions = []
        advantages = []

        if s_:
            assert False, "Not implemented"

        if e_:
            for ep in episodes:
                ids = [i_ for i_, ei_ in enumerate(self.experience['episode_id'])
                       if ei_ == ep]

                for t in ids:  # no last action
                    states.append(self.experience['states'][t])
                    actions.append(self.experience['actions'][t])
                    advantages.append(self.experience['advantages'][t])

        return to_tensor_f32(states), to_tensor_int(actions), to_tensor_f32(advantages)
    
    def get_next_training_batch(self):
        """
        A short cut for taking the last one episode
        """
        return self.get_training_batch(
            episodes=[self.experience['episode_id'][-1], ])

    def save(self, fname):
        with open(fname, 'w') as f:
            f.write('Nothing to save')

    def load(self, fname):
        return

# FRAMEWORK_ENCODER: Simple-conv-v0

# opts['input_channels'] = FRAMES_PER_STATE * INPUT_CHANNELS
class DeepConvEncoder(nn.Module):
    def __init__(self, opts):
        super(DeepConvEncoder, self).__init__()
        
        
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
            self.conv_layers.append( nn.Sequential(*lay_) )
            in_kernels = kn
            
        self.feature = nn.Sequential(*self.conv_layers)
        self.num_features = None
        
        if USE_CUDA:
            self.cuda()
        
    def get_feature_num(self, image_size=None):
        """
        :param image_size: info about state variable of images, see Preprocessor and ExperienceMemory
            ['height/width']
        """
        # TODO: Maybe Cuda dummy variable is needed.
        if self.num_features is None:
            assert not (image_size is None), "Image size must be given in the first time"
            dummy_input_t = torch.rand(1, self.input_channels, 
                                       image_size['height'], image_size['width'])
            if USE_CUDA:
                dummy_input_t = dummy_input_t.cuda()
            dummy_input = Variable(dummy_input_t)
            dummy_feature = self.feature(dummy_input)
            nfeat = np.prod(dummy_feature.size()[1:])
            self.num_features = nfeat
        return self.num_features
    
    def forward(self, x):
        return self.feature(x)
    
    def save(self, fname):
        torch.save(self.state_dict(), fname)
        
    def load(self, fname):
        self.load_state_dict(torch.load(fname))

# FRAMEWORK_DECODER: Policy-4-actions
class Decoder(nn.Module):
    def __init__(self, opts):  # TODO use decoder opts
        super(Decoder, self).__init__()
        self.input_num = opts['input_num']
        
        self._fc1 = nn.Linear(in_features=opts['input_num'], 
                              out_features=opts['fc1_hidden_unit_num'])
        self._fc2 = nn.Linear(in_features=opts['fc1_hidden_unit_num'], 
                              out_features=opts['output_num'])
        self._fullconn = nn.Sequential(self._fc1, self._fc2, nn.LogSoftmax())
        # LogSoftmax -- to comply with NLLLoss, which expects the LOG of predicted
        # probability and the target
        if USE_CUDA:
            self.cuda()
    
    def forward(self, x):
        return self._fullconn(x)
    
    def save(self, fname):
        torch.save(self.state_dict(), fname)
        
    def load(self, fname):
        self.load_state_dict(torch.load(fname))

# FRAMEWORK_RLNET: RLNet-v0
class RLNet(nn.Module):
    """
    NB Each component is responsible for itself on cuda or no cuda (some
    needs to play with some data for self inspection -- e.g. convolutional 
    layers only know the dimension of the input at runtime.).

    Cuda status must be consistent among all components.
    """
    def __init__(self, enc, dec):
        """
        :param enc: Feature extractor. See Encoder.
        :type enc: Encoder
        :param dec: Task target predictor
        :type dec: Decoder
        """
        super(RLNet, self).__init__()
        assert enc.get_feature_num() == dec.input_num
        self.enc = enc
        self.dec = dec
        
    def forward(self, x):
        y = self.enc(x)
        y = y.view(-1, self.enc.get_feature_num())
        y = self.dec(y)
        return y

# FRAMEWORK_POLICY: PG-v0
class Policy:
    """
    This policy is to be used with Policy Gradient. The RLNet outputs action probabilities, which 
    stochastically determines the action.
    """
    def __init__(self, rl_net):
        self.rl_net = rl_net
        self.rng = np.random.RandomState(0)
        
    def get_action(self, state):
        """
        :param state: a single frame
        :type state: np.ndarray
        """
        assert state.ndim == 3, \
            "Single state required, multi-state not implemented"
        state_s = state[np.newaxis, ...]
        state_tv_ = to_tensor_f32(state_s)
        action_prob = self.rl_net(Variable(state_tv_, requires_grad=False))
        action_prob = np.exp(to_numpy(action_prob.data)).squeeze()
        return self.rng.choice(action_prob.size, p=action_prob), action_prob
        
        
    def save(self, fname):
        with open(fname, 'w') as f:
            f.write('Nothing to save')
    
    def load(self, fname):
        return
    

class OneStepPolicyGradientTrainer:
    def __init__(self, net, memory, opts):
        """
        :type net:
        """
        self.optimiser = opts['Optimiser'](
            net.parameters(),   # e.g. torch.optim.Adam()
            lr=opts['learning_rate'])
        self.loss_fn = nn.NLLLoss(size_average=False)
        self.net = net
        self.memory = memory
        
    def __call__(self):
        return self.step()
        
    def step(self):
        states, actions, advantages = (
            Variable(t_, requires_grad=False) 
            for t_ in self.memory.get_next_training_batch()
        )
        logP = self.net(states)                       # predicted log-prob of taking different actions
        advantages.data -= advantages.data.mean()     # An operation for tensors not Variables
        advantages = advantages.unsqueeze(1)          # -> sample_num x 1
        advantages = advantages.expand(
            advantages.size(0), 
            logP.size(1))                   # "manual" broadcasting, -> #.samples x #.classes
        logP_adv = advantages * logP
        
        loss = self.loss_fn(logP_adv, actions)        # Integers for 2nd arg, 'target'
        
        # Back-prop
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
        
        loss_value = to_numpy(loss.data)[0]  # we will return a float, not a single-element array
        return loss_value

class Keeper:
    """
    Keeper helps administrate learning:
    ** Methods
    - save              and
    - load              handle checkpoints in the training sessions
    - record_env_step   and
    - record_train_step keeps records of training progress
    - report_step       prints recent progress
    ** Flags
    - need_train        means: according to the learning strategy,
      enough experience has been accquired, a training step is ready
      to run
    - need_save         enough training steps done, to save checkpoint
    - need_draw         to show the game in one episode
    """
    def __init__(self, objs, opts):
        """
        :param objs: objects that can be saved and reloaded in training sessions.
        :param opts: e.g. {
            'train_every_n_episodes': 1,
            'save_every_n_training_steps': 10,  # TODO: set to resonably large
            'draw_every_n_training_steps': -1,
            'save_path': 'SAVE', 
            'report': {
                'save_checkpoint': True,
                'every_n_steps': 1,
                'every_n_training': 1,
                'every_n_episodes': 1}
            }
        
        """
        self.opts = opts
        self.objects = objs
        self.object_filenames = [o_.__class__.__name__ for o_ in self.objects]
        for i in range(len(self.object_filenames)):
            while self.object_filenames[i] in self.object_filenames[:i]:
                self.object_filenames[i] += '-copy'

        self.savepath = opts['save_path']
        self.checkpoint_status_fullfname = os.path.join(self.savepath, 'latest.json')
        self.records = {
            'total_steps': 0,  # step-wise info
            'reward_history': [],
            'running_reward': None,
            'episodes': 0,  # episode-wise info
            'episode_length_history': [],
            'episode_reward_history': [],
            'running_episode_reward': None,
            'training_steps': 0,  # mini-batch-wise info
            'loss_history': [],  # training stuff goes here
            'running_loss': None,
            'checkpoint_history': [],  # checkpoints
            'latest_checkpoint': ''
        }
        self.last_term_t = 0
        self.need_train = False
        self.need_save = False
        self.need_draw = False  # NOTE this flag is reset by renderer
        self.need_stop = False

        self.report_opts = self.opts['report']
        self.last_reported_episode = -1
        self.last_reported_training = -1

    def save(self):
        checkpoint_path = os.path.join(self.savepath,
                                       'checkpoint-{:d}'.format(self.records['training_steps']))
        if not os.path.exists(checkpoint_path):
            os.mkdir(checkpoint_path)
        self.records['checkpoint_history'].append(checkpoint_path)
        
        obj_full_fnames = [os.path.join(checkpoint_path, f_) 
                           for f_ in self.object_filenames]
        for o, of in zip(self.objects, obj_full_fnames):
            o.save(of)
            # print "Save an object {} to {}".format(o.__class__.__name__, of)
        
        self.records['latest_checkpoint'] = checkpoint_path
        with open(self.checkpoint_status_fullfname, 'w') as f:
            json.dump(self.records, f, indent=2)

        self.need_save = False
        if self.report_opts['save_checkpoint']:
            print 'Checkpoint {} saved to {}'.format(
                len(self.records['checkpoint_history']),
                self.records['checkpoint_history']
            )

    def load(self):
        if not os.path.exists(self.checkpoint_status_fullfname):
            return

        with open(self.checkpoint_status_fullfname, 'r') as f:
            self.records = json.load(f)

        obj_full_fnames = [os.path.join(self.records['latest_checkpoint'], f_) 
                           for f_ in self.object_filenames]

        for o, of in zip(self.objects, obj_full_fnames):
            o.load(of)
            # print "Load an object {} from {}".format(o.__class__.__name__, of)

    def record_env_step(self, reward, term):
        r = float(reward)
        rec = self.records
        rec['reward_history'].append(r)
        rec['running_reward'] = running_val(rec['running_reward'], r)
        rec['total_steps'] += 1
        if term:
            # --------
            # NOT save full reward history in JSON. Clear for now
            # ep_r = np.sum(rec['reward_history'][self.last_term_t:])
            ep_r = np.sum(rec['reward_history'])
            rec['reward_history'] = []
            # --------
            rec['episode_reward_history'].append(ep_r)
            rec['running_episode_reward'] = running_val(rec['running_episode_reward'], ep_r)
            rec['episode_length_history'].append(rec['total_steps'] - self.last_term_t)
            self.last_term_t = rec['total_steps']
            rec['episodes'] += 1
            if rec['episodes'] % self.opts['train_every_n_episodes'] == 0:
                self.need_train = True  # reset when recording a training step
            self.need_draw = False

    def record_train_step(self, loss):
        """
        Record training loss. This is separate from recording environment rewards because
        training step happens every few steps.
        """
        loss = float(loss)
        self.need_train = False
        rec = self.records
        rec['loss_history'].append(loss)
        rec['running_loss'] = running_val(rec['running_loss'], loss)
        rec['training_steps'] += 1
        k_ = self.opts['save_every_n_training_steps']
        if k_ > 0 and rec['training_steps'] % k_ == 0:
            self.need_save = True
        k_ = self.opts['draw_every_n_training_steps']
        if k_ > 0 and rec['training_steps'] % k_ == 0:
            self.need_draw = True  # reset when an episode ends
        if rec['training_steps'] >= self.opts['max_training_steps']:
            self.need_stop = True  # TODO: use more elegant conditions

    def report_step(self):
        did_rep = False
        # agent-environment interaction steps -- fastest changing
        n = self.report_opts['every_n_steps']
        N = self.records['total_steps']
        if n > 0 and N > 0 and N % n == 0:
            print "Total step {}, reward {:.3f}, running_reward {:.3f} ".format(
                N,
                self.records['reward_history'][-1],
                self.records['running_reward']),
            did_rep = True

        # every a few episodes -- the reward for episodes is what we are really concerned
        n = self.report_opts['every_n_episodes']
        N = self.records['episodes']
        if n > 0 and N > 0 and N % n == 0 and N != self.last_reported_episode:
            print "Episode {} steps {:3d} reward {:.3f} running episode reward {:.3f} ".format(
                N,
                self.records['episode_length_history'][-1],
                self.records['episode_reward_history'][-1],
                self.records['running_episode_reward']),
            self.last_reported_episode = N
            did_rep = True

        # training
        n = self.report_opts['every_n_training']
        N = self.records['training_steps']
        if n > 0 and N > 0 and N % n == 0 and N != self.last_reported_training:
            print "Training step {}, loss {:.3f}, running_loss {:.3f} ".format(
                N,
                self.records['loss_history'][-1],
                self.records['running_loss']
            ),
            self.last_reported_training = N
            did_rep = True

        if did_rep:
            print

        return

# Framework definition:
# **Necessary to run this cell** to create experiment package for this framework
RL_components = {
    'Preprocessor': Preprocessor,
    'ExperienceMemoryManager': ExperienceMemory,
    'Encoder': DeepConvEncoder,
    'Decoder': Decoder,
    'RLNet': RLNet,
    'Policy': Policy,
    'Environment': Environment,
    'Trainer': OneStepPolicyGradientTrainer,
    'Keeper': Keeper,
}

USE_CUDA = torch.cuda.is_available()
#USE_CUDA = False

experience_opts = {
    'capacity': 1000,
    'discount': 0.99
}

encoder_opts = {
    'input_channels': 3,
    'convs': [
        {'kernel_size':3, 'conv_kernels': 32, 'pool_factor': 2, 'relu': True},
        {'kernel_size':3, 'conv_kernels': 64, 'pool_factor': 2, 'relu': True},
    ]
}

decoder_opts = {
    'input_num': None,
    'fc1_hidden_unit_num': 256,
    'output_num':4
}

trainer_opts = {'Optimiser': torch.optim.Adagrad, 'learning_rate':1e-4}

path_opts = {
    'BASE_PATH': FULL_PROJ_PATH,
    'RUN_PATH': 'RUNS',
    'experiment_id': 'TEST01'}

running_dir = os.path.join(path_opts['BASE_PATH'], 
                           path_opts['RUN_PATH'], 
                           path_opts['experiment_id'])

save_dir = os.path.join(running_dir, 'checkpoints')

if not os.path.exists(running_dir):
    os.mkdir(running_dir)  # NOT using makedirs, I want the 
    # users to be responsible for the parent directory (and 
    # overall structure)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

keeper_opts = {
    'train_every_n_episodes': 1,
    'save_every_n_training_steps': 5000,
    'draw_every_n_training_steps': -1,
    'max_training_steps': 2000000,
    'save_path': save_dir,
    'report': {'save_checkpoint': True,
               'every_n_steps': -1,
               'every_n_training': 1,
               'every_n_episodes': 1}
}
    
# CREATE LEARNING COMPONENTS
env = RL_components['Environment']()
preproc = RL_components['Preprocessor']()
mem = RL_components['ExperienceMemoryManager'](**experience_opts)
enc = RL_components['Encoder'](encoder_opts)
decoder_opts['input_num'] = enc.get_feature_num({'height':preproc.IM_HEIGHT, 'width':preproc.IM_WIDTH})
dec = RL_components['Decoder'](decoder_opts)
rlnet = RL_components['RLNet'](enc, dec)
policy = RL_components['Policy'](rlnet)
trainer = RL_components['Trainer'](rlnet, mem, trainer_opts)
keeper = RL_components['Keeper']([enc, dec, policy, mem], keeper_opts)  # objects has "save/load" interface

# RUNNING: this part does the actual work. 
# NOT necessary to run this cell to create experiment package for this framework
keeper.load()
state = preproc.process(env.reset())


import pdb
while not keeper.need_stop: 
    # pdb.set_trace()
    action, action_prob = policy.get_action(state)
    next_state, reward, is_terminal, _ = env.step(action)
    next_state = preproc(next_state)
    ep = keeper.records['episodes']
    mem.add_experience(ep, state, action, reward, is_terminal, None)
    # None: We don't use last prediction (will predict in traing step)
    
    if is_terminal:
        state = preproc.process(env.reset())
    else:
        state = next_state
        
    keeper.record_env_step(reward, is_terminal)
    
    if keeper.need_train:  # TODO train condition call back
        loss = trainer.step()
        keeper.record_train_step(loss)

    if keeper.need_save:
        keeper.save()

    if keeper.need_draw:
        env.render()
    
    keeper.report_step()

