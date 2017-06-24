import os
import torch
import torch.nn as nn
import json
from torch.autograd import Variable
import numpy as np
import imp  # Python 2
from collections import deque
import gym
import time
import subprocess
from dplay_utils.tensordata import to_tensor_f32
p = subprocess.Popen('hostname', shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
hostname = p.stdout.readlines()[0][:-1]
global USE_CUDA
USE_CUDA = torch.cuda.is_available()
if hostname == 'maibu':
    REL_PROJ_PATH = 'local/projects/dplay'
else:   # configure project folder on different machines
    REL_PROJ_PATH = 'projects/dplay'
    
FULL_PROJ_PATH = os.path.join(os.environ['HOME'], REL_PROJ_PATH)

from experience_managers.preprocessors import GymAtariFramePreprocessor_Stacker, GymAtariFramePreprocessor_Diff

from games.aigym import AtariEnvironment_Pong

from experience_managers.mem import ExperienceMemory

from networks.encoders.conv_encoders import DeepConvEncoder, DummyEncoder

from networks.decoders.policy_decoders import Decoder

from networks.nets import RLNet

from policies.sa_policies import Policy

from rl.train import OneStepPolicyGradientTrainer

from rl.keeppg import Keeper

# Framework definition:
# **Necessary to run this cell** to create experiment package for this framework
from rl.keeppg import RLAlgorithm
RL_components = {
    'Preprocessor': GymAtariFramePreprocessor_Diff,
    'ExperienceMemoryManager': ExperienceMemory,
    'Encoder': DummyEncoder,
    'Decoder': Decoder,
    'RLNet': RLNet,
    'Policy': Policy,
    'Environment': AtariEnvironment_Pong,
    'Trainer': OneStepPolicyGradientTrainer,
    'Keeper': Keeper,
}

experience_opts = {
    'capacity': 1000,
    'discount': 0.99
}

encoder_opts = {
    'input_channels': 1,
}

decoder_opts = {
    'input_num': None,
    'fc1_hidden_unit_num': 256,
    'output_num':4
}

trainer_opts = {'Optimiser': torch.optim.Adagrad, 'learning_rate':1e-6}

path_opts = {
    'BASE_PATH': FULL_PROJ_PATH,
    'RUN_PATH': 'RUNS',
    'experiment_id': 'DiffPreproc1'}

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
               'every_n_steps': 10000,
               'every_n_training': 1,
               'every_n_episodes': 1,
               'every_n_time_records': 100}
}
    
# CREATE LEARNING COMPONENTS
env = RL_components['Environment']()
preproc = RL_components['Preprocessor']()
mem = RL_components['ExperienceMemoryManager'](**experience_opts)
enc = RL_components['Encoder'](encoder_opts)
decoder_opts['input_num'] = enc.get_feature_num({'height':preproc.im_height, 'width':preproc.im_width})
dec = RL_components['Decoder'](decoder_opts)
rlnet = RL_components['RLNet'](enc, dec)
policy = RL_components['Policy'](rlnet)
trainer = RL_components['Trainer'](rlnet, mem, trainer_opts)
keeper = RL_components['Keeper']([enc, dec, policy, mem], keeper_opts)  # objects has "save/load" interface
alg = RLAlgorithm(keeper, env, preproc, mem, policy, trainer)

alg.run()

