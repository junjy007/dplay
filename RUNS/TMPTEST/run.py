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

test_components = {'Preprocessor': GymAtariFramePreprocessor_Stacker}

from experience_managers.preprocessors import GymAtariFramePreprocessor_Stacker

