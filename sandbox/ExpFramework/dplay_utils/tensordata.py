"""
Manage to-and-pro between torch and numpy tensor
"""
import torch
import numpy as np


USE_CUDA = torch.cuda.is_available()


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


def does_use_cuda():
    return USE_CUDA
