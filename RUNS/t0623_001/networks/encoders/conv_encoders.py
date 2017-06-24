import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from dplay_utils.tensordata import does_use_cuda


class DummyEncoder(nn.Module):
    def __init__(self, opts):
        super(DummyEncoder, self).__init__()
        self.input_channels = opts['input_channels']
        self.num_features = None
        if does_use_cuda():
            self.cuda()

    def forward(self, x):
        return x

    def get_feature_num(self, image_size=None):
        """
        :param image_size: info about state variable of images, see Preprocessor and ExperienceMemory
            ['height/width']
        """
        if self.num_features is None:
            assert not (image_size is None), "Image size must be given in the first time"
            self.num_features = \
                image_size['height'] * image_size['width'] * self.input_channels
        return self.num_features

    def save(self, fname):
        with open(fname, 'w') as f:
            f.write("Nothing to save")

    def load(self, fname):
        with open(fname, 'r') as f:
            pass

# noinspection PyListCreation
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
            self.conv_layers.append(nn.Sequential(*lay_))
            in_kernels = kn

        self.feature = nn.Sequential(*self.conv_layers)
        self.num_features = None

        if does_use_cuda():
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
            if does_use_cuda():
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
