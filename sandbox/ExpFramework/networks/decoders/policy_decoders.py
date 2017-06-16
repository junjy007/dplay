import torch
import torch.nn as nn
from dplay_utils.tensordata import does_use_cuda


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
        if does_use_cuda():
            self.cuda()

    def forward(self, x):
        return self._fullconn(x)

    def save(self, fname):
        torch.save(self.state_dict(), fname)

    def load(self, fname):
        self.load_state_dict(torch.load(fname))
