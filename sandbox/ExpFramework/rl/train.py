import torch.nn as nn
from torch.autograd import Variable
from dplay_utils.tensordata import to_numpy


class OneStepPolicyGradientTrainer:
    def __init__(self, net, memory, opts):
        """
        :type net:
        """
        self.optimiser = opts['Optimiser'](
            net.parameters(),  # e.g. torch.optim.Adam()
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
        logP = self.net(states)  # predicted log-prob of taking different actions
        advantages.data -= advantages.data.mean()  # An operation for tensors not Variables
        advantages = advantages.unsqueeze(1)  # -> sample_num x 1
        advantages = advantages.expand(
            advantages.size(0),
            logP.size(1))  # "manual" broadcasting, -> #.samples x #.classes
        logP_adv = advantages * logP

        loss = self.loss_fn(logP_adv, actions)  # Integers for 2nd arg, 'target'

        # Back-prop
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        loss_value = to_numpy(loss.data)[0]  # we will return a float, not a single-element array
        return loss_value
