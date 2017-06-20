import torch.nn as nn
from torch.autograd import Variable
# from dplay_utils.tensordata import to_numpy


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

    # noinspection PyPep8Naming
    def step(self):
        states, actions, advantages = (
            Variable(t_, requires_grad=False)
            for t_ in self.memory.get_next_training_batch()
        )
        advantages.data -= advantages.data.mean()  # An operation for tensors not Variables
        advantages.data /= advantages.data.std()  # Normalise advantage
        # ==== Batch computing loss for all trail steps ====
        # 21/6/2017: We cannot fit time-step-by-time-step weights in the computation.
        # Torch only accepts class-wise weights, needs to reimplement loss function.
        # --------
        # logP = self.net(states)  # predicted log-prob of taking different actions
        # advantages = advantages.unsqueeze(1)  # -> sample_num x 1
        # advantages = advantages.expand(
        #     advantages.size(0),
        #     logP.size(1))  # "manual" broadcasting, -> #.samples x #.classes
        # logP_adv = advantages * logP
        # loss = self.loss_fn(logP_adv, actions)  # Integers for 2nd arg, 'target'
        # loss.backward()
        # loss_value = to_numpy(loss.data)[0]  # we will return a float, not a single-element array
        # ===================================================

        # Back-prop for each trail-experience, i.e. T0, T1, ...
        self.optimiser.zero_grad()
        loss_value = 0.0
        for ti in range(states.size(0)):
            logP_i = self.net(states[ti].unsqueeze(0))
            L_i = self.loss_fn(logP_i, actions[ti]) * advantages[ti]
            L_i.backward()  # this will accumulate gradients in all network parameters
            # ** with applying weight = advantages[ti] **
            loss_value += L_i.data[0]  # L_i is now a singleton tensor

        self.optimiser.step()
        return loss_value
