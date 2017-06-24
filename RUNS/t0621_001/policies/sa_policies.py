"""
sa_policies: state-to-action policies, (as opposed to state-value policies)
"""

from torch.autograd import Variable
import numpy as np
from dplay_utils.tensordata import to_tensor_f32, to_numpy


# FRAMEWORK_POLICY: PG-v0
class Policy:
    """
    This policy is to be used with Policy Gradient. The RLNet outputs action probabilities, which
    stochastically determines the action.
    """

    def __init__(self, rl_net):
        """
        :param rl_net:
        """
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

    @staticmethod
    def save(fname):
        with open(fname, 'w') as f:
            f.write('Nothing to save')

    @staticmethod
    def load(fname):
        return
