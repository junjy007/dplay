from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

# ================================
# MANAGE PERFORMANCE INFORMATION
# ================================
EvaluationRecord = namedtuple("EvaluationRecord",
                              ["observations",
                               "hidden_features",
                               "predicted_action_values",
                               "actions"])


class FeatureCollector:
    """
    Callable object, when used as hook, it will collect the forward data
    of a pytorch OP
    """

    def __init__(self):
        self.data = []

    def __call__(self, m, data_in, data_out):
        # print type(m), len(data_in), type(data_in[0]), type(data_out)
        # print data_in[0].data.size()
        self.data.append(data_in[0].data.clone().cpu().numpy())
        return

    def reset(self):
        self.data = []


# ================================
# VISUALISATION FUNCTIONS
# ================================
def observation_history_to_image(s, decay=0.5):
    """
    :param s:
    :param decay:
    :return:

    NOTE assume #.colour channels = 1.
    """
    history_len = s.shape[0]

    sim = s[0].astype(np.float)
    coeff = decay
    coeff_sum = 1.0
    for i in range(1, history_len):
        sim += s[i].astype(np.float) * coeff
        coeff_sum += coeff
        coeff *= decay
    sim /= coeff_sum
    sim = sim.astype(np.uint8)
    return sim


def draw_state_evaluation(
        record, feat2d, save_fname_prefix,
        action_labels=None):
    margin = 0.02

    feat2d_xlim = feat2d_ylim = []
    if feat2d is not None:
        feat2d_xlim = [feat2d[:, 0].min(), feat2d[:, 0].max()]
        feat2d_ylim = [feat2d[:, 1].min(), feat2d[:, 1].max()]

    if action_labels is not None:
        action_labels = np.asarray(action_labels)
        unique_actions, a_uind = np.unique(action_labels, return_inverse=True)
        # if action is a_, then unique_actions[a_uind[a_]] is the action

    num_actions = record.predicted_action_values[0].size
    n_steps = len(record.observations)
    for t in range(n_steps):
        # prepare canvas
        plt.figure(1, figsize=(16, 10))
        gs = gridspec.GridSpec(nrows=4, ncols=6,
                               left=margin, right=1 - margin,
                               bottom=margin, top=1 - margin,
                               wspace=0.01, hspace=0.01)
        ax0 = plt.subplot(gs[:3, :3])
        ax1 = plt.subplot(gs[-1, :3])
        ax2 = plt.subplot(gs[:3, 3:])

        # draw the observation
        im = observation_history_to_image(record.observations[t])
        ax0.cla()
        ax0.imshow(im, interpolation='nearest', cmap='gray')
        ax0.set_title("T={:d}".format(t))
        ax0.set_xticks([])
        ax0.set_yticks([])

        # draw action value bars
        av_ = record.predicted_action_values[t][0]
        a_ = record.actions[t]
        action_value_lim = [av_.min() - 0.01, av_.max() + 0.01]

        ax1.cla()

        if action_labels is not None:
            ax1.set_xticklabels(unique_actions)
            sum_av_ = np.array([np.sum(av_[action_labels == ua_])
                                for ua_ in unique_actions])
            ax1.bar(np.arange(unique_actions.size), sum_av_)
            ax1.set_xticks(np.arange(num_actions))
            ax1.set_xticklabels(unique_actions)
            ax1.bar(a_uind[a_], sum_av_[a_uind[a_]], fc='r')
        else:
            ax1.bar(np.arange(num_actions), av_)
            ax1.set_xticks(np.arange(num_actions))
            ax1.bar(a_, av_[a_], fc='r')
            ax1.set_ylim(action_value_lim)
            # ax1.set_yticks(np.arange(-1, 1.01, .5))
            # ax1.grid('on')

        # draw feature 2d of the current observation's
        if feat2d is not None:
            ax2.cla()
            plt.plot(feat2d[:, 0], feat2d[:, 1], 'b+')
            plt.scatter(feat2d[t, 0], feat2d[t, 1], c='r', s=64)
            ax2.set_xlim(feat2d_xlim)
            ax2.set_ylim(feat2d_ylim)

        fname = save_fname_prefix + '_{:05d}.png'.format(t)
        plt.savefig(fname)
        print "\rT={:d}".format(t),


# ================================
# MAIN EVALUATION PROCEDURE
# ================================

# Dealing with using multiple models