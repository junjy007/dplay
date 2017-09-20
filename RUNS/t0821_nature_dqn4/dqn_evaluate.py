import numpy as np
import os
import gym
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.manifold import TSNE
from dqn_model import DQN
from dqn_learn import OptimiserSpec, dqn_learning
from dqn_utils.mygym import get_env
from dqn_utils.atari_wrapper import wrap_deepmind
from dqn_utils.replaybuffer import ReplayBuffer
from dqn_utils.evaluation import *
from collections import namedtuple
USE_CUDA = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor

def dqn_evaluate(
        env,
        q_func,
        trained_model_fname,
        replay_buffer_size=250000,
        frame_history_len=4,
        hidden_feature=None,
        max_eval_steps=np.inf,
        max_eval_episodes=np.inf,
        save_dir='save'):
    """
    :param env:
    :type env: gym.Wrapper
    :param q_func: constructor of a q-estimator
    :param trained_model_fname: checkpoint file
    :param replay_buffer_size:
    :param frame_history_len:
    :param hidden_feature: which hidden layer to check
    :param save_dir:
    :return:

    TODO: now collect single feature per forward pass, should be able to
    collect multiple
    """
    # - init model parameters (derived)
    img_h, img_w, img_c = env.observation_space.shape
    input_arg = frame_history_len * img_c  # in_channels = #.frame-history * channels per frame
    num_actions = env.action_space.n
    replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)

    # - greedy policy
    def select_greedy_action(model, s):
        s_ = torch.from_numpy(s).type(FloatTensor).unsqueeze(0) / 255.0
        # unsqueeze(0) => to make the observation a one-sample batch
        predicted_action_values = model(Variable(s_, volatile=True)).data  # type: torch.FloatTensor
        greedy_action = predicted_action_values.max(dim=1)[1].cpu()
        # the 2nd return val of max is the index of the max (argmax) in each row (since
        # we have specified dim=1 in the function call)
        return greedy_action, predicted_action_values

    Q = q_func(input_arg, num_actions).type(FloatTensor)
    try:
        print "Loading from {}".format(trained_model_fname)
        Q.load(trained_model_fname)
    except:
        print "Failed to load from checkpoint {}".format(trained_model_fname)
        return

    collector_hook = FeatureCollector()
    if hidden_feature is not None:
        tmp_mod_list = [m for m in Q.modules()]
        hl = tmp_mod_list[hidden_feature]  # get the last forward layer
        hl.register_forward_hook(collector_hook)

    # Run the game and collect all we needed.
    last_obs = env.reset()
    rec = EvaluationRecord(
        observations=[],
        hidden_features=collector_hook.data,
        predicted_action_values=[],
        actions=[]
    )

    rr = 0
    epi_rr = 0
    episodes = 0
    for t in range(max_eval_steps):
        replay_buffer.store_frame(last_obs)
        recent_observations = replay_buffer.encode_recent_observation()
        action, action_values = select_greedy_action(Q, recent_observations)
        action = action[0, 0]
        action_values = action_values.cpu().numpy()
        rec.observations.append(recent_observations)
        rec.predicted_action_values.append(action_values)
        rec.actions.append(action)
        obs, reward, done, _ = env.step(action)
        rr += reward
        epi_rr += reward
        last_obs = obs

        print "\r t {}, Ep {}: {:.2f}(epi)  {:.2f}(tot)".format(t, episodes, epi_rr, rr),
        if done:
            episodes += 1
            epi_rr = 0

    return rec
