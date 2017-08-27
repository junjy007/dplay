from collections import namedtuple
from itertools import count
import numpy as np
import json
import torch
from torch import autograd
import gym
import gym.spaces
from dqn_utils.replaybuffer import ReplayBuffer
from dqn_utils.schedule import LinearSchedule
from dqn_utils.mygym import get_wrapper_by_name

USE_CUDA = torch.cuda.is_available()
# I don't think dtype is a good name, it should be tensor-type
ttype = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor


# noinspection PyAbstractClass
class Variable(autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        if USE_CUDA:
            data = data.cuda()
        super(Variable, self).__init__(data, *args, **kwargs)


OptimiserSpec = namedtuple("OptimiserSpec", ["constructor", "kwargs"])


# noinspection PyPep8Naming,PyBroadException
def dqn_learning(
        env,
        q_func,
        optimiser_spec,
        exploration,
        stopping_criterion=None,
        replay_buffer_size=250000,
        batch_size=32,
        gamma=0.99,
        learning_starts=50000,
        learning_freq=4,
        frame_history_len=4,
        target_update_freq=10000,
        save_dir='save'
):
    """
    :param env:
        gym environment to train the net
    :type  env: gym.Env
    :param q_func:
        constructor of q-func approx net
        accept named
    :type q_func: callable()
    :param optimiser_spec:
    :type  optimiser_spec: OptimiserSpec
    :param exploration:
        specifies how the algorithm is randomised
    :type  exploration: LinearSchedule
    :param stopping_criterion:
    :param replay_buffer_size:
    :param batch_size:
    :param gamma:
    :param learning_starts:
    :param learning_freq:
    :param frame_history_len:
    :param target_update_freq:
    :param save_dir:
    :return:
    """
    assert type(env.action_space) == gym.spaces.Discrete
    assert type(env.observation_space) == gym.spaces.Box

    # build model
    img_h, img_w, img_c = env.observation_space.shape
    input_arg = frame_history_len * img_c  # in_channels = #.frame-history * channels per frame
    num_actions = env.action_space.n

    # construct an epsilon-greedy policy
    rng = np.random.RandomState(0)

    def select_epsilon_greedy_action(model, s, t_):
        # noinspection PyArgumentList
        sample = rng.rand()
        eps_threshold = exploration.value(t_)
        if sample > eps_threshold:
            s_ = torch.from_numpy(s).type(ttype).unsqueeze(0) / 255.0
            # unsqueeze(0) => to make the observation a one-sample batch
            predicted_action_values = model(Variable(s_, volatile=True)).data  # type: torch.FloatTensor
            greedy_action = predicted_action_values.max(dim=1)[1].cpu()
            # the 2nd return val of max is the index of the max (argmax) in each row (since
            # we have specified dim=1 in the function call)

            return greedy_action
        else:
            # noinspection PyArgumentList
            return torch.IntTensor([[rng.randint(num_actions)]])

    # init target q function and q function (two models in the Nature 2015 paper)
    Q = q_func(input_arg, num_actions).type(ttype)
    target_Q = q_func(input_arg, num_actions).type(ttype)

    # optimiser
    optimiser = optimiser_spec.constructor(Q.parameters(), **optimiser_spec.kwargs)

    # replay buffer ...
    replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)

    # ENV RUNNING
    try:
        with open("{}/latest.json", 'r') as f:
            train_status = json.load(f)
            Q.load(train_status['latest_checkpoint'])

        num_param_updates = train_status['num_param_updates']
        mean_episode_reward = train_status['mean_episode_reward'][-1]
        best_mean_episode_reward = train_status['best_mean_episode_reward'][-1]
        start_step = train_status['steps']

    except:
        train_status = {
            'num_param_updates': 0,
            'mean_episode_reward': [],
            'best_mean_episode_reward': [],
            'latest_checkpoint': "",
            'steps': 0
        }
        num_param_updates = 0
        mean_episode_reward = -9999
        best_mean_episode_reward = -9999
        start_step = 0

    last_obs = env.reset()
    SAVE_EVERY_N_STEPS = 10000
    target_Q.load_state_dict(Q.state_dict())
    for t in count(start_step):
        if stopping_criterion is not None and stopping_criterion(env):
            break

        last_idx = replay_buffer.store_frame(last_obs)

        recent_observations = replay_buffer.encode_recent_observation()

        if t > learning_starts:
            action = select_epsilon_greedy_action(Q, recent_observations, t)[0, 0]
        else:
            action = rng.randint(num_actions)

        obs, reward, done, _ = env.step(action)
        reward = max(-1.0, min(reward, 1.0))
        replay_buffer.store_effect(last_idx, action, reward, done)
        if done:
            obs = env.reset()

        last_obs = obs

        # experience replay and train the network
        if t > learning_starts and \
                                t % learning_freq == 0 and \
                replay_buffer.can_sample(batch_size):

            obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = \
                replay_buffer.sample(batch_size)
            obs_batch = Variable(torch.from_numpy(obs_batch).type(ttype) / 255.0)
            act_batch = Variable(torch.from_numpy(act_batch).long())
            rew_batch = Variable(torch.from_numpy(rew_batch))
            next_obs_batch = Variable(torch.from_numpy(next_obs_batch).type(ttype) / 255.0)
            not_done_mask = Variable(torch.from_numpy(1 - done_mask)).type(ttype)

            if USE_CUDA:
                act_batch = act_batch.cuda()
                rew_batch = rew_batch.cuda()

            # Compute currently predicted Q, q-func takes the state and outputs the
            # a value for each action
            current_Q_values = Q(obs_batch).gather(1, act_batch.unsqueeze(1))
            # Compute the next max Q, by allowing the q-func to "look forward" one step
            next_max_q = target_Q(next_obs_batch).detach().max(dim=1)[0]  # what does "detach" mean?
            next_Q_values = not_done_mask * next_max_q
            target_Q_values = rew_batch + (gamma * next_Q_values)
            # Loss
            bellman_error = target_Q_values - current_Q_values  # type: Variable
            clipped_bellman_error = bellman_error.clamp(-1, 1)
            d_error = clipped_bellman_error * -1.0   # type: Variable # ? why this is the gradient?
            # Updating the network
            optimiser.zero_grad()
            current_Q_values.backward(d_error.data.unsqueeze(1))  # // what??

            optimiser.step()
            num_param_updates += 1

            # update target Q net from time to time
            if num_param_updates % target_update_freq == 0:
                target_Q.load_state_dict(Q.state_dict())

        # log progress
        episode_rewards = get_wrapper_by_name(env, "Monitor").get_episode_rewards()
        if len(episode_rewards) > 0:
            mean_episode_reward = np.mean(episode_rewards[-100:])
        if len(episode_rewards) > 100:
            best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)

        train_status['num_param_updates'] = num_param_updates
        train_status['mean_episode_reward'].append(mean_episode_reward)
        train_status['best_mean_episode_reward'].append(best_mean_episode_reward)

        if t % SAVE_EVERY_N_STEPS == 0:
            latest_checkpoint = '{}/cp{:d}.torchmodel'.format(save_dir, t)
            Q.save(latest_checkpoint)
            train_status['steps'] = t + 1
            train_status['latest_checkpoint'] = latest_checkpoint
            with open('{}/latest.json'.format(save_dir), 'w') as f:
                json.dump(train_status, f)

            print "Timestep {}".format(t)
            print "- mean reward {:.2f}".format(mean_episode_reward)
            print "- best mean reward {:.2f}".format(best_mean_episode_reward)
            print "- checkpoint saved to {}".format(latest_checkpoint)

    return  # dqn_learning
