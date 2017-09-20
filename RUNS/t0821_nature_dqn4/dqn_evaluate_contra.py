from collections import namedtuple
from itertools import count
import numpy as np
import os
import torch
from torch import autograd
from dqn_utils.replaybuffer import ReplayBuffer
from dqn_utils.evaluation import EvaluationRecord
from dqn_utils.env_wrapper_NES import get_contra_env
from dqn_model import DQN

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


# noinspection PyPep8Naming
class MultiDQNPolicy:
    def __init__(self, Q_func_constructor,
                 env,
                 saved_models,
                 model_start_states):

        self._Q_func_constructor = Q_func_constructor
        self._saved_models = saved_models
        self._scene_images = []
        for nss in model_start_states:
            env.load_state(nss)
            env.reset()
            env.step(0)
            self._scene_images.append(env.frame().copy())

        self._scene_threshold = [22.5, 28.5, 22.5, 35.5]
        self._scene_min_steps = [0, 275, 250, 300]
        self._current_model_id = 0
        self._model = None

    def get_current_model(self):
        if self._model is None:
            fname = self._saved_models[self._current_model_id]
            model = self._Q_func_constructor()
            model.load(fname)
            self._model = model
        else:
            model = self._model
        return model

    def set_current_model(self, i):
        if i == self._current_model_id:
            return
        else:
            self._current_model_id = i
            self._model = None

    def change_model_when_ready(self, env, t):
        """
        :param env: a game environment, now we consider only the image
        :param t: the current steps -- to prevent early change
        :return:
         - True if next stage is entered, False otherwise
         - The numeric difference between current scene and the starting scene of
           next model
        """
        i = self._current_model_id
        if i == len(self._scene_images) - 1:  # final stage already
            # TODO
            # determine if boss defeated
            return False, 0

        i += 1
        current_fr = env.frame()
        ref_fr = self._scene_images[i]
        image_diff = np.abs((current_fr - ref_fr).astype(np.float)).mean()

        if image_diff < self._scene_threshold[i] and t > self._scene_min_steps[i]:
            self.set_current_model(i)
            return True, image_diff
        return False, image_diff


# ================
# Policies

def greedy_priority_explore(avals, rng, epsilon, agamma=0.5):
    """
    With high probability, say, 0.95, choose the best action, with a small chance
    choose action w.r.t. their values -- 2nd 50%, 3rd 25%, ...
    :return: chosen action
    """
    if rng.rand() > epsilon:
        return np.argmax(avals)
    else:
        p = np.power(agamma, range(avals.size - 1))
        p /= p.sum()
        p = np.cumsum(p)
        a = np.argsort(avals)[0][::-1][1:]
        for a_, p_ in zip(a, p):  # from the second largest
            if rng.rand() < p_:
                return a_
        return a[-1]


def select_greedy_action(model, s, rng, epsilon=0.05, agamma=0.5):
    s_ = torch.from_numpy(s).type(ttype).unsqueeze(0) / 255.0
    # unsqueeze(0) => to make the observation a one-sample batch
    predicted_action_values = model(Variable(s_, volatile=True)).data  # type: torch.FloatTensor
    avals = predicted_action_values.cpu().numpy()
    a = greedy_priority_explore(avals, rng, epsilon=epsilon, agamma=agamma)

    # greedy_action = predicted_action_values.max(dim=1)[1].cpu()
    # # the 2nd return val of max is the index of the max (argmax) in each row (since
    # # we have specified dim=1 in the function call)
    return a, avals


# ================
# Evaluation and tests

# noinspection PyUnusedLocal,PyPep8Naming
def test_repeat_model_eval(nssfile, modelfile, rng, expected=None):
    """
    Now we know that, even with exactly same inputs (frame histories), the
    Perform exactly same experiment, check the model output of predicted action values

    If the values change from one run to the next, find out what has changed.
    :param nssfile:
    :param modelfile:
    :param rng: random number generator
    :param expected: expected quantities, now we check the observation encoded by ReplayMemory
    :type expected: dict
    :return:
    """

    obs_x = None
    if type(expected) == dict:
        obs_x = expected.get('obs', None)  # expected recent observations

    env = get_contra_env(nssfile)

    replay_buffer_size = 250000
    frame_history_len = 4

    img_h, img_w, img_c = env.observation_space.shape
    input_arg = frame_history_len * img_c  # in_channels = #.frame-history * channels per frame
    num_actions = env.action_space.n

    Q = DQN(input_arg, num_actions, img_h, img_w).type(ttype)
    Q.load(modelfile)
    replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)

    last_obs = env.reset()
    actval_rec = []
    act_rec = []
    obs_rec = []
    frm_rec = []
    for t in count():
        buf_idx = replay_buffer.store_frame(last_obs)
        recent_observations = replay_buffer.encode_recent_observation()
        action, action_values = select_greedy_action(Q, recent_observations, rng)

        act_rec.append(action)
        actval_rec.append(action_values)
        obs_rec.append(recent_observations)
        frm_rec.append(last_obs)
        if obs_x is not None:
            # noinspection PyTypeChecker
            if not np.all(recent_observations == obs_x[t]):
                print "Diff happens at {}".format(t)
                break

        obs, reward, done, _ = env.step(action)
        replay_buffer.store_effect(buf_idx, action, reward, done)

        last_obs = obs
        if done:
            break

    return obs_rec, act_rec, actval_rec, frm_rec, replay_buffer


# noinspection PyPep8Naming
def evaluate_single_policy(nssfile, modelfile, rng,
                           frame_history_len=4,
                           epsilon=0.05, agamma=0.5, playdir=None):
    """
    Apply a single policy, returns the performance records
    :param nssfile:
    :param modelfile:
    :param rng: random number generator
    :param frame_history_len:
    :param playdir:
    :return:
    """
    replay_buffer_size = 10000
    env = get_contra_env(nssfile)

    img_h, img_w, img_c = env.observation_space.shape
    input_arg = frame_history_len * img_c  # in_channels = #.frame-history * channels per frame
    num_actions = env.action_space.n

    Q = DQN(input_arg, num_actions, img_h, img_w).type(ttype)
    Q.load(modelfile)

    rec = EvaluationRecord(
        observations=[],
        hidden_features=None,  # collector_hook.data,
        predicted_action_values=[],
        actions=[])

    replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)
    last_obs = env.reset()
    for t in count():
        buf_idx = replay_buffer.store_frame(last_obs)
        recent_observations = replay_buffer.encode_recent_observation()
        action, action_values = select_greedy_action(Q, recent_observations, rng, epsilon, agamma)
        rec.observations.append(recent_observations)
        rec.predicted_action_values.append(action_values)
        rec.actions.append(action)
        obs, reward, done, _ = env.step(action)
        replay_buffer.store_effect(buf_idx, action, reward, done)
        last_obs = obs
        if done:
            break

        if playdir is not None and t % 100 == 0:
            # save state for new play
            save_state_file = os.path.join(playdir, "step-{:04d}.nss".format(t))
            save_state_file = os.path.abspath(save_state_file)
            env.save_state(save_state_file)
    return rec

# Play "through" a stage.
def evaluate_multimodel_by_start_image(play_dir, models, start_mini_stage=0):
    """
    There is a "stage.nss" in play dir

    :param play_dir:
    :param models: saved model and the starting scene from which those models are trained
    :param start_mini_stage:
    :return:
    """
    play_dir = os.path.abspath(play_dir)
    assert os.path.exists(play_dir), "Play dir does not exist"
    def full_fn(fn):
        return os.path.join(play_dir, fn)

    # setup memory
    buffer_size = 50000
    frame_history_len = 4
    replay_buffer = ReplayBuffer(buffer_size, frame_history_len)

    # setup models
    env = get_contra_env()
    img_h, img_w, img_c = env.observation_space.shape
    input_arg = frame_history_len * img_c  # in_channels = #.frame-history * channels per frame
    num_actions = env.action_space.n
    def Contra_DQN():
        return DQN(input_arg, num_actions, img_h, img_w).type(ttype)

    mp = MultiDQNPolicy(Contra_DQN, env,
                        models['trained_models'],
                        models['training_start_stages'])

    # where to start playing
    mini_stage = start_mini_stage
    im_diff_to_next = []


    randseeds = [0, 14, 0, 0]
    action_seq = [
        np.zeros(10000, dtype=np.int),
        np.zeros(10000, dtype=np.int),
        np.zeros(10000, dtype=np.int),
        np.zeros(10000, dtype=np.int)]

    while mini_stage < 4:
        next_stage_state_file = full_fn('s{}.nss'.format(mini_stage))
        print "Now do {}".format(next_stage_state_file)
        env.load_state(next_stage_state_file)

        # repeatedly try until we get through
        mini_stage_done = False
        rs = randseeds[mini_stage]
        while not mini_stage_done:
            print "Mini stage {}, rs {}".format(mini_stage, rs)
            rng = np.random.RandomState(rs)
            rs += 1
            last_obs = env.reset()
            for t in count():
                buf_idx = replay_buffer.store_frame(last_obs)
                recent_observations = replay_buffer.encode_recent_observation()

                did_change_model, imd = \
                    mp.change_model_when_ready(env, t)
                im_diff_to_next.append(imd)
                Q = mp.get_current_model()

                if did_change_model:
                    print "ms {}({}), t {}, id {:.2f}".format(mini_stage, rs-1, t, imd)
                    # save
                    mini_stage_done = True
                    print "Mini stage {} cleared".format(mini_stage)
                    mini_stage += 1
                    next_stage_state_file = full_fn('s{}.nss'.format(mini_stage))
                    env.save_state(next_stage_state_file)
                    break

                action, action_values = select_greedy_action(Q, recent_observations, rng,
                                                             **models['eval_settings'][mini_stage])

                action_seq[mini_stage][t] = action
                action_seq[mini_stage][t+1] = 999

                # rec.observations.append(recent_observations)
                # rec.predicted_action_values.append(action_values)
                # rec.actions.append(action)
                obs, reward, done, _ = env.step(action)
                replay_buffer.store_effect(buf_idx, action, reward, done)
                # rr += reward
                last_obs = obs
                if done:
                    if mini_stage == 3 and reward>=0:
                        mini_stage_done = True
                        mini_stage += 1
                    break
    return action_seq

def show_game_play(nssfiles, actseq):
    i = 0
    env = get_contra_env()
    for nss, aa in zip(nssfiles, actseq):
        env.load_state(nss)
        env.reset()
        env.begin_movie('tmp/play1/{}.fm2'.format(i))
        for a_ in aa:
            if a_>=999:
                break
            env.step(a_)
        env.end_movie()
        i+=1




