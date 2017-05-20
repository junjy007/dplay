import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import gym
import os
import json

USE_CUDA = torch.cuda.is_available()

FRAMES_PER_STATE = 4
DISCOUNT = 0.99


class Preprocessor:
    def __init__(self):
        self.output_info = {
            'image_width': 80,
            'image_height': 80,
        }
        pass

    # noinspection PyPep8Naming,PyMethodMayBeStatic
    def process(self, I):
        """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
        I = I[35:195]  # crop
        I = I[::2, ::2, 0]  # downsample by factor of 2
        I[I == 144] = 0  # erase background (background type 1)
        I[I == 109] = 0  # erase background (background type 2)
        I[I != 0] = 1  # everything else (paddles, ball) just set to 1
        return np.ascontiguousarray(I.astype(np.float32))


# TODO [move to a json file]
INPUT_CHANNELS = 1
KSIZE1 = 3
KNUM1 = 32
KSIZE2 = 3
KNUM2 = 64
HIDDEN_UNITS_NUM1 = 256
ACTION_NUM = 4


class PolicyNet(nn.Module):
    def __init__(self, im_width, im_height):
        super(PolicyNet, self).__init__()
        state_channels = FRAMES_PER_STATE * INPUT_CHANNELS
        self._conv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=state_channels, out_channels=KNUM1, kernel_size=KSIZE1, padding=(KSIZE1 - 1) / 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self._conv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=KNUM1, out_channels=KNUM2, kernel_size=KSIZE2, padding=(KSIZE2 - 1) / 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self._feature = nn.Sequential(self._conv_layer1, self._conv_layer2)
        dummy_input = Variable(torch.rand(1, state_channels, im_height, im_width))
        dummy_feature = self._feature(dummy_input)
        nfeat = np.prod(dummy_feature.size()[1:])

        self._fc1 = nn.Linear(in_features=nfeat, out_features=HIDDEN_UNITS_NUM1)
        self._fc2 = nn.Linear(in_features=HIDDEN_UNITS_NUM1, out_features=ACTION_NUM)
        self._fullconn = nn.Sequential(self._fc1, self._fc2, nn.LogSoftmax())
        # LogSoftmax -- to comply with NLLLoss, which expects the LOG of predicted
        # probability and the target

        self._num_features = nfeat

        if USE_CUDA:
            with torch.cuda.device(0):
                self.cuda()

    def forward(self, x):
        y = self._feature(x)
        y = y.view(-1, self._num_features)
        y = self._fullconn(y)
        return y


def get_state(frame, last_state=None, preprocessor=Preprocessor()):
    """
    :param frame: raw pixel matrix observation
    :param preprocessor:
    :param last_state:
    :type last_state: np.ndarray or None
    :return:
        1 torch variable usable for forward operation,
        2 corresponding np array tracking the last state
    """
    processed_frame = preprocessor.process(frame)
    assert processed_frame.ndim == 2, "Preprocessed frame must be grayscale"
    if last_state is None:
        state = np.tile(processed_frame[np.newaxis, :, :], (FRAMES_PER_STATE, 1, 1))
    else:
        state = np.concatenate((last_state[1:, :, :], processed_frame[np.newaxis, ...]),
                               axis=0)
    return state


def get_state_var(state):
    """
    :param state: pre-processed frame(s)
    :type state: np.ndarray
    :return:
    """
    state_channels = FRAMES_PER_STATE * INPUT_CHANNELS
    assert state.ndim == 4 and state.shape[1] == FRAMES_PER_STATE * INPUT_CHANNELS, \
        "State must be minibatch images of {}(frames) x {}(channels) " \
        "= {} channels, now get {}".format(
        FRAMES_PER_STATE, INPUT_CHANNELS, state_channels, state.shape)
    state_variable = Variable(torch.from_numpy(state), requires_grad=False)
    if USE_CUDA:
        with torch.cuda.device(0):
            state_variable = state_variable.cuda(0)
    return state_variable


def save_experience(experience, state, action_prob, action_taken, reward, advantage=None):
    """
    :param experience: in/out, see [train_step]
    :param state:
    :param action_prob:
    :param action_taken: actual action taken NOTE: input is int, but saved as one-hot
    :type action_taken: int
    :param reward: immediate reward
    :param advantage: eventual reward, counting for discounted future rewards

    :return: updated experience history with the latest experience appended.

    NOTE if the advantage is provided, a mile-stone reward is known, all previously
     undetermined advantage will be populated.
    """
    experience['states'].append(state)
    experience['action_probs'].append(action_prob)
    experience['actions'].append(action_taken)
    experience['rewards'].append(reward)
    experience['advantages'].append(advantage)

    if advantage is None:
        return

    adv = experience['advantages']
    T = len(adv)
    future_reward = adv[T - 1]
    for t in range(T - 2, -1, -1):
        if not (adv[t] is None):
            break  # Only compute for advantages that are NOT previously undermined
        future_reward *= DISCOUNT
        adv[t] = experience['rewards'][t] + future_reward
        future_reward = adv[t]
    return


def get_torch_experience(experience):
    """
    :param experience: see [train_step]
    :type experience: dict
    :return: another dictionary, each entry of the experience is wrapped in a
     continginuous numpy array.
    """
    def stack_to_tensor(x):
        return torch.from_numpy(np.stack(x).astype(np.float32))

    # noinspection PyArgumentList
    exp = {
        'states': stack_to_tensor(experience['states']),
        'advantages': stack_to_tensor(experience['advantages']),
        'actions': torch.LongTensor(experience['actions']),
    }

    for k in exp.keys():
        if USE_CUDA:
            with torch.cuda.device(0):
                exp[k] = exp[k].cuda()

    return exp


def train_step(net, optimiser, experience):
    """
    :param net:
    :type net: nn.Module
    :param optimiser:
    :param experience: history of {
      - state,
      - action-probabilities,
      - action,
      - instance reward,
      - advantage,
      - advantage-ready: if advantages are computed}
    :type experience: dict
    :return:
    """

    """
    extract states from experience as a variable
    compute probabilities of those states (saved probabilities unused)
    loss: (probabilities vs. actual action taken) * action advantages
    compute grad w.r.t. loss
    update model
    """
    exp = get_torch_experience(experience)
    predicted_action_prob = net.forward(Variable(exp['states'], requires_grad=False))
    action_taken = Variable(exp['actions'], requires_grad=False)
    advantages = exp['advantages']
    advantages = advantages - advantages.mean()
    advantages = Variable(advantages, requires_grad=False).unsqueeze(1)  # -> sample_num x 1
    advantages = advantages.expand(advantages.size(0), predicted_action_prob.size(1))  # manual broadcasting
    # this is more efficient than "repeat" see
    # https://github.com/pytorch/pytorch/issues/491
    modulated_pred_action_prob = predicted_action_prob * advantages
    loss_fn = nn.NLLLoss(size_average=False)  # likelihood loss w.r.t. advantages
    loss = loss_fn(modulated_pred_action_prob, action_taken)  # type: Variable

    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

    loss_value = loss.data
    if USE_CUDA:
        loss_value = loss_value.cpu()
    return loss_value[0]


# TODO [move to hyper-param-def section, with a number much larger]
MAX_EPISODES = 100000
SAVE_EVERY_N_STEPS = 1000
LEARNING_RATE = 1e-5
SAVE_PATH = 'RUNS_PolicyGrad'
if not os.path.exists(SAVE_PATH):
    os.mkdir(SAVE_PATH)
CHECKPOINT_PATTERN = 'checkpoint-{}'


# noinspection PyShadowingNames
def learn_policy(policy_net, env, frame_preprocessor):
    """
    :param policy_net: a network evaluating action probability
    :type policy_net: nn.Module
    :param env: an OpenAI game, like an Atari environment
    :type env: gym.core.Env
    :param frame_preprocessor: preprocess environment frame into a grayscale array
    :return:
    """
    checkpoint_status_file = os.path.join(SAVE_PATH, 'checkpoints.json')
    if os.path.exists(checkpoint_status_file):
        with open(checkpoint_status_file, 'r') as f:
            history = json.load(f)
            episode = history['episode']
            running_reward = history['running_reward']
            running_loss = history['running_loss']
            loss_history = history['loss_history']
            reward_history = history['reward_history']

        saved_checkpoint_filename = os.path.join(SAVE_PATH, CHECKPOINT_PATTERN.format(episode))
        policy_net.load_state_dict(torch.load(saved_checkpoint_filename))
        print "Load model from {}".format(saved_checkpoint_filename)
    else:
        episode = 0
        running_loss = None
        running_reward = None
        loss_history = []
        reward_history = []

    state = get_state(env.reset(), None, preprocessor=frame_preprocessor)  # type: Variable
    rng = np.random.RandomState(0)
    optimiser = torch.optim.Adagrad(policy_net.parameters(), lr=LEARNING_RATE)
    experience = {k_: [] for k_ in ['states', 'action_probs', 'actions', 'rewards', 'advantages']}
    while episode < MAX_EPISODES:
        if episode % 20 == 0:
            env.render()

        action_prob_v = policy_net(get_state_var(state[np.newaxis, ...]))  # type: Variable
        action_prob = action_prob_v.data
        if USE_CUDA:
            action_prob = action_prob.cpu()
        action_prob = action_prob.numpy().squeeze()  # type: np.ndarray
        action = rng.choice(ACTION_NUM, p=np.exp(action_prob))
        new_frame, reward, _, _2 = env.step(action)
        advantage = reward if reward != 0 else None
        save_experience(experience, state, action_prob, action, reward, advantage)

        if reward == 0:
            state = get_state(new_frame, state, preprocessor=frame_preprocessor)
        else:
            episode += 1
            state = get_state(env.reset(), None, preprocessor=frame_preprocessor)

            # train one step
            loss = train_step(policy_net, optimiser, experience)
            experience = {k_: [] for k_ in ['states', 'action_probs', 'actions', 'rewards', 'advantages']}

            running_reward = reward if running_reward is None else running_reward * 0.99 + reward * 0.01
            running_loss = loss if running_loss is None else running_loss * 0.99 + loss * 0.01
            reward_history.append(reward)
            loss_history.append(loss)

            print "Ep {} Reward {} / {} (running) Loss {:6.3f} / {} (running)".format(
                episode, reward, running_reward, loss, running_loss
            )

            if episode % SAVE_EVERY_N_STEPS == 0:
                fname = os.path.join(SAVE_PATH, CHECKPOINT_PATTERN.format(episode))
                torch.save(policy_net.state_dict(), fname)
                with open(checkpoint_status_file, 'w') as f:
                    json.dump({'episode': episode,
                               'running_loss': running_loss,
                               'running_reward': running_reward,
                               'loss_history': loss_history,
                               'reward_history': reward_history}, f, indent=2)
                print "Save model to {}".format(fname)

if __name__ == '__main__':
    env = gym.make('Pong-v0')
    frame_preprocessor = Preprocessor()
    policy_net = PolicyNet(im_width=frame_preprocessor.output_info['image_width'],
                           im_height=frame_preprocessor.output_info['image_height'])
    learn_policy(policy_net, env, frame_preprocessor)
