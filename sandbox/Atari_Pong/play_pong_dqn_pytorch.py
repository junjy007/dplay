import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import gym
import os
import pdb
import logging
from collections import deque

def setup_logger(log_file_name=None):
    logFormatter = logging.Formatter("%(asctime)s] [%(levelname)-5.5s]  %(message)s") 
    # ("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger()
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)
    if log_file_name is None:
        return
    fileHandler = logging.FileHandler(log_file_name)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

# import cv2

I_LAST_STATE, I_ACTION, I_REWARD, I_NEW_STATE, I_TERM = 0, 1, 2, 3, 4
USE_CUDA = torch.cuda.is_available()

class Preprocessor:
    def __init__(self):
        self.output_info = {
            'image_width': 80,
            'image_height': 80,
        }
        pass

    # noinspection PyPep8Naming
    def process(self, I):
        """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
        I = I[35:195]  # crop
        I = I[::2, ::2, 0]  # downsample by factor of 2
        I[I == 144] = 0  # erase background (background type 1)
        I[I == 109] = 0  # erase background (background type 2)
        I[I != 0] = 1  # everything else (paddles, ball) just set to 1
        return I.astype(np.float32)


class QNet(nn.Module):
    def __init__(self, opts):
        super(QNet, self).__init__()
        self.opts = opts

        paddings = [ (self.opts[l]['kernel_size'] - 1) / 2
                     for l in ['conv-layer1', 'conv-layer2'] ]
        self.fn_conv1 = nn.Conv2d(
            in_channels=4,
            out_channels=self.opts['conv-layer1']['num_kernels'],
            kernel_size=self.opts['conv-layer1']['kernel_size'],
            padding=paddings[0],
            stride=1)
        self.fn_conv2 = nn.Conv2d(
            in_channels=self.opts['conv-layer1']['num_kernels'],
            out_channels=self.opts['conv-layer2']['num_kernels'],
            kernel_size=self.opts['conv-layer2']['kernel_size'],
            padding=paddings[1])
        num_final_conv_outputs = \
              self.opts['conv-layer2']['num_kernels'] \
            * self.opts['input']['im_width'] / 2 / 2 \
            * self.opts['input']['im_height'] / 2 / 2

        self.fc1 = nn.Linear(in_features=num_final_conv_outputs,
                             out_features=self.opts['fc1']['num_outputs'])
        self.fc2 = nn.Linear(in_features=self.opts['fc1']['num_outputs'],
                             out_features=self.opts['output']['num_actions'])

        self._conv1 = None
        self._conv2 = None
        self._fc1 = None
        self._fc2 = None


    def forward(self, x):
        self._conv1 = F.max_pool2d(F.relu(self.fn_conv1(x)), kernel_size=2)
        self._conv2 = F.max_pool2d(F.relu(self.fn_conv2(self._conv1)), kernel_size=2)
        self._flatten = self._conv2.view(-1, self.num_flat_features(self._conv2))
        self._fc1 = F.relu(self.fc1(self._flatten))
        self._fc2 = self.fc2(self._fc1)
        return self._fc2

    def num_flat_features(self, x):
        sample_size = x.size()[1:]
        return np.prod(sample_size)


class QLearner:
    NUM_FRAMES_PER_STATE = 4
    MEMORY_SIZE = 20000
    MIN_EXP_TO_TRAIN = 1000  # at least such experiences to start training
    MINIBATCH_SIZE = 100
    DISCOUNT = 0.99

    INITIAL_PROB_RANDOM_ACTION = 1.0
    FINAL_PROB_RANDOM_ACTION   = 0.05
    EXPLORE_STEPS = 10000

    SAVE_EVERY_N_STEPS = 10000

    def __init__(self, qnn, preproc, phrase='train', savepath='PongDQN'):
        """
        :param qnn: The net of q-function prediction
        :type qnn: torch.nn.Module
        """
        self.qnn = qnn
        self.preproc = preproc
        self._rng = np.random.RandomState(0)
        self.phrase = phrase

        self.experience = deque()
        self._last_action = None
        self._last_state = None
        self._is_first_frame = None
        self.reset_episode()

        self.loss_fn = torch.nn.MSELoss(size_average=False)
        self.optim   = torch.optim.Adagrad(qnn.parameters(), lr=1e-5)
        self.train_iter = 0

        self.prob_random_action = self.INITIAL_PROB_RANDOM_ACTION
        self.delta_prob_random_action = \
            (self.FINAL_PROB_RANDOM_ACTION - self.INITIAL_PROB_RANDOM_ACTION) \
            / self.EXPLORE_STEPS
        self.savepath = savepath
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        else:
            # if warm start, trying to load the model
            if os.path.exists(savepath+"/latest.txt"):
                with open(savepath+"/latest.txt", 'r') as f:
                    self.train_iter = int(f.read())
                fname = "{}/checkpoint-{}".format(self.savepath, self.train_iter)
                print "Load model from {}".format(fname)
                qnn.load_state_dict(torch.load(fname))


    # noinspection PyAttributeOutsideInit
    def reset_episode(self):
        self._last_action = [0, 0, 0, 0]
        self._last_state = None
        self._is_first_frame = True

    def train_step(self):
        # Take experience
        minibatch_idx = self._rng.choice(len(self.experience),
                                         size=self.MINIBATCH_SIZE,
                                         replace=False)
        x = np.ascontiguousarray([self.experience[s][0] for s in minibatch_idx],
                                 dtype=np.float32)
        x = Variable(torch.from_numpy(x), requires_grad=False)
        if USE_CUDA:
            x = x.cuda()
        q_pred_last_state_all_actions = self.qnn.forward(x)
        # extract the q-prediction for the particular action
        last_action = Variable(
            torch.from_numpy(
                np.ascontiguousarray([self.experience[s][1] for s in minibatch_idx],
                                     dtype=np.float32)
            ),
            requires_grad=False)
        if USE_CUDA:
            last_action = last_action.cuda()
        q_pred_last_state_last_action = \
            (q_pred_last_state_all_actions * last_action).sum(dim=1)

        x1 = np.ascontiguousarray([self.experience[s][3] for s in minibatch_idx],
                                  dtype=np.float32)
        x1 = Variable(torch.from_numpy(x1), requires_grad=False)
        if USE_CUDA:
            x1 = x1.cuda()
        q_pred_new_state_all_actions = self.qnn.forward(x1)
        q_pred_new_state_best_action = torch.max(q_pred_new_state_all_actions, dim=1)[0].squeeze()
        # q_pred_new_state_best_action.requires_grad = False
        q_pred_new_state_best_action_nograd = q_pred_new_state_best_action.detach()

        r = np.ascontiguousarray([self.experience[s][2] for s in minibatch_idx],
                                 dtype=np.float32)
        r = Variable(torch.from_numpy(r), requires_grad=False)
        if USE_CUDA:
            r = r.cuda()
        q_target = q_pred_new_state_best_action_nograd * self.DISCOUNT + r

        loss = self.loss_fn(q_pred_last_state_last_action, q_target)  # type: Variable

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return loss


    def predict_q(self, states):
        """
        Using current Q-Network to predict action values for states.

        NOTE: "states" is a set, for a single state, make a dummy single
        element set.
        :param states:
        :type states: np.ndarray
        :return:
        """
        assert states.ndim == 4, "Expect a 4-way tensor as " \
            "observation. For single observation, wrap it with " \
            "as 1 x ..."
        x = Variable(torch.from_numpy(states), volatile=True)
        if USE_CUDA:
            x = x.cuda()
        return self.qnn.forward(x)

    def process_observation(self, frame, reward, terminal):
        """
        When the environment has reacted to the last action, and returned us
         - reward
         - new-observation
         - if terminate
        deal with the new experience accordingly

        :param frame: Next frame from the game environment
        :type frame: np.ndarray
        :param reward: reward of next frame
        :param terminal: if the episode terminated: note Atari Pong will never
          return a terminate==True, instead, when a round has a winner, the
          reward will become -1 or +1 (it is zero in the process of the game)
        :return: None: but self._last_action is ready
        """
        # Preprocess self.preproc_fn(raw_pixel_data)
        # Will results in a gray-scale image of screen pixels
        observation = np.ascontiguousarray(self.preproc.process(frame))  # type:np.ndarray
        assert(len(observation.shape) == 2)  # gray-scale one channel
        # input = Variable(torch.from_numpy(observation)).unsqueeze(0).unsqueeze(0)  # type: Variable

        # Dealing with when a game starts (no last state)
        loss = None
        if self._is_first_frame:
            self._last_state = np.tile(observation[np.newaxis, :, :],
                                       (self.NUM_FRAMES_PER_STATE, 1, 1))
            self._is_first_frame = False
            return 0, loss

        current_state = np.concatenate((self._last_state[1:, :, :],
                                        observation[np.newaxis, :, :]),
                                        axis=0)

        # collect observations
        if self.phrase == 'train':
            self.experience.append(
                (self._last_state,
                 self._last_action,
                 reward,
                 current_state,
                 terminal))
            num_exp = len(self.experience)
            if num_exp > self.MEMORY_SIZE:
                self.experience.popleft()
            if num_exp > self.MIN_EXP_TO_TRAIN:
                loss = self.train_step()
                self.train_iter += 1
                if self.train_iter % self.SAVE_EVERY_N_STEPS == 0:
                    fname = "{}/checkpoint-{}".format(self.savepath,
                            self.train_iter)
                    print "Save model to {}".format(fname)
                    torch.save(self.qnn.state_dict(), fname)
                    with open(self.savepath+"/latest.txt", 'w') as f:
                        f.write(str(self.train_iter))

            if self.prob_random_action > self.FINAL_PROB_RANDOM_ACTION:
                self.prob_random_action += self.delta_prob_random_action

        if terminal:
            self.reset_episode()
            action = 0
        else:
            action_code = [0, 0, 0, 0]
            if self._rng.rand() > self.prob_random_action:
                # determine action by current qnn
                action_eval = self.predict_q(current_state[np.newaxis, ...])[0]  # type: Variable
                if USE_CUDA:
                    action_eval = action_eval.cpu()
                action_eval = action_eval.data.numpy()
                action = np.argmax(action_eval)
            else:
                action = self._rng.randint(4)
            action_code[action] = 1
            self._last_state = current_state
            self._last_action = action_code
        return action, loss

def dqn_main():
    MAX_EPISODES = 1000000
    RECENT_NUM = 10
    SAVE_PATH = 'dqn_tmp'
    VISUALISE_PER_N_EPISODES = 10
    env = gym.make('Pong-v0')

    preprocessor = Preprocessor()

    dqn_opts = {
        'input': {'im_width': preprocessor.output_info['image_width'],
                  'im_height': preprocessor.output_info['image_height']},
        'conv-layer1': {'num_kernels': 32,
                        'kernel_size': 3},
        'conv-layer2': {'num_kernels': 64,
                        'kernel_size': 3},
        'fc1': {'num_outputs': 256},
        'output': {'num_actions': 4}

    }
    qnn = QNet(dqn_opts)
    if USE_CUDA:
        print "Use CUDA"
        qnn.cuda()

    qlearn = QLearner(qnn, preprocessor)

    n_episodes_done = 0
    state = env.reset()
    reward = 0
    term   = False
    recent_rewards = []
    running_rewards = None
    total_steps = 0
    while n_episodes_done < MAX_EPISODES:
        if n_episodes_done % VISUALISE_PER_N_EPISODES == 0:
            env.render()

        action, loss = qlearn.process_observation(state, reward, term)
        state, reward, term, _ = env.step(action)
        total_steps += 1
        if loss:
            loss = loss.data[0]

        if reward != 0:
            print "ep {}, total_step {}, reward {}, loss {}".format(n_episodes_done,
                                                         total_steps, reward, loss)
            state = env.reset()
            reward = 0
            term = True
            n_episodes_done += 1
            recent_rewards.append(reward)
            if n_episodes_done % RECENT_NUM == 0:
                r = np.mean(recent_rewards)
                running_rewards = r if running_rewards is None \
                    else running_rewards * 0.99 + r
                recent_rewards = []
                message = "ep {}, total-step {}, train-step {}, train-loss {}, R {}, " \
                          "recent [{}]/running R {}".format(n_episodes_done, total_steps,
                          qlearn.train_iter, loss, RECENT_NUM, r, running_rewards )
                logging.info(message)

if __name__=='__main__':
    setup_logger("{}.log".format(__file__))
    dqn_main()

