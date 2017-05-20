import tensorflow as tf
from tensorflow.python.framework import dtypes as tf_dtypes
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import gym
import os
from collections import deque

# from pygame import K_DOWN, K_UP
# from pygame_player import PyGamePlayer
# import cv2

I_LAST_STATE, I_ACTION, I_REWARD, I_NEW_STATE, I_TERM = 0, 1, 2, 3, 4

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
        return I.astype(np.float)


class QNet(nn.Module):
    def __init__(self, opts):
        super(QNet, self).__init__()
        self.opts = opts

        paddings = [ (self.opts[l]['kernel_size'] - 1) / 2
                     for l in ['conv-layer1', 'conv-layer2'] ]
        padding1 = (self.opts['conv-layer1']['kernel_size'] - 1) / 2
        self.fn_conv1 = nn.Conv2d(
            in_channels=1,
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


class DeepQPlayer:
    NUM_ACTIONS = 4
    NUM_FRAMES_PER_STATE = 4
    CONV_LAYER_1 = {'NUM_KERNELS': 32,
                    'KERNEL_HEIGHT': 8,
                    'KERNEL_WIDTH': 8,
                    'STRIDE_HEIGHT': 4,
                    'STRIDE_WIDTH': 4,
                    'POOL_HEIGHT': 2,
                    'POOL_WIDTH': 2,
                    }
    CONV_LAYER_2 = {'NUM_KERNELS': 64,
                    'KERNEL_HEIGHT': 4,
                    'KERNEL_WIDTH': 4,
                    'STRIDE_HEIGHT': 2,
                    'STRIDE_WIDTH': 2,
                    'POOL_HEIGHT': 2,
                    'POOL_WIDTH': 2,
                    }
    FC_LAYER_1 = {'NUM_UNITS': 256}

    LEARN_RATE = 1e-6

    GAMMA=0.99
    INIT_PROB_RANDOM_ACTION = 1.
    FINAL_PROB_RANDOM_ACTION = 0.005
    NUM_RANDOM_EXPLORE_STEPS = 500000
    REDUCE_RANDOM_PER_STEP = \
        (INIT_PROB_RANDOM_ACTION - FINAL_PROB_RANDOM_ACTION) \
        / NUM_RANDOM_EXPLORE_STEPS

    MINI_BATCH_SIZE = 200
    OBS_MEM_SIZE = 500000
    MIN_INIT_OBS = 50000
    SAVE_INTERVAL = 1000

    def __init__(self,
                 im_preproc,
                 checkpoint_path="tmp_dqn",
                 phrase='train',
                 warm_start=True,
                 verbose=0,
                 randomseed=0):
        """
        :param im_preproc:
        :type im_preproc: Preprocessor
        :param checkpoint_path:
        :param phrase:
        :param warm_start:
        :param verbose:
        :param randomseed:
        """
        #
        # 1. Setting up constants, etc.
        #
        self._rng = np.random.RandomState(randomseed)
        self.last_score = 0
        self.verbose = verbose
        self.checkpoint_path = checkpoint_path
        self.phrase = phrase
        self.preproc = im_preproc

        #
        # 2. Build deep network to learn the Q-function
        #
        tf.reset_default_graph()  # in iPython env this is useful
        self._session = tf.Session()
        self._input_layer, self._output_layer = self._create_network()
        self._action = tf.placeholder("float", [None, self.NUM_ACTIONS])  # DQN is an off policy algorithm,
        # So the experience includes the actual actions that are taken.
        self._target = tf.placeholder("float", [None])  # TD-style backup evaluation of the actions

        # self._output_layer is the predicted value for the actions
        # self._action should be one_hot_code, such as [0,0,1]
        readout_action = tf.reduce_sum(
            tf.multiply(self._output_layer, self._action),
            reduction_indices=1)
        cost = tf.reduce_mean(tf.square(self._target - readout_action))
        self._train_operation = tf.train.AdamOptimizer(self.LEARN_RATE).minimize(cost)

        self._observations = deque()
        self._last_scores = deque()

        #
        # 3. Init the game play
        #
        self._is_first_frame = True
        self._last_action = np.zeros(self.NUM_ACTIONS)
        self._last_action[1] = 1  # [0,1,0] not moving

        self._last_state = None
        self._prob_random_action = self.INIT_PROB_RANDOM_ACTION
        self._train_iter = 0

        self._session.run(tf.initialize_all_variables())
        if not os.path.exists(self.checkpoint_path):
            os.mkdir(self.checkpoint_path)
        self._saver = tf.train.Saver()
        cp = tf.train.get_checkpoint_state(self.checkpoint_path)

        if cp and cp.model_checkpoint_path and warm_start:
            self._saver.restore(self._session, cp.model_checkpoint_path)
            print("Loaded checkpoints %s" % cp.model_checkpoint_path)
        elif self.phrase == 'test':
            raise Exception("Could not load checkpoints for playback")


    def get_train_iter_steps(self):
        return self._train_iter

    def process_observation(self, frame_pixels, reward, terminal):
        """
        When the environment has reacted to the last action, and returned us
         - reward
         - new-observation
         - if terminate
        deal with the new experience accordingly

        :param frame_pixels: Next frame from the game environment
        :type frame_pixels: np.ndarray
        :param reward: reward of next frame
        :param terminal: if the episode terminated: note Atari Pong will never
          return a terminate==True, instead, when a round has a winner, the
          reward will become -1 or +1 (it is zero in the process of the game)
        :return: None: but self._last_action is ready
        """
        # Preprocess self.preproc_fn(raw_pixel_data)
        # Will results in a gray-scale image of screen pixels
        observation = self.preproc.process(frame_pixels)  # type:np.ndarray
        assert(len(observation.shape) == 2)  # gray-scale one channel

        # Dealing with when a game starts (no last state)
        if self._is_first_frame:
            self._last_state = np.tile(observation[:, :, np.newaxis],
                                       (1, 1, self.NUM_FRAMES_PER_STATE))
            self._last_action = np.asarray([0, 0, 1, 0]) if self._rng.rand() < 0.5 \
                else np.asarray([0, 0, 0, 1])
            return self._last_action

        current_state = np.concatenate((self._last_state[:, :, 1:], observation[:, :, np.newaxis]),
                                       axis=2)

        # collect observations
        if self.phrase == 'train':
            self._observations.append(
                (self._last_state,
                 self._last_action,
                 reward,
                 current_state,
                 terminal))
            if len(self._observations) > self.OBS_MEM_SIZE:
                self._observations.popleft()
            if len(self._observations) > self.MIN_INIT_OBS:
                self._train()
                self._train_iter += 1
            if self._prob_random_action > self.FINAL_PROB_RANDOM_ACTION:
                random_reduce = \
                    (self.INIT_PROB_RANDOM_ACTION - self.FINAL_PROB_RANDOM_ACTION) / \
                    self.NUM_RANDOM_EXPLORE_STEPS
                self._prob_random_action -= random_reduce  # self.REDUCE_RANDOM_PER_STEP

        if terminal:
            self._is_first_frame = True  # will not collect experience
            return [0, 0, 0, 0]  # the action doesn't matter

        self._last_state = current_state
        self._last_action = self._get_action_code(current_state)
        return self._last_action


    def _get_action_code(self, s):
        action_code = np.zeros(self.NUM_ACTIONS)
        if self.phrase == 'train' and self._rng.rand() < self._prob_random_action:
            act_i = self._rng.randint(self.NUM_ACTIONS)
        else:
            readout = self._session.run(self._output_layer,
                                        feed_dict={self._input_layer: [s, ]})[0]
            act_i = np.argmax(readout)
        action_code[act_i] = 1
        return action_code

    def _train(self):
        obs_batch = self._rng.choice(self._observations,
                                     size=self.MINI_BATCH_SIZE,
                                     replace=False)

        obs_last_states = [d[I_LAST_STATE] for d in obs_batch]
        obs_actions = [d[I_ACTION] for d in obs_batch]
        obs_rewards = [d[I_REWARD] for d in obs_batch]
        obs_new_states = [d[I_NEW_STATE] for d in obs_batch]
        # Using current q-function to evaluate the actions in the NEW states
        q_new = self._session.run(self._output_layer,
                                  feed_dict={self._input_layer: obs_new_states})
        target_reward = []
        for i in range(self.MINI_BATCH_SIZE):
            if obs_batch[i][I_TERM]:
                target_reward.append(obs_rewards[i])
            else:
                target_reward.append(
                    obs_rewards[i] + self.GAMMA * np.max(q_new[i]))

        self._session.run(self._train_operation,
                          feed_dict={
                              self._input_layer: obs_last_states,
                              self._action: obs_actions,
                              self._target: target_reward})

        if self._train_iter % self.SAVE_INTERVAL == 0:
            self._saver.save(self._session,
                             os.path.join(self.checkpoint_path,
                                          'network'),
                             global_step=self._train_iter)



    def _create_network(self):
        imh = self.preproc.output_info['image_height']
        imw = self.preproc.output_info['image_width']
        w1 = tf.Variable(tf.truncated_normal(
            [self.CONV_LAYER_1['KERNEL_HEIGHT'],
             self.CONV_LAYER_1['KERNEL_WIDTH'],
             self.NUM_FRAMES_PER_STATE,
             self.CONV_LAYER_1['NUM_KERNELS']],
            stddev=0.01))
        b1 = tf.Variable(tf.constant(0.01,
                                     shape=[self.CONV_LAYER_1['NUM_KERNELS'], ]))

        w2 = tf.Variable(tf.truncated_normal(
            [self.CONV_LAYER_2['KERNEL_HEIGHT'],
             self.CONV_LAYER_2['KERNEL_WIDTH'],
             self.CONV_LAYER_1['NUM_KERNELS'],
             self.CONV_LAYER_2['NUM_KERNELS']],
            stddev=0.01))
        b2 = tf.Variable(tf.constant(0.01,
                                     shape=[self.CONV_LAYER_2['NUM_KERNELS'], ]))

        # nr1 = np.ceil(float(self.SCREEN_HEIGHT)
        #               / self.CONV_LAYER_1['STRIDE_HEIGHT'])
        # # ceil <- we use 'SAME' for paddling
        # nc1 = np.ceil(float(self.SCREEN_WIDTH)
        #               / self.CONV_LAYER_1['STRIDE_WIDTH'])
        # print("(conv-1):(%d, %d)" % (nr1, nc1))
        # nr1 = np.ceil(float(nr1) / self.CONV_LAYER_1['POOL_HEIGHT'])
        # nc1 = np.ceil(float(nc1) / self.CONV_LAYER_1['POOL_WIDTH'])
        # print("(conv-pool-1):(%d, %d)" % (nr1, nc1))
        # nr2 = np.ceil(float(nr1) / self.CONV_LAYER_2['STRIDE_HEIGHT'])
        # nc2 = np.ceil(float(nc1) / self.CONV_LAYER_2['STRIDE_WIDTH'])
        # print("(conv-2):(%d, %d)" % (nr2, nc2))
        # nr2 = np.ceil(float(nr2) / self.CONV_LAYER_2['POOL_HEIGHT'])
        # nc2 = np.ceil(float(nc2) / self.CONV_LAYER_2['POOL_WIDTH'])
        # print("(conv-pool-2):(%d, %d)" % (nr2, nc2))
        # num_convpool_outputs = int(nr2 * nc2 * self.CONV_LAYER_2['NUM_KERNELS'])
        # print("units:%d" % (num_convpool_outputs,))
        # print("hidden fc-layer %d" % (self.FC_LAYER_1['NUM_UNITS'],))


        input_layer = tf.placeholder(dtype=tf_dtypes.float32,
                                     shape=[None, imh, imw, self.NUM_FRAMES_PER_STATE])

        sr_ = self.CONV_LAYER_1['STRIDE_HEIGHT']
        sc_ = self.CONV_LAYER_1['STRIDE_WIDTH']
        conv_layer_1 = tf.nn.relu(
            tf.nn.conv2d(input_layer, w1,
                         strides=[1, sr_, sc_, 1], padding='SAME')  # conv2d
            + b1)  # ReLU

        sr_ = self.CONV_LAYER_1['POOL_HEIGHT']
        sc_ = self.CONV_LAYER_1['POOL_WIDTH']
        pool_layer1 = tf.nn.max_pool(conv_layer_1,
                                     ksize=[1, sr_, sc_, 1],
                                     strides=[1, sr_, sc_, 1],
                                     padding='SAME')

        sr_ = self.CONV_LAYER_2['STRIDE_HEIGHT']
        sc_ = self.CONV_LAYER_2['STRIDE_WIDTH']
        conv_layer_2 = tf.nn.relu(
            tf.nn.conv2d(pool_layer1, w2,
                         strides=[1, sr_, sc_, 1], padding='SAME') + b2)

        sr_ = self.CONV_LAYER_2['POOL_HEIGHT']
        sc_ = self.CONV_LAYER_2['POOL_WIDTH']
        pool_layer2 = tf.nn.max_pool(conv_layer_2,
                                     ksize=[1, sr_, sc_, 1],
                                     strides=[1, sr_, sc_, 1],
                                     padding='SAME')


        print "To Create the Fully Connected Layers"
        ## flattened_layer = tf.reshape(pool_layer2, [-1, num_convpool_outputs])

        ## w3 = tf.Variable(tf.truncated_normal(
        ##     [num_convpool_outputs, self.FC_LAYER_1['NUM_UNITS']], stddev=0.01))
        ## b3 = tf.Variable(tf.constant(0.01, shape=[self.FC_LAYER_1['NUM_UNITS'], ]))
        ## w4 = tf.Variable(tf.truncated_normal(
        ##     [self.FC_LAYER_1['NUM_UNITS'], self.NUM_ACTIONS], stddev=0.01))
        ## b4 = tf.Variable(tf.constant(0.01, shape=[self.NUM_ACTIONS, ]))

        ## fc_layer = tf.nn.relu(tf.matmul(flattened_layer, w3) + b3)

        ## output_layer = tf.matmul(fc_layer, w4) + b4

        output_layer = None

        return input_layer, output_layer


def dqn_main():
    MAX_EPISODES = 1000000
    RECENT_NUM = 100
    SAVE_PATH = 'dqn_tmp'
    VISUALISE_PER_N_EPISODES = 100
    env = gym.make('Pong-v0')

    state = env.reset()
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
    qn = QNet(dqn_opts)

    raw_frame = env.reset()
    observation = preprocessor.process(raw_frame)  # type: np.ndarray
    print "Get observation {} of {}".format(observation.shape, observation.dtype)
    observation1 = np.ascontiguousarray(observation, dtype=np.float32)

    input = Variable(torch.from_numpy(observation1))
    input = input.unsqueeze(0).unsqueeze(0)
    print "Input {}".format(input.size())
    x = qn.forward(input)
    # print "Conv layer 1 output: {}".format(qn._conv1)
    print "Conv done {} -> {}".format(input.size(), x.size())


    loss_fn = torch.nn.MSELoss(size_average=False)
    optimiser = torch.optim.Adagrad(qn.parameters(), lr=1e-5)
    ####

    # dql = DeepQPlayer(checkpoint_path=SAVE_PATH,
    #                 im_preproc=preprocessor)

    return

    n_episodes_done = 0
    reward = 0
    term   = 0
    recent_rewards = []
    running_rewards = None
    while n_episodes_done < MAX_EPISODES:
        if n_episodes_done % VISUALISE_PER_N_EPISODES == 0:
            env.render()

        action_code = dql.process_observation(state, reward, term)
        action = np.argmax(action_code)
        state, reward, term, _ = env.step(action)

        if reward != 0:
            term = True
            n_episodes_done += 1
            recent_rewards.append(reward)
            if n_episodes_done % RECENT_NUM == 0:
                r = np.mean(recent_rewards)
                running_rewards = r if running_rewards is None \
                    else running_rewards * 0.99 + r
                print "Ep {} / Train {}: Recent {} episodes average reward {}, long-term reward {}".format(
                    n_episodes_done, dql.get_train_iter_steps(), RECENT_NUM, r, running_rewards)

if __name__=='__main__':
    dqn_main()

