"""
Minimalist playing with Pong Game, using one hidden layer.
"""

import numpy as np
import tensorflow as tf
import gym
import os

# global parameters
batch_size = 1  # every how many episodes to do a param update?
save_every_n_batches = 500
render_every_n_episodes = 50
learning_rate = 1e-6  # feel free to play with this to train faster or more stably.
gamma = 0.99  # discount factor for reward
H = 200  # number of hidden layer neurons
D = 80 * 80  # input dimensionality


def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


# noinspection PyPep8Naming
def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()


# noinspection PyShadowingNames
class PDNet:
    # noinspection PyPep8Naming
    def __init__(self, D, H):
        self.observations_pl = None
        self.action_probability_op = None
        self.action_samples_pl = None
        self.retrospect_advantages_pl = None
        self.grads_op = None
        self.W1_grad_pl = None
        self.W2_grad_pl = None
        self.update_W_op = None
        self.W_dims = None
        self.D = D
        self.H = H

        self._build_net()

    def _build_net(self):
        H = self.H
        D = self.D

        # This defines the network as it goes from taking an observation of the environment to
        # giving a probability of chosing to the action of moving left or right.
        observations = tf.placeholder(tf.float32, [None, D], name="input_x")
        W1 = tf.get_variable("W1", shape=[D, H],
                             initializer=tf.contrib.layers.xavier_initializer())
        layer1 = tf.nn.relu(tf.matmul(observations, W1))
        W2 = tf.get_variable("W2", shape=[H, 1],
                             initializer=tf.contrib.layers.xavier_initializer())
        score = tf.matmul(layer1, W2)
        probability = tf.nn.sigmoid(score)

        # From here we define the parts of the network needed for learning a good policy.
        # tvars = tf.trainable_variables()
        tvars = [W1, W2]  # to ensure the order
        input_y = tf.placeholder(tf.float32, [None, 1], name="actions")
        advantages = tf.placeholder(tf.float32, name="reward_signal")

        ################
        # TO BATCH HERE
        ################

        # The loss function. This sends the weights in the direction of making actions
        # that gave good advantage (reward over time) more likely, and actions that didn't less likely.
        loglik = tf.log(input_y * (input_y - probability) + (1 - input_y) * (input_y + probability))
        loss = -tf.reduce_mean(loglik * advantages)
        newGrads = tf.gradients(loss, tvars)

        # Once we have collected a series of gradients from multiple episodes, we apply them.
        # We don't just apply gradeients after every episode in order to account for noise in the reward signal.
        adam = tf.train.AdamOptimizer(learning_rate=learning_rate)  # Our optimizer
        W1Grad = tf.placeholder(tf.float32,
                                name="batch_grad1")  # Placeholders to send the final gradients through when we update.
        W2Grad = tf.placeholder(tf.float32, name="batch_grad2")
        batchGrad = [W1Grad, W2Grad]
        updateGrads = adam.apply_gradients(zip(batchGrad, tvars))
        train_op = adam.minimize(loss)

        self.observations_pl = observations
        self.action_probability_op = probability
        self.action_samples_pl = input_y
        self.retrospect_advantages_pl = advantages
        self.grads_op = newGrads
        self.W1_grad_pl = W1Grad
        self.W2_grad_pl = W2Grad
        self.update_W_op = updateGrads
        self.W_dims = [(D, H), (H, 1)]
        self.train_op = train_op

    # WORKING ZONE
    def _build_net_2(self):
        H = self.H
        D = self.D

        # This defines the network as it goes from taking an observation of the environment to
        # giving a probability of chosing to the action of moving left or right.
        observations = tf.placeholder(tf.float32, [None, D], name="input_x")
        W1 = tf.get_variable("W1", shape=[D, H],
                             initializer=tf.contrib.layers.xavier_initializer())
        layer1 = tf.nn.relu(tf.matmul(observations, W1))
        W2 = tf.get_variable("W2", shape=[H, 1],
                             initializer=tf.contrib.layers.xavier_initializer())
        score = tf.matmul(layer1, W2)
        probability = tf.nn.sigmoid(score)

        # From here we define the parts of the network needed for learning a good policy.
        # tvars = tf.trainable_variables()
        tvars = [W1, W2]  # to ensure the order
        input_y = tf.placeholder(tf.float32, [None, 1], name="actions")
        advantages = tf.placeholder(tf.float32, name="reward_signal")

        ################
        # TO BATCH HERE
        ################

        # The loss function. This sends the weights in the direction of making actions
        # that gave good advantage (reward over time) more likely, and actions that didn't less likely.
        loglik = tf.log(input_y * (input_y - probability) + (1 - input_y) * (input_y + probability))
        loss = -tf.reduce_mean(loglik * advantages)
        newGrads = tf.gradients(loss, tvars)

        # Once we have collected a series of gradients from multiple episodes, we apply them.
        # We don't just apply gradeients after every episode in order to account for noise in the reward signal.
        adam = tf.train.AdamOptimizer(learning_rate=learning_rate)  # Our optimizer
        W1Grad = tf.placeholder(tf.float32,
                                name="batch_grad1")  # Placeholders to send the final gradients through when we update.
        W2Grad = tf.placeholder(tf.float32, name="batch_grad2")
        batchGrad = [W1Grad, W2Grad]
        updateGrads = adam.apply_gradients(zip(batchGrad, tvars))
        train_op = adam.minimize(loss)

        self.observations_pl = observations
        self.action_probability_op = probability
        self.action_samples_pl = input_y
        self.retrospect_advantages_pl = advantages
        self.grads_op = newGrads
        self.W1_grad_pl = W1Grad
        self.W2_grad_pl = W2Grad
        self.update_W_op = updateGrads
        self.W_dims = [(D, H), (H, 1)]
        self.train_op = train_op


# noinspection PyShadowingNames
def learner(env, policy_net, save_path=None):
    """
    Learn a playing agentcy

    :param env:
    :param policy_net:
    :type policy_net: PDNet
    :return:
    """
    xs, hs, drs, ys = [], [], [], []
    recent_batches_avg_reward = None
    reward_sum = 0
    episode_number = 1
    total_episodes = 10000
    num_training_steps = 0
    init = tf.initialize_all_variables()

    # Launch the graph
    global_step = tf.Variable(0, name='global_step', trainable=False)
    global_step_inc = tf.assign_add(global_step, 1, name='global_step_inc')
    saver = tf.train.Saver(
        max_to_keep=10,
        keep_checkpoint_every_n_hours=1)

    sess = tf.Session()
    cp_state = tf.train.get_checkpoint_state(save_path)
    if cp_state and cp_state.model_checkpoint_path:
        saver.restore(sess, save_path)
        print "Restore training from {} @ step {}".format(save_path, global_step.eval(sess))
    else:
        sess.run(tf.global_variables_initializer())
        print "Training from scratch"

    sess.run(init)
    observation = env.reset()  # Obtain an initial observation of the environment
    prev_x = None

    # Reset the gradient placeholder. We will collect gradients in
    # gradBuffer until we are ready to update our policy network.
    gradBuffer = [np.zeros(w_shape) for w_shape in policy_net.W_dims]
    steps = 0
    mini_batch_epx, mini_batch_epy, mini_batch_epr = [], [], []
    while episode_number <= total_episodes:
        render = episode_number % render_every_n_episodes == 0
        if render:
            env.render()

        # preprocess the observation, set input to network to be difference image
        cur_x = prepro(observation)
        x = cur_x - prev_x if prev_x is not None else np.zeros(D)
        prev_x = cur_x

        # Run the policy network and get an action to take.
        tfprob = sess.run(policy_net.action_probability_op,
                          feed_dict={policy_net.observations_pl: x.reshape((1, policy_net.D))})
        action = 2 if np.random.uniform() < tfprob else 3  # up and down
        if render:
            print "ep {} step {}: up-prob: {}, action: {}".format(
                episode_number, steps, tfprob, action)

        xs.append(x)  # observation
        y = 1 if action == 2 else 0  # a "fake label"
        ys.append(y)

        # step the environment and get new measurements
        observation, reward, _, _ = env.step(action)  # termination is judged by reward
        drs.append(reward)  # record reward (has to be done after we call step() to get reward for previous action)
        steps += 1

        if reward != 0:  # whenever there is a reward, game is ended.
            observation = env.reset()
            steps = 0
            reward_sum += reward
            episode_number += 1
            # stack together all inputs, hidden states, action gradients, and rewards for this episode
            epx = np.vstack(xs)
            epy = np.vstack(ys)
            epr = np.vstack(drs)
            xs, hs, dlogps, drs, ys = [], [], [], [], []  # reset array memory

            # compute the discounted reward backwards through time
            discounted_epr = discount_rewards(epr)
            # size the rewards to be unit normal (helps control the gradient estimator variance)
            discounted_epr -= np.mean(discounted_epr)
            reward_stdvar = np.std(discounted_epr)
            MIN_REARD_STDVAR_TO_NORMALISE = 1e-6
            if reward_stdvar > MIN_REARD_STDVAR_TO_NORMALISE:
                discounted_epr /= np.std(discounted_epr)

            mini_batch_epx.append(epx)
            mini_batch_epy.append(epy)
            mini_batch_epr.append(discounted_epr)

            # # Get the gradient for this episode, and save it in the gradBuffer
            # tGrad = sess.run(policy_net.grads_op,
            #                  feed_dict={policy_net.observations_pl: epx,
            #                             policy_net.action_samples_pl: epy,
            #                             policy_net.retrospect_advantages_pl: discounted_epr})
            # for ix, grad in enumerate(tGrad):
            #     gradBuffer[ix] += grad

            # If we have completed enough episodes, then update the policy network with our gradients.
            if episode_number % batch_size == 0:
                sess.run(policy_net.train_op, feed_dict={
                    policy_net.observations_pl: np.vstack(mini_batch_epx),
                    policy_net.action_samples_pl: np.vstack(mini_batch_epy),
                    policy_net.retrospect_advantages_pl: np.vstack(mini_batch_epr)
                })
                sess.run(global_step_inc)
                mini_batch_epx, mini_batch_epy, mini_batch_epr = [], [], []
                num_training_steps += 1

                if num_training_steps % save_every_n_batches == 0:
                    saver.save(sess, save_path, global_step=global_step)


                # Output Loss, Episode number, training times, Recent reword
                mini_batch_avg_reward = reward_sum / float(batch_size)
                recent_batches_avg_reward = mini_batch_avg_reward \
                        if recent_batches_avg_reward is None \
                        else recent_batches_avg_reward * 0.99 + mini_batch_avg_reward * 0.01

                reward_message = "Reward: Batch {:.3f}, Recent {:.3f}".format(
                    mini_batch_avg_reward, recent_batches_avg_reward )
                batch_message = "Ep {:6d}, Mini-batch {:5d}".format(episode_number, num_training_steps)
                print "[{}] {}".format(batch_message, reward_message)

                reward_sum = 0.0

            #     sess.run(policy_net.update_W_op,
            #              feed_dict={policy_net.W1_grad_pl: gradBuffer[0],
            #                         policy_net.W2_grad_pl: gradBuffer[1]})
            #     for gi in range(len(gradBuffer)):
            #         gradBuffer[gi][...] = 0.

            #     # Give a summary of how well our network is doing for each batch of episodes.
            #     reward_sum /= float(batch_size)
            #     running_reward = reward_sum if running_reward is None \
            #         else running_reward * 0.99 + reward_sum * 0.01

            #     if running_reward > 0.5:
            #         print "Task solved in", episode_number, 'episodes!'

            #    print "ep {:d}:, recent {:d} reward: {:.3f}, longterm reward {:.3f}".format(
            #        episode_number, batch_size, reward_sum, running_reward)

            #    reward_sum = 0


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        save_path = os.path.join(sys.argv[1], 'checkpoint')
    else:
        save_path = None
    env = gym.make('Atari_Pong-v0')
    policy_net = PDNet(D, H)
    learner(env, policy_net, save_path)
