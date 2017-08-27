"""
21 Aug 2017
Adopted from github.com/transedward/pytorch-dqn
I cannot figure out all the details from their article. Check this implementation.

"""

import gym
import torch.optim as optim

from dqn_model import DQN
from dqn_learn import OptimiserSpec, dqn_learning
from dqn_utils.mygym import get_env, get_wrapper_by_name
from dqn_utils.schedule import LinearSchedule

BATCH_SIZE = 32
GAMMA = 0.99
REPLAY_BUFFER_SIZE = 1000000
LEARNING_STARTS = 50000
LEARNING_FREQ = 4
FRAME_HISTORY_LEN = 4
TARGER_UPDATE_FREQ = 10000
LEARNING_RATE = 0.00025
ALPHA = 0.95
EPS = 0.01


# noinspection PyShadowingNames
def main(env, num_timesteps):
    # noinspection PyShadowingNames
    def stopping_criterion(env):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return get_wrapper_by_name(env, "Monitor").get_total_steps() >= num_timesteps

    optimizer_spec = OptimiserSpec(
        constructor=optim.RMSprop,
        kwargs=dict(lr=LEARNING_RATE, alpha=ALPHA, eps=EPS),
    )

    exploration_schedule = LinearSchedule(1000000, 0.1)

    dqn_learning(
        env=env,
        q_func=DQN,
        optimiser_spec=optimizer_spec,
        exploration=exploration_schedule,
        stopping_criterion=stopping_criterion,
        replay_buffer_size=REPLAY_BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        learning_starts=LEARNING_STARTS,
        learning_freq=LEARNING_FREQ,
        frame_history_len=FRAME_HISTORY_LEN,
        target_update_freq=TARGER_UPDATE_FREQ,
        save_dir='checkpoints'
    )

def evaluate(env, checkpoint):
    # load the model
    # start env
    state = env.reset()

if __name__ == '__main__':
    # Get Atari games.
    benchmark = gym.benchmark_spec('Atari40M')

    # Change the index to select a different game.
    task = benchmark.tasks[0]

    # Run training
    seed = 0  # Use a seed of zero (you may want to randomize the seed!)
    env = get_env(task, seed)

    main(env, task.max_timesteps)
