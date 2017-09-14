"""
21 Aug 2017
Adopted from github.com/transedward/pytorch-dqn
I cannot figure out all the details from their article. Check this implementation.

Usage:
    maincontra.py [state STATE_FILE] [reward REWARD_SCHEME]
                  [--save_dir=<sdir>] [--continue-play]

Options:
    --save_dir=<sdir>   where to save checkpoints [default: checkpoints_contra]
    --continue-play     if true, when killed, will continue
"""

import gym
import torch.optim as optim
import docopt
from dqn_model import DQN
from dqn_learn_contra import OptimiserSpec, dqn_learning
from dqn_utils.mygym import get_env, get_wrapper_by_name
from dqn_utils.schedule import LinearSchedule
from dqn_utils.env_wrapper_NES import get_contra_env

BATCH_SIZE = 32
GAMMA = 0.99
REPLAY_BUFFER_SIZE = 200000
LEARNING_STARTS = 50000
LEARNING_FREQ = 4
FRAME_HISTORY_LEN = 4
TARGER_UPDATE_FREQ = 10000
LEARNING_RATE = 0.00025
ALPHA = 0.95
EPS = 0.01


# noinspection PyShadowingNames
def main(env, num_timesteps, save_dir):
    # noinspection PyShadowingNames
    def stopping_criterion(env):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return False

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
        save_dir=save_dir
    )

def evaluate(env, checkpoint):
    # load the model
    # start env
    state = env.reset()

if __name__ == '__main__':
    # Run training
    args = docopt.docopt(__doc__)
    state_file = args['STATE_FILE'] if args['state'] else None
    reward_scheme_file = args['REWARD_SCHEME'] if args['reward'] else "Contra_reward_scheme_gameplay0.json"
    print reward_scheme_file
    env = get_contra_env(state_file, args['--continue-play'], reward_scheme_file)
    main(env, 50000000, args['--save_dir'])
