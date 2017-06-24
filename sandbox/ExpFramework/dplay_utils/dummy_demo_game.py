import gym
import random

def make_game():
    return gym.make('Pong-v0')

def make_policy(cp_dir, cp_id):
    def policy(s):
        return random.choice(4)

    return policy
