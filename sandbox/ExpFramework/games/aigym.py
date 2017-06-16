"""
Environments from OpenAI gym
"""
import gym

class AtariEnvironment_Pong:
    def __init__(self):
        self._env = gym.make('Pong-v0')
        self._stopped = False

    def reset(self):
        self._stopped = False
        return self._env.reset()
    
    def step(self, act):
        if not self._stopped:
            s, r, t, info = self._env.step(act)
            if r!=0:
                t = True
            if t:
                self._stopped = True
        else:
            s = r = info = None
        return s, r, self._stopped, info
    
    def render(self):
        return self._env.render()
