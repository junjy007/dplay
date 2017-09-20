import os
import time
import numpy as np
from nes_env.nes_python_interface import NESInterface
import json


ACT_NOOP = 0
ACT_A = 1
ACT_B = 2
ACT_UP = 3
ACT_RIGHT = 4
ACT_LEFT = 5
ACT_DOWN = 6
ACT_A_UP = 7
ACT_A_RIGHT = 8
ACT_A_LEFT = 9
ACT_A_DOWN = 10
ACT_B_UP = 11
ACT_B_RIGHT = 12
ACT_B_LEFT = 13
ACT_B_DOWN = 14
ACT_B_UP_RIGHT = 15
ACT_B_UP_LEFT = 16
ACT_B_DOWN_RIGHT = 17
ACT_B_DOWN_LEFT = 18
ACT_RANDOM = 20
ACT_SELECT  =  22
ACT_START  =  23
SECOND_ACTS = [
    ACT_NOOP,  # ACT_NOOP = 0
    ACT_NOOP,  # ACT_A = 1
    ACT_NOOP,  # ACT_B = 2
    ACT_UP,    # ACT_UP = 3
    ACT_RIGHT, # ACT_RIGHT = 4
    ACT_LEFT,  # ACT_LEFT = 5
    ACT_DOWN,  # ACT_DOWN = 6
    ACT_UP,    # ACT_A_UP = 7
    ACT_RIGHT, # ACT_A_RIGHT = 8
    ACT_LEFT,  # ACT_A_LEFT = 9
    ACT_DOWN,  # ACT_A_DOWN = 10
    ACT_UP,  # ACT_B_UP = 11
    ACT_RIGHT,  # ACT_B_RIGHT = 12
    ACT_LEFT,  # ACT_B_LEFT = 13
    ACT_DOWN,  # ACT_B_DOWN = 14
    ACT_RIGHT,  # ACT_B_UP_RIGHT = 15
    ACT_LEFT,  # ACT_B_UP_LEFT = 16
    ACT_RIGHT,  # ACT_B_DOWN_RIGHT = 17
    ACT_LEFT,  # ACT_B_DOWN_LEFT = 18
]

GAME_CONTRA = 1 # 1game id for Contra


SCORE_IDX = 0
LIVE_STATUS_MEANING = ["Dead", "Live", "Dying"]
LIVE_STATUS_IDX = 2
PLAY_STATUS_MEANING = ["NO-PLAY", "NO-PLAY", "NO-PLAY", "NO-PLAY", "PLAYING",
               "NO-PLAY", "NO-PLAY", "NO-PLAY", "WIN", "WIN"]
PLAY_STATUS_IDX = 4

class ActionSpace:
    def __init__(self, n):
        self.n = n
        self.random_act_id = ACT_RANDOM

class ObservationSpace:
    def __init__(self, shape):
        self.shape = list(shape)

class EnvContra(object):
    def __init__(self, state_fname=None, continue_play=False,
                 reward_scheme_file="Contra_reward_scheme_gameplay0.json"):
        wd = os.path.split(os.path.abspath(__file__))[0]
        nespath = os.path.join(wd, "nes_env", "res", "Contra.nes")
        reward_scheme_file_full = os.path.join(wd, "nes_env", "res", reward_scheme_file)
        self.game = NESInterface(nespath, GAME_CONTRA)
        self.height = self.game.height
        self.width = self.game.width
        self._screen_rgb = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self._screen_1 = np.zeros((self.height, self.width), dtype=np.uint8)
        self._score = 0
        self._live_status = 0
        self._play_status = 0
        self._boss_killed = 0
        self.action_space = ActionSpace(19)
        self.observation_space = ObservationSpace([self.height, self.width, 3])
        self.total_killed = 0
        self.total_win = 0
        self.enemy_hp = np.zeros(16)
        self._last_reset_killed = 0
        if state_fname is not None and (not os.path.isabs(state_fname)):
            state_fname = os.path.join(wd, "nes_env", "res", state_fname)
        self.state_fname = state_fname
        self.continue_play = continue_play
        with open(reward_scheme_file_full, 'r') as f:
            self.reward_scheme = json.load(f)
            # save reward files in checkpoints!


    def report(self, eff):
        _livestat = eff[LIVE_STATUS_IDX]
        _playstat = eff[PLAY_STATUS_IDX]
        print "Score {}, P1 {}, Play Status {}".format(
            eff[SCORE_IDX], LIVE_STATUS_MEANING[_livestat],
            PLAY_STATUS_MEANING[_playstat])


    def reset(self):
        self.game.reset_game()
        eff = []
        if self.state_fname is None:
            self._act(ACT_NOOP, 100)
            self._act(ACT_START)
            self._act(ACT_NOOP)
            eff = self._act(ACT_START)
            self.skip_non_playable(eff)
            eff = self._act(ACT_RIGHT, 80)
        else:
            #print "Load state {}".format(self.state_fname)
            self.game.loadState(self.state_fname)
            eff = self._act(ACT_NOOP)
        self._score = eff[0]
        self._live_status = eff[2]
        self._boss_killed = (eff[4] == 8 or eff[4] == 9)
        self._play_status = eff[4]
        self._last_reset_killed = self.total_killed
        for i_, addr in enumerate(range(0x78, 0x88)):
            self.enemy_hp[i_] = self.game.cheatGetByte(addr)
        return self._getframe()

    def _getframe(self):
        return self.game.getScreenRGB(self._screen_1, self._screen_rgb)

    def frame(self):
        return self._getframe()

    def _act(self, a, rep=1, delay=0):
        """
        Emulator runs 32 times the actual game speed, so
        we press the button and skip 31 frames
        :return:

        effects of the act:
        - score
        - P1 lives
		- P1 status 0 dead 1 alive 2 dying
		- Stage 00-07
		- Screen type 04 - normal game play; 08/09 - boss defeat
        """
        eff = None
        for i in range(rep):
            eff = self.game.act(a)
            if delay > 0:
                time.sleep(delay/1000.0)
        return eff


    def step(self, a, skip=2):
        frames = np.zeros( (skip,) + self._screen_rgb.shape, dtype=np.uint8)
        eff = self._act(a)
        frames[0, ...] = self._getframe()
        for i in range(skip-1):
            eff = self._act(SECOND_ACTS[a])
            frames[i+1, ...] = self._getframe()

        # compose next state: taking max value of RGB
        next_s = np.max(frames.reshape(skip, -1), axis=0).reshape(self._screen_rgb.shape)

        # determine effects and reward
        # maintain score
        delta_score = eff[0] - self._score
        self._score = eff[0]

        # if player being killed
        killed = 0
        if eff[2] != 1: # now dead or dying
            if self._live_status == 1:
                killed = 1
        self._live_status = eff[2]
        if killed:
            self.total_killed += 1
            print "killed {}".format(self.total_killed)
            if self.continue_play:
                self.game.cheatSetLives99()

        # if kill boss
        kill_boss = 0
        if eff[4] == 8 or eff[4] == 9:
            if not self._boss_killed:
                kill_boss = 1 # get reward once
            self._boss_killed = True
        self._play_status = eff[4]

        stage = eff[3]
        a_right = a in [ ACT_RIGHT, ACT_A_RIGHT, ACT_B_RIGHT,
                       ACT_B_UP_RIGHT, ACT_B_DOWN_RIGHT, ]
        a_left = a in [ ACT_LEFT, ACT_A_LEFT, ACT_B_LEFT,
                         ACT_B_UP_LEFT, ACT_B_DOWN_LEFT, ]
        a_jump = a in [ACT_A, ACT_A_UP, ACT_A_RIGHT, ACT_A_LEFT, ACT_A_DOWN ]
        a_up = a in [ ACT_UP, ACT_A_UP, ACT_B_UP, ACT_B_UP_RIGHT, ACT_B_UP_LEFT]
        progress = 0
        if stage==1 or stage==3:
            progress = int(a_up)
        else:
            progress = int(a_right) - int(a_left)
        #  ( (stage == 2 or stage == 5) and a_jump ) or \

        rew = delta_score * self.reward_scheme['score_gain'] \
              + killed * self.reward_scheme['killed'] \
              + kill_boss * self.reward_scheme['kill_boss'] \
              + progress * self.reward_scheme['progress'] \
              + self.reward_scheme['time'] # time

        if self._boss_killed or stage!=0:
            self.total_win += 1
            print "win {}".format(self.total_win)
            self.reset()
            self.skip_non_playable(eff) # play one level at a time

        if killed:
            if (not self.continue_play) or \
               (self.total_killed - self._last_reset_killed > 3):
                print "game over"
                self.reset()
            self.skip_non_playable(eff)

        return next_s, rew, (killed or kill_boss), None


    def skip_non_playable(self, eff, max_skip=1000):
        """
        Skip some non-playable scenes in the game.
        :param eff:
        :param max_skip:
        :return:
        """
        skipped = False
        for t in range(max_skip):
            live_status = eff[LIVE_STATUS_IDX]
            play_status = eff[PLAY_STATUS_IDX]
            normal_play = live_status == 1 and play_status == 4
            if normal_play:
                break
            skipped = True
            self._act(ACT_NOOP)
            eff = self._act(ACT_NOOP) # so we always skip even number of frames
            # print eff
        if skipped:
            self._act(ACT_NOOP) # first playable frame after dying - player is invisible


    def save_state(self, fname):
        self.game.saveState(fname)

    def load_state(self, fname):
        self.state_fname = fname
        self.reset()

    def begin_movie(self, fname):
        self.game.beginMovie(fname)

    def end_movie(self):
        self.game.endMovie()

class ContraWrapper(object):
    def __init__(self, env=None):
        """

        :param env:
        :type env: EnvContra
        """
        self._env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def reset(self):
        return self._env.reset()

    def step(self, act):
        return self._env.step(act)

    def frame(self):
        return self._env.frame()

    def save_state(self, fname):
        self._env.save_state(fname)

    def load_state(self, fname):
        self._env.load_state(fname)

    def begin_movie(self, fname):
        self._env.begin_movie(fname)

    def end_movie(self):
        self._env.end_movie()

def _maxpool2(s):
    """
    :param s:
    :type s: np.ndarray
    :return:
    """
    h = s.shape[0]
    w = s.shape[1]
    assert h % 2 == 0 and w % 2 == 0, \
        "width and height must be even"

    if s.ndim == 3 and s.shape[2] == 3 : # RGB
        # Max is done in low-level frame skip
        s0 = np.transpose(s, (2, 0, 1)) # rgb x h x w
        new_s = np.maximum(s0[:, ::2, ::2], s0[:, 1::2, ::2])
        new_s = np.maximum(new_s, s0[:, 1::2, 1::2])
        new_s = np.maximum(new_s, s0[:, ::2, 1::2])
        new_s = new_s.transpose((1, 2, 0))

        # new_s = s[::2, ::2, :]
    elif s.ndim == 1: # gray
        new_s = np.maximum(s[::2, ::2], s[1::2, ::2])
        new_s = np.maximum(new_s, s[1::2, 1::2])
        new_s = np.maximum(new_s, s[::2, 1::2])
    return new_s

class MaxPool2ContraWrapper(ContraWrapper):
    def __init__(self, env):
        super(MaxPool2ContraWrapper, self).__init__(env)
        h, w = self.observation_space.shape[:2]
        assert h % 2 == 0 and w % 2 == 0, \
            "width and height must be even"
        self.observation_space.shape[0] = h // 2
        self.observation_space.shape[1] = w // 2


    def reset(self):
        return _maxpool2(self._env.reset())

    def step(self, act):
        s, rew, done, msg = self._env.step(act)
        s1 = _maxpool2(s)
        return s1, rew, done, msg


# skip in low-level act proc
class RepeatActionContraWrapper(ContraWrapper):
    def __init__(self, env, rep=2):
        super(RepeatActionContraWrapper, self).__init__(env)
        self._rep = rep

    def reset(self):
        return self._env.reset()

    def step(self, act):
        total_rew = 0
        for i in range(self._rep):
            s, rew, done, msg = self._env.step(act)
            total_rew += rew
            if done:
                break
        return s, total_rew, done, msg


def get_contra_env(state_fname=None, continue_play=False,
                   reward_scheme_file="Contra_reward_scheme_gameplay0.json"):
    game = EnvContra(state_fname, continue_play, reward_scheme_file)
    game = MaxPool2ContraWrapper(game)
    game = RepeatActionContraWrapper(game, 2)
    return game