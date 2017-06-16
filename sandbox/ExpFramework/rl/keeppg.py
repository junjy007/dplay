"""
Policy Grad Keepers
"""
import os
import json
import numpy as np
from collections import deque
import time


# TODO: put in local utilities.
def running_val(running_v, v):
    a = 0.99
    return v if running_v is None \
        else running_v * a + v * (1.0 - a)


class Keeper:
    """
    Keeper helps administrate learning:
    ** Methods
    - save              and
    - load              handle checkpoints in the training sessions
    - record_env_step   and
    - record_train_step keeps records of training progress
    - report_step       prints recent progress
    ** Flags
    - need_train        means: according to the learning strategy,
      enough experience has been accquired, a training step is ready
      to run
    - need_save         enough training steps done, to save checkpoint
    - need_draw         to show the game in one episode
    """

    def __init__(self, objs, opts):
        """
        :param objs: objects that can be saved and reloaded in training sessions.
        :param opts: e.g. {
            'train_every_n_episodes': 1,
            'save_every_n_training_steps': 10,  # TODO: set to resonably large
            'draw_every_n_training_steps': -1,
            'save_path': 'SAVE',
            'report': {
                'save_checkpoint': True,
                'every_n_steps': 1,
                'every_n_training': 1,
                'every_n_episodes': 1,
                'every_n_time_records': 100}
            }

        """
        self.opts = opts
        self.objects = objs
        self.object_filenames = [o_.__class__.__name__ for o_ in self.objects]
        for i in range(len(self.object_filenames)):
            while self.object_filenames[i] in self.object_filenames[:i]:
                self.object_filenames[i] += '-copy'

        self.savepath = opts['save_path']
        self.checkpoint_status_fullfname = os.path.join(self.savepath, 'latest.json')
        self.records = {
            'total_steps': 0,  # step-wise info
            'reward_history': [],
            'running_reward': None,
            'episodes': 0,  # episode-wise info
            'episode_length_history': [],
            'episode_reward_history': [],
            'running_episode_reward': None,
            'training_steps': 0,  # mini-batch-wise info
            'loss_history': [],  # training stuff goes here
            'running_loss': None,
            'checkpoint_history': [],  # checkpoints
            'latest_checkpoint': ''
        }
        self.last_term_t = 0
        self.need_train = False
        self.need_save = False
        self.need_draw = False  # NOTE this flag is reset by renderer
        self.need_stop = False

        self.report_opts = self.opts['report']
        self.last_reported_episode = -1
        self.last_reported_training = -1
        self.last_reported_time_cost = -1

        self.timer = None
        self.timers = {k_: deque([], 100) for k_ in
                       ['record_env_step', 'record_train_step',
                        'report_step', 'policy.get_action', 'env.step',
                        'mem.add_experience', 'trainer.step']}

    def save(self):
        checkpoint_path = os.path.join(self.savepath,
                                       'checkpoint-{:d}'.format(self.records['training_steps']))
        if not os.path.exists(checkpoint_path):
            os.mkdir(checkpoint_path)
        self.records['checkpoint_history'].append(checkpoint_path)

        obj_full_fnames = [os.path.join(checkpoint_path, f_)
                           for f_ in self.object_filenames]
        for o, of in zip(self.objects, obj_full_fnames):
            o.save(of)
            # print "Save an object {} to {}".format(o.__class__.__name__, of)

        self.records['latest_checkpoint'] = checkpoint_path
        with open(self.checkpoint_status_fullfname, 'w') as f:
            json.dump(self.records, f, indent=2)

        self.need_save = False
        if self.report_opts['save_checkpoint']:
            print 'Checkpoint {} saved to {}'.format(
                len(self.records['checkpoint_history']),
                self.records['checkpoint_history']
            )

    def load(self):
        if not os.path.exists(self.checkpoint_status_fullfname):
            return

        with open(self.checkpoint_status_fullfname, 'r') as f:
            self.records = json.load(f)

        obj_full_fnames = [os.path.join(self.records['latest_checkpoint'], f_)
                           for f_ in self.object_filenames]

        for o, of in zip(self.objects, obj_full_fnames):
            o.load(of)
            # print "Load an object {} from {}".format(o.__class__.__name__, of)

    def set_timer(self):
        self.timer = time.time()

    def record_time(self, fn):
        t = time.time() - self.timer
        self.timers[fn].append(t)
        self.set_timer()

    def record_env_step(self, reward, term):
        r = float(reward)
        rec = self.records
        rec['reward_history'].append(r)
        rec['running_reward'] = running_val(rec['running_reward'], r)
        rec['total_steps'] += 1
        if term:
            # --------
            # NOT save full reward history in JSON. Clear for now
            # ep_r = np.sum(rec['reward_history'][self.last_term_t:])
            ep_r = np.sum(rec['reward_history'])
            rec['reward_history'] = []
            # --------
            rec['episode_reward_history'].append(ep_r)
            rec['running_episode_reward'] = running_val(rec['running_episode_reward'], ep_r)
            rec['episode_length_history'].append(rec['total_steps'] - self.last_term_t)
            self.last_term_t = rec['total_steps']
            rec['episodes'] += 1
            if rec['episodes'] % self.opts['train_every_n_episodes'] == 0:
                self.need_train = True  # reset when recording a training step
            self.need_draw = False

    def record_train_step(self, loss):
        """
        Record training loss. This is separate from recording environment rewards because
        training step happens every few steps.
        """
        loss = float(loss)
        self.need_train = False
        rec = self.records
        rec['loss_history'].append(loss)
        rec['running_loss'] = running_val(rec['running_loss'], loss)
        rec['training_steps'] += 1
        k_ = self.opts['save_every_n_training_steps']
        if k_ > 0 and rec['training_steps'] % k_ == 0:
            self.need_save = True
        k_ = self.opts['draw_every_n_training_steps']
        if k_ > 0 and rec['training_steps'] % k_ == 0:
            self.need_draw = True  # reset when an episode ends
        if rec['training_steps'] >= self.opts['max_training_steps']:
            self.need_stop = True  # TODO: use more elegant conditions

    def report_step(self):
        did_rep = False
        # agent-environment interaction steps -- fastest changing
        n = self.report_opts['every_n_steps']
        N = self.records['total_steps']
        if n > 0 and N > 0 and N % n == 0:
            print "Total step {}, reward {:.3f}, running_reward {:.3f} ".format(
                N,
                self.records['reward_history'][-1],
                self.records['running_reward']),
            did_rep = True

        # every a few episodes -- the reward for episodes is what we are really concerned
        n = self.report_opts['every_n_episodes']
        N = self.records['episodes']
        if n > 0 and N > 0 and N % n == 0 and N != self.last_reported_episode:
            print "Episode {} steps {:3d} reward {:.1f} running episode reward {:.2f} ".format(
                N,
                self.records['episode_length_history'][-1],
                self.records['episode_reward_history'][-1],
                self.records['running_episode_reward']),
            self.last_reported_episode = N
            did_rep = True

        # training
        n = self.report_opts['every_n_training']
        N = self.records['training_steps']
        if n > 0 and N > 0 and N % n == 0 and N != self.last_reported_training:
            print "Training step {}, loss {:.3f}, running_loss {:.3f} ".format(
                N,
                self.records['loss_history'][-1],
                self.records['running_loss']
            ),
            self.last_reported_training = N
            did_rep = True

        # time cost
        n = self.report_opts['every_n_time_records']
        N = self.records['episodes']  # This will stop working for a different training procedure.
        if n > 0 and N > 0 and N % n == 0 and N != self.last_reported_time_cost:
            print '\nRecent {:4d}:'.format(n),
            for k_ in self.timers.keys():
                print '{}:{:.3f}'.format(k_, np.mean(self.timers[k_])),
            self.last_reported_time_cost = N
            did_rep = True

        if did_rep:
            print

        return
