"""
Policy Grad Keepers
"""
import os
import json
import numpy as np
from collections import deque
import time
import matplotlib.pyplot as plt


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
            'total_time': 0.0,
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
            'latest_checkpoint': '',
            'recent_episodes': 0.0,
            'recent_reward': 0.0,
            'recent_time_cost': 0.0,
            'recent_steps': 0.0,
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
        self.primary_step_timer = time.time()  # NOT for profiling different functions
        # or components. Recording total running time of training.

        self.timers = {k_: deque([], 100) for k_ in
                       ['record_env_step', 'record_train_step',
                        'report_step', 'policy.get_action', 'env.step',
                        'mem.add_experience', 'trainer.step']}

        # prepare for visual report NOTE: having nothing to do with need_draw,
        self.fig1 = plt.figure(figsize=(8, 6), dpi=100)
        self.fig2 = plt.figure(figsize=(8, 6), dpi=100)

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
                self.records['checkpoint_history'][-1]
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
        rec = self.records
        rec['total_steps'] += 1
        dt = time.time() - self.primary_step_timer
        self.records['total_time'] += dt
        self.records['recent_time_cost'] += dt
        self.records['recent_steps'] += 1
        self.primary_step_timer = time.time()
        self.records['recent_reward'] += reward
        if term:
            rec['episodes'] += 1
            rec['recent_episodes'] += 1
            if rec['episodes'] % self.opts['train_every_n_episodes'] == 0:
                self.need_train = True  # reset when recording a training step
            self.need_draw = False

    # noinspection PyTypeChecker
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

    # noinspection PyPep8Naming,PyTypeChecker
    def report_step(self):
        did_rep = False
        # agent-environment interaction steps -- fastest changing
        n = self.report_opts['every_n_steps']
        N = self.records['total_steps']
        if n > 0 and N > 0 and N % n == 0:
            en_ = float(self.records['recent_episodes'])
            self.records['reward_history'].append(self.records['recent_reward'] / en_)
            self.records['episode_length_history'].append(self.records['recent_steps'] / en_)
            print "Total steps {}, Total time {}, Training-steps {}, last {} steps: time {}, " \
                "avg-reward {:.3f}, avg-episode-length {:.3f}".format(
                    N, time.strftime("%H:%M:%S", time.gmtime(self.records['total_time'])),
                    self.records['training_steps'],
                    n, time.strftime("%S", time.gmtime(self.records['recent_time_cost'])),
                    self.records['reward_history'][-1],
                    self.records['episode_length_history'][-1]),
            self.plot_performance()
            self.records['recent_episodes'] = 0.0
            self.records['recent_time_cost'] = 0.0
            self.records['recent_reward'] = 0.0
            self.records['recent_steps'] = 0.0
            did_rep = True

        # every a few episodes -- the reward for episodes is what we are really concerned
        # n = self.report_opts['every_n_episodes']
        # N = self.records['episodes']
        # if n > 0 and N > 0 and N % n == 0 and N != self.last_reported_episode:
        #     print "Episode {} steps {:3d} reward {:.1f} running episode reward {:.2f} ".format(
        #         N,
        #         self.records['episode_length_history'][-1],
        #         self.records['episode_reward_history'][-1],
        #         self.records['running_episode_reward']),
        #     self.last_reported_episode = N
        #     did_rep = True

        # training
        # n = self.report_opts['every_n_training']
        # N = self.records['training_steps']
        # if n > 0 and N > 0 and N % n == 0 and N != self.last_reported_training:
        #     print "Training step {}, loss {:.3f}, running_loss {:.3f} ".format(
        #         N,
        #         self.records['loss_history'][-1],
        #         self.records['running_loss']
        #     ),
        #     self.last_reported_training = N
        #     did_rep = True

        # # time cost
        # n = self.report_opts['every_n_time_records']
        # N = self.records['episodes']  # This will stop working for a different training procedure.
        # if n > 0 and N > 0 and N % n == 0 and N != self.last_reported_time_cost:
        #     print '\nRecent {:4d}:'.format(n),
        #     for k_ in self.timers.keys():
        #         print '{}:{:.3f}'.format(k_, np.mean(self.timers[k_])),
        #     self.last_reported_time_cost = N
        #     did_rep = True

        if did_rep:
            print

        return

    def plot_performance(self, max_records=500):
        """
        NOT related to need_draw flag, which is about rendering the game frames on screen
        :return:
        """

        rs = np.asarray(self.records['reward_history'])
        ls = np.asarray(self.records['episode_length_history'])
        n = rs.shape[0]
        assert n == ls.shape[0]
        if n > max_records:
            idx = np.linspace(0, n - 1, max_records)
            idx_n = idx.astype(np.int)
            rs = rs[idx_n]
            ls = ls[idx_n]
        else:
            idx = np.arange(n)

        figs = [self.fig1, self.fig2]
        data = [rs, ls]
        titles = ['Average Episode Reward', 'Average Episode Length']
        fnames = [os.path.join(self.savepath, s + '.png') for s in ['rewards', 'epilen']]

        for fig, d, tl, fn in zip(figs, data, titles, fnames):
            fig.clf()
            plt.figure(fig.number)
            xx = (idx * self.report_opts['every_n_steps']).astype(np.int)
            plt.plot(xx, d, 'b-')
            if n > 20:
                plt.xticks(np.linspace(0, n-1, 10))
            plt.title(tl)
            plt.savefig(fn)


class RLAlgorithm:
    def __init__(self, keeper, env, preproc, mem, policy, trainer):
        """
        Reinforce Learning algorithm
        :param keeper:
        :param env:
        :param preproc:
        :param mem:
        :param policy:
        :param trainer:
        """
        self.keeper = keeper
        self.policy = policy
        self.env = env
        self.preproc = preproc
        self.mem = mem
        self.trainer = trainer

    def run(self):
        self.keeper.load()
        self.preproc.reset()

        state = self.preproc.process(self.env.reset())
        while not self.keeper.need_stop:
            action, action_prob = self.policy.get_action(state)
            next_state_raw, reward, is_terminal, _ = self.env.step(action)
            next_state = self.preproc(next_state_raw)
            episode_id = self.keeper.records['episodes']
            self.mem.add_experience(episode_id, state, action, reward, is_terminal, None)

            if is_terminal:
                self.preproc.reset()
                state = self.preproc.process(self.env.reset())
            else:
                state = next_state

            self.keeper.record_env_step(reward, is_terminal)

            if self.keeper.need_train:  # TODO train condition call back
                loss = self.trainer.step()
                self.keeper.record_train_step(loss)

            if self.keeper.need_save:
                self.keeper.save()

            if self.keeper.need_draw:
                self.env.render()

            self.keeper.report_step()
