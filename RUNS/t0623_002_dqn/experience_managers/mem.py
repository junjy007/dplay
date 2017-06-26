from collections import deque
from dplay_utils.tensordata import to_tensor_f32, to_tensor_int


class ExperienceMemory:
    """
    At a time step t, this memory manager expects (args to add_experience):
        S_{t-1}, A_{t-1}, R_t, Is_Terminal_{t}, Agent_Response_To(S_{t-1})
    i.e. the arguments (numbers) are
                t-1  | t
    - STATE:      1  |
    - ACTION:     2  |
    - REWARD:        | 3
    - TERMINAL:      | 4
    - RESP:         =5=>
    and
    - ADVANTAGE_{t}: = Reward_t + Reward_{t+1}*discount + ...
      will be filled when episode ends

    NB   S_t (the current state) will be added in the next step. The last state of an episode is
      NOT recorded (no action will be taken, after all!)
    NB-1 All info is not used in all learning algorithms
    NB-2 RESP will be saved to time step t. This is to align with the supervision information
         that will finally arrive to train the agent. E.g. in Q-learning, the agent would
         try to evaluate all actions given state S_{t-1}, the evaluation will be comapred
         to reward received at time {t}.
    """

    def __init__(self, capacity, discount):
        self.capacity = capacity
        self.discount = discount

        self.experience = {
            'states': deque(),
            'actions': deque(),
            'rewards': deque(),
            'advantages': deque(),
            'term_status': deque(),
            'prev_responses': deque(),
            'episode_id': deque()
        }
        self.first_step_in_episode = True

    def add_experience(self, episode_id, state, action, reward, is_terminal,
                       prev_resp=None, do_compute_advantage=True):
        """
        :param episode_id: do NOT record the episode ID in this object, which can cause inconsistence
          when save/load training sessions.
        :param state: See class doc, the last step state.
        :type state: np.ndarray
        :param action:
        :type action: int
        :param reward:
        :param is_terminal:
        :param prev_resp:
        :param do_compute_advantage: if set, I will compute discounted advantage (see
          class doc) when an episode ends.
        :return:
        """

        self.experience['states'].append(state)
        self.experience['actions'].append(action)
        self.experience['rewards'].append(reward)
        self.experience['term_status'].append(is_terminal)
        self.experience['advantages'].append(0)
        self.experience['prev_responses'].append(prev_resp)
        self.experience['episode_id'].append(episode_id)

        N = len(self.experience['states'])
        if N > self.capacity:
            for k_ in self.experience:
                self.experience[k_].popleft()
            N -= 1

        if is_terminal and do_compute_advantage:
            self.experience['advantages'][-1] = future_reward = reward
            this_episode = self.experience['episode_id'][-1]
            for t in range(-2, - N - 1, -1):
                if self.experience['episode_id'][t] != this_episode:
                    break  # have reached the previous episode
                self.experience['advantages'][t] = \
                    self.experience['rewards'][t] + future_reward * self.discount
                future_reward = self.experience['advantages'][t]

    def get_training_batch(self, num_steps=None, episodes=None):
        """
        Specific to the RL algorithm. This implementation is for Policy Gradient using a single episode
        :return: PyTorch Tensors:
          states: (N x C x H x W) image-states
          actions: N-integers
          advantages: N-floats
        """
        s_ = not (num_steps is None)
        e_ = not (episodes is None)
        assert (s_ and not e_) or (not s_ and e_), \
            "Either the number of steps or the episode id list must be given"

        # implement only using episode
        states = []
        actions = []
        advantages = []

        if s_:
            assert False, "Not implemented"

        if e_:
            for ep in episodes:
                ids = [i_ for i_, ei_ in enumerate(self.experience['episode_id'])
                       if ei_ == ep]

                for t in ids:  # no last action
                    states.append(self.experience['states'][t])
                    actions.append(self.experience['actions'][t])
                    advantages.append(self.experience['advantages'][t])

        return to_tensor_f32(states), to_tensor_int(actions), to_tensor_f32(advantages)
    
    def get_next_training_batch(self):
        """
        A short cut for taking the last one episode
        """
        return self.get_training_batch(
            episodes=[self.experience['episode_id'][-1], ])

    def save(self, fname):
        with open(fname, 'w') as f:
            f.write('Nothing to save')

    def load(self, fname):
        return

class TDMemory:
    """
    Memory for temporal difference algorithms, with new
    """