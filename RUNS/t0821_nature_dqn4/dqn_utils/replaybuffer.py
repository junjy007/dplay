"""
Replay Buffer manages transition history in memory.
"""
import numpy as np


class ReplayBuffer(object):
    def __init__(self, size, frame_history_len):
        self.size = size
        self.frame_history_len = frame_history_len
        self.next_idx = 0
        self.num_in_buffer = 0
        self.obs = None
        self.action = None
        self.reward = None
        self.done = None
        self.rng = np.random.RandomState(42)


    def can_sample(self, batch_size):
        """
        if there are enough samples in the memory to sample from
        not sure this is useful -- generally you would need much more
        samples to start the sampling

        :param batch_size:
        :return:
        """
        return batch_size < self.num_in_buffer


    def _encode_sample(self, indexes):
        # take some (consequtive) frames, "encode" each of them,
        # and make a mini-batch
        obs_ = [self._encode_observation(idx)[np.newaxis, :] for idx in indexes]
        obs_batch = np.concatenate(obs_, axis=0)
        act_batch = self.action[indexes]
        rew_batch = self.reward[indexes]
        next_obs_ = [self._encode_observation(idx+1)[np.newaxis, :] for idx in indexes]
        next_obs_batch = np.concatenate(next_obs_, axis=0)
        done_mask = np.array([1.0 if self.done[idx] else 0.0 for idx in indexes], dtype=np.float32)
        return obs_batch, act_batch, rew_batch, next_obs_batch, done_mask


    def sample(self, batch_size):
        """
        Sample batch_size different transitions.

        obs_batch[i], act_batch[i], ==> rew_batch[i], next_obs_batch[i], done_mask[i]

        if done_mask[i]==1: next_obs_batch[i] is undefined

        NOTE returns numpy arrays, this object does NOT handle neural network learning
        framework containers (so the output can be converted to PyTorch or Tensorflow
        or whatever). The pixel values are in 8-bit unsigned integers, to save memory
        transfer cost between main memory and GPU

        :param batch_size:
        :return:
          obs_batch: np.ndarray
            Observation minibatch
            Array with shape (batch_size, channels, height, width), where
            channels = image_colour_channels * frame_history_len; dtype=np.uint8

          act_batch: np.ndarray
            Actions
            Array of shape (batch_size,) dtype=np.int32
          rew_batch: np.ndarray
            Rewards
            Array of shape (batch_size,) dtype=np.float32
          next_obs_batch: np.ndarray
            Next observation minibatch, same shape and type as $obs_batch
        """

        assert self.can_sample(batch_size), "Too few samples"
        indexes = self.rng.choice(self.num_in_buffer-1, size=(batch_size,), replace=False)
        return self._encode_sample(indexes)


    def _encode_observation(self, idx):
        # taking $frame_history_len consecutive frames, where the latest one is specified by idx
        end_i = idx + 1
        start_i = end_i - self.frame_history_len

        if start_i < 0 and self.num_in_buffer != self.size:
            # the buffer is not full, so self.obs[ -1 % self.size ] would be meaningless
            start_i = 0

        new_start_i = start_i
        for i in range(start_i, end_i-1):
            # if there is some episode termination in the latest $frame_history_len frames,
            # we should NOT take the frames from previous episodes
            if self.done[i % self.size]:
                new_start_i = i + 1
        start_i = new_start_i
        missing_context = self.frame_history_len - (end_i - start_i)
        if start_i < 0 or missing_context > 0:
            frames = [np.zeros_like(self.obs[0]) for _ in range(missing_context)]
            for idx in range(start_i, end_i):
                frames.append(self.obs[idx % self.size])
            return np.concatenate(frames, 0)
        else:
            # this optimization has potential to saves about 30% compute time \o/
            img_h, img_w = self.obs.shape[2], self.obs.shape[3]
            return self.obs[start_i:end_i].reshape(-1, img_h, img_w)


    def store_frame(self, frame):
        """
        Store a single frame in the buffer at the next available index, overwriting
        old frames if necessary.


        Parameters
        :param frame:
            Array of shape (img_h, img_w, colour_channels) and dtype np.uint8
            and the frame will transpose to shape (colour_channels, img_h, img_w) to be stored

        :type frame: np.ndarray
        Returns
        -------
        idx: int
            Index at which the frame is stored. To be used for `store_effect` later.
        """
        frame = frame.transpose(2, 0, 1)

        if self.obs is None:
            frame_buffer_shape = [self.size,] + list(frame.shape)
            self.obs = np.empty(frame_buffer_shape, dtype=np.uint8)
            self.action = np.empty([self.size], dtype=np.int32)
            self.reward = np.empty([self.size], dtype=np.float32)
            self.done = np.empty([self.size], dtype=np.bool)

        ret = self.next_idx
        self.obs[self.next_idx] = frame
        self.next_idx = (self.next_idx + 1) % self.size
        self.num_in_buffer = min(self.size, self.num_in_buffer + 1)

        return ret

    def store_effect(self, idx, action, reward, done):
        """
        :param idx:
            Index in buffer of recently observed frame (returned by `store_frame`).
        :param action:
        :param reward:
        :param done:
        :return:
        """
        self.action[idx] = action
        self.reward[idx] = reward
        self.done[idx] = done

    def encode_recent_observation(self):
        assert self.num_in_buffer>0
        return self._encode_observation((self.next_idx-1) % self.size)
































