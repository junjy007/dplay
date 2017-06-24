"""
This module contains preprocessors converting raw states to objects accepted by experience managers
"""
import numpy as np


# noinspection PyPep8Naming
class ImagePreprocessor(object):
    """
    The raw observations are images. Preprocessing will result in output images.
    The preprocessing result will be a [Channels x Height x Width] image-array
    """

    def __init__(self):
        self.in_im_width = 0
        self.in_im_height = 0
        self.in_im_channels = 0
        self.im_width = 0
        self.im_height = 0

    def process(self, I):
        """
        :param I: [Rows x Columns x Channels] image
        :type I: np.ndarray
        :return: 
        """
        assert I.shape[0] == self.in_im_height and I.shape[1] == self.in_im_width
        if I.ndim == 3:
            assert I.shape[2] == self.in_im_channels
        return None


# noinspection PyPep8Naming
class GymAtariFramePreprocessor(ImagePreprocessor):
    def __init__(self):
        super(GymAtariFramePreprocessor, self).__init__()
        self.in_im_width = 160
        self.in_im_height = 210
        self.in_im_channels = 3

    def process(self, I):
        return super(GymAtariFramePreprocessor, self).process(I)


# noinspection PyPep8Naming,PyPep8Naming
class GymAtariFramePreprocessor_Stacker(GymAtariFramePreprocessor):
    """
    Raw pixel to numpy array as a "single state observation", which will be
    dealt with by experience memory.

    This object will compute consecutive frames.
    """

    def __init__(self, stack_frames=3):
        """
        :param stack_frames: #.frames making one state variable
        """
        super(GymAtariFramePreprocessor_Stacker, self).__init__()
        self.channels = stack_frames
        self.im_width = 80
        self.im_height = 80
        self.frames = np.zeros((self.channels, self.im_height, self.im_width), dtype=np.float32)
        self.is_first_frame = True

    def __call__(self, I):
        return self.process(I)

    def reset(self):
        self.is_first_frame = True

    # noinspection PyPep8Naming,PyMethodMayBeStatic
    def process(self, I):
        """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
        super(GymAtariFramePreprocessor_Stacker, self).process(I)
        I = I[35:195]  # crop
        I = I[::2, ::2, 0]  # downsample by factor of 2
        I[I == 144] = 0  # erase background (background type 1)
        I[I == 109] = 0  # erase background (background type 2)
        I[I != 0] = 1  # everything else (paddles, ball) just set to 1
        I = np.float32(I)
        if self.is_first_frame:
            for c in range(self.channels):
                self.frames[c, ...] = I
            self.is_first_frame = False
        else:
            self.frames[:-1, ...] = self.frames[1:, ...]
            self.frames[-1, ...] = I
        return self.frames


# noinspection PyPep8Naming
class GymAtariFramePreprocessor_Diff(GymAtariFramePreprocessor):
    """
    Raw pixel to numpy array as a "single state observation", which will be
    dealt with by experience memory.

    This object will compare two frames, take the difference
    """

    def __init__(self):
        super(GymAtariFramePreprocessor_Diff, self).__init__()
        self.channels = 1
        self.im_width = 80
        self.im_height = 80
        self.frames = np.zeros((self.channels, self.im_height, self.im_width), dtype=np.float32)
        self.is_first_frame = True
        self.prev_frame = None

    def __call__(self, I):
        return self.process(I)

    def reset(self):
        self.is_first_frame = True
        self.prev_frame = None

    # noinspection PyPep8Naming,PyMethodMayBeStatic
    def process(self, I):
        """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
        super(GymAtariFramePreprocessor_Diff, self).process(I)
        I = I[35:195]  # crop
        I = I[::2, ::2, 0]  # downsample by factor of 2
        I[I == 144] = 0  # erase background (background type 1)
        I[I == 109] = 0  # erase background (background type 2)
        I[I != 0] = 1  # everything else (paddles, ball) just set to 1
        I = np.float32(I)
        if self.is_first_frame:
            self.frames[...] = 0
            self.is_first_frame = False
        else:
            self.frames[...] = I - self.prev_frame
        self.prev_frame = I
        return self.frames
