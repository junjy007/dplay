# nes_python_interface.py
# Author: Ben Goodrich, Ehren J. Brav
# This partially implements a python version of the arcade learning
# environment interface.
__all__ = ['NESInterface']

from ctypes import *
import numpy as np
from numpy.ctypeslib import as_ctypes
import os

_apath = os.path.abspath(os.path.dirname(__file__)) 
#nes_lib = cdll.LoadLibrary(os.path.join(_apath, 'libfceux.so'))
#nes_lib = cdll.LoadLibrary(os.path.join(_apath, 'libfceux_plus_noskip.so'))
nes_lib = cdll.LoadLibrary(os.path.join(_apath, 'nes_interface/build/libnesi.so'))

class NESInterface(object):
    def __init__(self, rom, id):
        nes_lib.NESInterface.argtypes = [c_char_p, c_int]
        nes_lib.NESInterface.restype = c_void_p
        byte_string_rom = rom.encode('utf-8')
        self.obj = nes_lib.NESInterface(byte_string_rom, int(id))
        self.width, self.height = self.getScreenDims()

    # def getString(self, key):
    #     return nes_lib.getString(self.obj, key)
    # def getInt(self, key):
    #     return nes_lib.getInt(self.obj, key)
    # def getBool(self, key):
    #     return nes_lib.getBool(self.obj, key)
    # def getFloat(self, key):
    #     return nes_lib.getFloat(self.obj, key)

    # def setString(self, key, value):
    #   nes_lib.setString(self.obj, key, value)
    # def setInt(self, key, value):
    #   nes_lib.setInt(self.obj, key, value)
    # def setBool(self, key, value):
    #   nes_lib.setBool(self.obj, key, value)
    # def setFloat(self, key, value):
    #   nes_lib.setFloat(self.obj, key, value)

    # def loadROM(self, rom_file):
    #     nes_lib.loadROM(self.obj, rom_file)

    def act0(self, action):
        nes_lib.act.argtypes = [c_void_p, c_int]
        nes_lib.act.restype = c_int
        return nes_lib.act(self.obj, int(action))

    def act(self, action):
        nes_lib.act.argtypes = [c_void_p, c_int, c_void_p]
        nes_lib.act.restype = c_int
        effects = np.zeros(shape=(32,), dtype=c_int)
        rew = nes_lib.act(self.obj, int(action), as_ctypes(effects))
        return effects

    def game_over(self):
        nes_lib.gameOver.argtypes = [c_void_p]
        nes_lib.gameOver.restype = c_bool
        return nes_lib.gameOver(self.obj)

    def reset_game(self):
        nes_lib.resetGame.argtypes = [c_void_p]
        nes_lib.resetGame.restype = None
        nes_lib.resetGame(self.obj)

    def getLegalActionSet(self):
        nes_lib.getNumLegalActions.argtypes = [c_void_p]
        nes_lib.getNumLegalActions.restype = c_int
        act_size = nes_lib.getNumLegalActions(self.obj)
        act = np.zeros(shape=(act_size,), dtype=c_int)
        nes_lib.getLegalActionSet.argtypes = [c_void_p, c_void_p]
        nes_lib.getLegalActionSet.restype = None
        nes_lib.getLegalActionSet(self.obj, as_ctypes(act))
        return act

    def getMinimalActionSet(self):
        # For NES we assume this is the same
        # as the legal actions.
        return self.getLegalActionSet()

    def getFrameNumber(self):
        nes_lib.getFrameNumber.argtypes = [c_void_p]
        nes_lib.getFrameNumber.restype = c_int
        return nes_lib.getFrameNumber(self.obj)

    def lives(self):
        nes_lib.lives.argtypes = [c_void_p]
        nes_lib.lives.restype = c_int
        return nes_lib.lives(self.obj)

    def cheatSetLives99(self):
        nes_lib.cheatSetLives99.argtypes = [c_void_p]
        nes_lib.cheatSetLives99.restype = None
        nes_lib.cheatSetLives99(self.obj)
        return

    def cheatGetByte(self, addr):
        nes_lib.cheatGetByte.argtypes = [c_int]
        nes_lib.cheatGetByte.restype = c_ubyte
        return nes_lib.cheatGetByte(self.obj, int(addr))

    def cheatSetByte(self, addr, val):
        nes_lib.cheatSetByte.argtypes = [c_void_p, c_int, c_ubyte]
        nes_lib.cheatSetByte.restype = None
        # ubyte param will be passed in the function call using int,
        # nes_lib already knows its argument types
        nes_lib.cheatSetByte(self.obj, int(addr), int(val))
        return

    def getEpisodeFrameNumber(self):
        nes_lib.getEpisodeFrameNumber.argtypes = [c_void_p]
        nes_lib.getEpisodeFrameNumber.restype = c_int
        return nes_lib.getEpisodeFrameNumber(self.obj)

    def getScreenDims(self):
        """returns a tuple that contains (screen_width, screen_height)
        """
        nes_lib.getScreenHeight.argtypes = [c_void_p]
        nes_lib.getScreenHeight.restype = c_int
        nes_lib.getScreenWidth.argtypes = [c_void_p]
        nes_lib.getScreenWidth.restype = c_int
        width = nes_lib.getScreenWidth(self.obj)
        height = nes_lib.getScreenHeight(self.obj)
        return (width, height)

    def getScreen(self, screen_data=None):
        """This function fills screen_data with the RAW Pixel data
        screen_data MUST be a numpy array of uint8/int8. This could be initialized like so:
        screen_data = np.empty(w*h, dtype=np.uint8)
        Notice,  it must be width*height in size also
        If it is None,  then this function will initialize it
        Note: This is the raw pixel values,  before any RGB palette transformation takes place
        """
        if(screen_data is None):
            screen_data = np.zeros(self.width*self.height, dtype=np.uint8)
            
        nes_lib.getScreen.argtypes = [c_void_p, c_void_p, c_int]
        nes_lib.getScreen.restype = None
        nes_lib.getScreen(self.obj, as_ctypes(screen_data), c_int(screen_data.size))
        return screen_data

    def getScreenRGB(self, screen_data=None, rgb_screen=None):
        """This function fills screen_data with the data in RGB format
        screen_data MUST be a numpy array of uint8. This can be initialized like so:
        screen_data = np.empty((height,width,3), dtype=np.uint8)
        If it is None,  then this function will initialize it.
        """
        if(screen_data is None):
            screen_data = np.empty((self.height, self.width, 1), dtype=np.uint8)
        nes_lib.getScreen.argtypes = [c_void_p, c_void_p, c_int]
        nes_lib.getScreen.restype = None

        # First get the raw screen.
        nes_lib.getScreen(self.obj, as_ctypes(screen_data), c_int(screen_data.size))

        # Now convert to RGB.
        if rgb_screen is None:
            rgb_screen = np.empty((self.height, self.width, 3), dtype=np.uint8)
        nes_lib.fillRGBfromPalette.argtypes = [c_void_p, c_void_p, c_void_p, c_int]
        nes_lib.fillRGBfromPalette.restype = None
        nes_lib.fillRGBfromPalette(self.obj, as_ctypes(screen_data), as_ctypes(rgb_screen), c_int(screen_data.size))
        return rgb_screen

    def getScreenGrayscale(self, screen_data=None):
        """This function fills screen_data with the data in grayscnes
        screen_data MUST be a numpy array of uint8. This can be initialized like so:
        screen_data = np.empty((height,width,1), dtype=np.uint8)
        If it is None,  then this function will initialize it.
        """
        if(screen_data is None):
            screen_data = np.empty((self.height, self.width, 1), dtype=np.uint8)
        nes_lib.getScreen.argtypes = [c_void_p, c_void_p, c_int]
        nes_lib.getScreen.restype = None
        nes_lib.getScreen(self.obj, as_ctypes(screen_data[:]), c_int(screen_data.size))
        return screen_data

    def getRAMSize(self):
        return nes_lib.getRAMSize(self.obj)

    def getRAM(self, ram=None):
        """This function grabs the RAM.
        ram MUST be a numpy array of uint8/int8. This can be initialized like so:
        ram = np.array(ram_size, dtype=uint8)
        Notice: It must be ram_size where ram_size can be retrieved via the getRAMSize function.
        If it is None,  then this function will initialize it.
        """
        if(ram is None):
            ram_size = nes_lib.getRAMSize(self.obj)
            ram = np.zeros(ram_size, dtype=np.uint8)
        nes_lib.getRAM(self.obj, as_ctypes(ram))
        return ram

    def saveScreenPNG(self, filename):
        """Save the current screen as a png file"""
        return nes_lib.saveScreenPNG(self.obj, filename)

    def saveState(self, state_fname):
        """Saves the state of the system"""
        byte_string_state_fname = state_fname.encode('utf-8')
        nes_lib.saveState.argtypes = [c_void_p, c_char_p]
        nes_lib.saveState.restype = None
        return nes_lib.saveState(self.obj, byte_string_state_fname)

    def loadState(self, state_fname):
        """Loads the state of the system"""
        byte_string_state_fname = state_fname.encode('utf-8')
        nes_lib.loadState.argtypes = [c_void_p, c_char_p]
        nes_lib.loadState.restype = c_bool
        return nes_lib.loadState(self.obj, state_fname)

    def cloneState(self):
        """This makes a copy of the environment state. This copy does *not*
        include pseudorandomness, making it suitable for planning
        purposes. By contrast, see cloneSystemState.
        """
        return nes_lib.cloneState(self.obj)

    def restoreState(self, state):
        """Reverse operation of cloneState(). This does not restore
        pseudorandomness, so that repeated calls to restoreState() in
        the stochastic controls setting will not lead to the same
        outcomes.  By contrast, see restoreSystemState.
        """
        nes_lib.restoreState(self.obj, state)

    def cloneSystemState(self):
        """This makes a copy of the system & environment state, suitable for
        serialization. This includes pseudorandomness and so is *not*
        suitable for planning purposes.
        """
        return nes_lib.cloneSystemState(self.obj)

    def restoreSystemState(self, state):
        """Reverse operation of cloneSystemState."""
        nes_lib.restoreSystemState(self.obj, state)

    def deleteState(self, state):
        """ Deallocates the NESState """
        nes_lib.deleteState(state)

    def encodeStateLen(self, state):
        return nes_lib.encodeStateLen(state)

    def encodeState(self, state, buf=None):
        if buf == None:
            length = nes_lib.encodeStateLen(state)
            buf = np.zeros(length, dtype=np.uint8)
        nes_lib.encodeState(state, as_ctypes(buf), c_int(len(buf)))
        return buf

    def decodeState(self, serialized):
        return nes_lib.decodeState(as_ctypes(serialized), len(serialized))

    def __del__(self):
        nes_lib.delete_NES.argtypes = [c_void_p]
        nes_lib.delete_NES.restype = None
        nes_lib.delete_NES(self.obj)
