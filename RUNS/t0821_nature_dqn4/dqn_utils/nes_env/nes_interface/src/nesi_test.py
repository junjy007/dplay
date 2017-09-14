__all__ = ['NESInterface']

from ctypes import *
import numpy as np
from numpy.ctypeslib import as_ctypes
import os


PROJ_DIR = os.path.abspath( os.path.join(
    os.path.dirname(__file__), '..' ))
nes_lib = cdll.LoadLibrary(os.path.join(PROJ_DIR, 'build', 'libnesi.so'))

class NESInterface(object):
    def __init__(self):
        rom_fname = os.path.join(PROJ_DIR, 'res', 'Contra.nes')
        rom_id = 1 # Contra ID
        nes_lib.NESInterface.argtypes = [c_char_p, c_int]
        nes_lib.NESInterface.restype = c_void_p
        byte_string_rom = rom_fname.encode('utf-8')
        self.obj = nes_lib.NESInterface(byte_string_rom, rom_id)
        self.width, self.height = self.getScreenDims()

    def act(self, action):
        nes_lib.act.argtypes = [c_void_p, c_int, c_void_p]
        nes_lib.act.restype = c_int
        effects = np.zeros(shape=(32,), dtype=c_int)
        rew = nes_lib.act(self.obj, int(action), as_ctypes(effects))
        return effects

    def reset_game(self):
        nes_lib.resetGame.argtypes = [c_void_p]
        nes_lib.resetGame.restype = None
        nes_lib.resetGame(self.obj)

    def cheatSetLives99(self):
        nes_lib.cheatSetLives99.argtypes = [c_void_p]
        nes_lib.cheatSetLives99.restype = None
        nes_lib.cheatSetLives99(self.obj)
        return

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
        return nes_lib.loadState(self.obj, byte_string_state_fname)

    def beginAVI(self, avi_fname):
        """begin recording avi"""
        byte_string_state_fname = avi_fname.encode('utf-8')
        nes_lib.beginAVI.argtypes = [c_void_p, c_char_p]
        nes_lib.beginAVI.restype = c_int
        return nes_lib.beginAVI(self.obj, byte_string_state_fname)

    def endAVI(self):
        """stop and save AVI movie"""
        nes_lib.endAVI.argtypes = [c_void_p]
        nes_lib.endAVI.restype = None
        nes_lib.endAVI(self.obj)



if __name__ == '__main__':
    pass