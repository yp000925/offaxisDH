import sys
import os
os.chdir("/Users/zhangyunping/PycharmProjects/offaxisDH/python")
from helpers import Hologram
path = "../Data/offaxis/bg_both.bmp"
h = Hologram.from_tif(path, wavelength=650e-6, dx=3.45e-6, dy=3.45e-6)
propagation_distance = 0.155
wave = h.reconstruct(propagation_distance)