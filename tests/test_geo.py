# This is a code to view and test the geometry
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.image as mpimg
import os,sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import mlrt 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# settings
M = 512
film_size = [M,M]
pixel_size = 0.00645 # 6.45 um


lens = mlrt.Lensgroup(device=device,pixel_size=pixel_size,film_size=film_size)
lens.load_file(Path('./lenses/LB1761.txt'))
assert 1e-4 <= lens.pixel_size <= 0.05, "pixel_size should be in mm (e.g., 0.006 for 6 µm)"

ray = lens.sample_ray_from_point(D_mm=30, wavelength=500, N=2000, filter_to_stop=True)
_, oss = lens.trace_to_sensor_r(ray, ignore_invalid=False)
ax = lens.plot_layout2d(show=False, fname="optic_geometry.png") # set show to false to save to image file 
lens.plot_raytraces_world(oss, ax=ax, show=False, fname="ray_geometry.png" )# set show to false to save to image file 
