import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.image as mpimg
import sys
import mlrt 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# settings
M = 512
film_size = [M,M]
pixel_size = 0.00645 # 6.45 um


lens = mlrt.Lensgroup(device=device,pixel_size=pixel_size,film_size=film_size)
lens.load_file(Path('./lenses/LB1761.txt'))
assert 1e-4 <= lens.pixel_size <= 0.05, "pixel_size should be in mm (e.g., 0.006 for 6 µm)"
# lens.plot_layout2d()
lens.plot_setup2D()
# 
D = 1000.0 # mm
wav = 500 # nm
N_rays_debug = 60 * 60      # rays to check
N_rays_render = 120 * 120
# ray_dbg = lens.sample_ray_from_point(D_mm=D, wavelength=wav, N=N_rays_debug)

wavelengths = [500]  # R,G,B
imgs = []
for lam in wavelengths:
    ray = lens.sample_ray_from_point(
        D_mm=D, wavelength=lam, N=N_rays_render, theta_max=None, filter_to_stop=True
    )
    I = lens.render(ray)  # [M,M], accumulation
    if torch.max(I) > 0:
        I = I / torch.max(I)   # normalization
    imgs.append(I.cpu())

