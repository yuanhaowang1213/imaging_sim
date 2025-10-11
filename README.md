# imaging_sim
Angle-grid ray tracing from an on-axis point source to PSF on the sensor

## Goal
Simulate ray propagation from a single scene point through a simple biconvex lens to a discrete sensor plane, produce the PSF (energy distribution on sensor), and study how sampling, wavelength, field angle, focus, and aperture affect image quality. We use geometric optics (Snell), BK7 glass, and ignore diffraction and internal reflections.
## High level design

A high-level overview and referenced code are in[./doc/general.md](./doc/general.md)

## Code Explanation

The code was modified from [do: A differentiable engine for Deep Lens design of computational imaging systems](https://github.com/vccimaging/DiffOptics).

Main modifications include:
- Added **PSF metrics computation** (RMS radius, EE50 radius).  
- Corrected **rendering alignment** for even pixel sizes (eliminating half-pixel shift).  
- Added **sampling functions** based on object distance.  
- Implemented **build_lens()** for parameter tuning and multi-surface lens construction.  
- Conducted **unit tests** for individual components.  
- Simplified and modularized the codebase for clarity and maintainability.
- Others

### Environment
To install the dependency, please refer to [create_env.sh](./create_env.sh). The code mainly use pytorch, matplotlib, numpy, scipy, pytest. 

### Running the code 
Detailed explanations of each module can be found in [doc/codeexplanation.md](./doc/codeexplanation.md).  
The main entry point is described in [run.sh](./run.sh).
```bash
sh run.sh
```

## Experiment Setup, Results and Analysis

Detailed experiment setup and results are available in:
- [doc/experiment.md](./doc/experiment.md)  
- [doc/experiment.pdf](./doc/experiment.pdf)  
- Simplified presentation slides: [doc/lenssimulation.pdf](./doc/lenssimulation.pdf)





