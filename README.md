# imaging_sim
Angle-grid ray tracing from an on-axis point source to PSF on the sensor

## Goal
Simulate ray propagation from a single scene point through a simple biconvex lens to a discrete sensor plane, produce the PSF (energy distribution on sensor), and study how sampling, wavelength, field angle, focus, and aperture affect image quality. We use geometric optics (Snell), BK7 glass, and ignore diffraction and internal reflections.
## High level design

High level idea of the design and refered code could be found [here](./doc/general.md)

## Code Explaination

The code was modified from [do: A differentiable engine for Deep Lens design of computational imaging systems](https://github.com/vccimaging/DiffOptics). We added the PSF metrics function, corrected the rendering for even pixel size where there might be half pixel shifting, added sampling function from the object distance, enabled a build_lens function for paremeter tuning, tested the components in the code, simplified the code and etc.

### Environment
To install the dependency, please refer [here](./create_env.sh)

### Code running
The main code explaination is [here](./doc/codeexplanation.md)

## Experiment Setup, Results and Analysis

The detailed experiment could be found in [doc/experiment.md](./doc/experiment.md) and simplified slides could be found in [doc/lenssimulation.pdf](./doc/lenssimulation.pdf)




