# Experiment

FFor the evaluation of the results, we consider the Thorlabs LB1761 (a simple N-BK7 biconvex singlet lens) and perform a simulation using an f/8 aperture placed 2 mm  from the second lens surface. We map \[R1, T, R2, D2, OD\] (mm) as:

**R1 = 24.5**, **T = 9**, **R2 = −24.5**, **D2 = 22.2**, **OD = 3.175** (≈ f/8)

We conducted following experiment:

- Function test
- Layout & Ray visualization
- Best bocus estimation
- Sampling number (N) sweep
- Wavelength (lambda) sweep
- Through-focus (D2) sweep
- Aperture sweep (OD)

Futher, we consider a double Gaussian Lens  [US253251A](https://patents.google.com/patent/US2532751), and conducted following experiments

- wavelength sweep
- sampling number (N) sweep
- Through-focus (D2) sweep

## Function Test
Function test was conducted in the tests folder, named as [test_geo.py](../tests/test_geo.py) and [test_ray_tracing.py](../tests/test_ray_tracing.py)
## Lens layout
The lens could be viewed as 
![](../out/biconvex_layout.png)
## Ray tracing
The ray tracing could be viewed as
![](../out/biconvex_rays.png)

## PSF Exmples
After optimization, we found that placing the sensor at **20.3 mm** after the aperture yields the best focus.
![](../out/biconvex_psf_log.png)

## N-sweep (sampling/)
We sweeped N for {50, 100, 400,1600,3200,6400}, and calculated the **Metrics vs N**

|    N | ee50_mm | rms_radius_mm |
| ---: | :-----: | :-----------: |
|   50 | 0.05433 |    0.06684    |
|  100 | 0.05395 |    0.06596    |
|  400 | 0.05354 |    0.06503    |
| 1600 | 0.05406 |    0.06455    |
| 3200 | 0.05385 |    0.06442    |
| 6400 | 0.05383 |    0.06432    |
The convergence begins around N ≥ 1600, where the RMS and EE50 values stabilize to ~0.064 mm and ~0.0538 mm, respectively — indicating sufficient ray sampling density for accurate PSF estimation.

### Wavelength Sweep

We sweep wavelength for 


### Through-focus (D2 sweep)

We sweep D2 from 

### Aperture Sweep (OD)

We sweep OD:  1.5875 mm (f/16), 3.175 mm (f/8), 6.35 mm (f/4),12.7 mm (f/2)

### Off-axis sweep