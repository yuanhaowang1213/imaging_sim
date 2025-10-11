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
After optimization, we found that placing the sensor at **20.5 mm** after the aperture yields the best focus.
![](../out/biconvex_psf_log.png)

## N-sweep (sampling/)
We sweeped N for **[50, 100, 400,1600,3200,6400]**, and calculated the **Metrics vs N**. 
To further examine the focus sensitivity, we use a lareger aperture **OD = 6.35** and repeated the PSF analysis. This deliberate defocus allows us to study how the spot size and energy distribution degrade when the image plane is displaced, providing a better understanding of depth-of-focus and system tolerance.

We illustrate the **N = 50, 400, 3200**, respectively and more to be found in the [folder](../out/sweep_N)
| N = 50 | N = 400 | N = 3200 |
|:-------:|:--------:|:---------:|
| ![PSF 50](../out/sweep_N/biconvex_psf_50_log.png) | ![PSF 400](../out/sweep_N/biconvex_psf_400_log.png) | ![PSF 400](../out/sweep_N/biconvex_psf_3200_log.png)  |


EE50 and RMS metrics are recorded in [metrics.csv](../out/sweep_N/metrics.csv) and ploted ![](../out/sweep_N/metrics_vs_N.png)

The convergence begins around N ≥ 1600, where the RMS and EE50 values stabilize to ~0.045 mm and ~0.021 mm, respectively — indicating sufficient ray sampling density for accurate PSF estimation. To get better simulation, we choose N=3200 for the rest of experiments.

### Wavelength Sweep

We sweep wavelength for **range(430 ,670, 10) nm**, a range of vible spectrum.
| lambda = 430 | lambda = 520 | lambda = 610 |
|:-------:|:--------:|:---------:|
| ![PSF 430](../out/sweep_lambda/biconvex_psf_430_log.png) | ![PSF 520](../out/sweep_lambda/biconvex_psf_520_log.png) | ![PSF 610](../out/sweep_lambda/biconvex_psf_610_log.png)  |


EE50 and RMS metrics are recorded in [metrics.csv](../out/sweep_lambda/metrics.csv) and ploted ![](../out/sweep_lambda/metrics_vs_lambda_nm.png)



With BK7 dispersion enabled, the smallest PSF occurs near ~500 nm and grows toward both spectral ends. This matches expectation: the effective focal length shifts with λ (chromatic focus), so a single sensor position cannot be perfectly focused for all wavelengths.

### Through-focus (D2 sweep)

We sweep D2 for range **range (19.5,21.5,13) mm,**, the best focus is at 20.5


EE50 and RMS metrics are recorded in [metrics.csv](../out/sweep_D2/metrics.csv) and ploted ![](../out/sweep_D2/metrics_vs_D2.png)

### Aperture Sweep (OD)

We sweep OD: **1.5875 mm (f/16), 3.175 mm (f/8), 6.35 mm (f/4),12.7 mm (f/2)**

EE50 and RMS metrics are recorded in [metrics.csv](../out/sweep_OD/metrics.csv) and ploted ![](../out/sweep_OD/metrics_vs_OD.png)

### Off-axis sweep

We consider a larger aperture **6.35 mm (f/4)** for this experiment to have a better view of the results.

EE50 and RMS metrics are recorded in [metrics.csv](../out/offaxis/metrics.csv) and ploted ![](../out/offaxis/metrics_vs_x_off_mm.png)


### field_grid sweep

We consider a larger aperture **6.35 mm (f/4)** for this experiment to have a better view of the results.

EE50 and RMS metrics are recorded in [metrics.csv](../out/field_grid/metrics.csv) and ploted ![](../out/field_grid/metrics_rms_radius_mm_field_heatmap.png)