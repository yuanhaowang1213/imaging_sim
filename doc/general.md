## General Consideration

## Method
### Geometry & Variables
- **Surfaces**
  - **S1**: AIR → BK7 at \(z=0\), radius of curvature \(R_1\)
  - **S2**: BK7 → AIR at \(z=T\), radius \(R_2\)

### Ray Sampling
- **Angle-uniform cone**
- **Off-axis**: source shifted to \((x_{	ext{off}}, y_{	ext{off}})\). Use **chief-ray-aligned** cone (`sample_offaxis_point_axis`) so the cone is centered on the stop.

### Refraction & Intersection

- Exact Snell (vector form); no paraxial approximation.  
- Aspheric surfaces (spherical as special case).  
- Newton iteration for intersection.

- **Optional aperture stop**: AIR–AIR at \(z = T + \Delta z_{	ext{stop}}\), diameter \(OD\)
- **Sensor plane**: \(z = T + \Delta z_{	ext{stop}} + D2\)
- **Lens diameter**: \(LD\)
- **Sensor**: size \(h 	imes h\) mm with \(M 	imes M\) pixels (pixel size \(p = h/M\))

### Rendering
Here, the task is to simulate PSF, but we also consider about the rendering of the images if we have some time for extenstion.

We took the following steps:
![](./imgs/ray_tracing.png)

## Modeling
### Lens modeling
We use a Zemax-style **Surface Data Table** to align with lens settings.

Here is an explanation of different components.

---
| Column       | Meaning              | Description                                                                                           |
| ------------ | -------------------- | ----------------------------------------------------------------------------------------------------- |
| **Type**     | Surface type         | `O` = Object, `S` = Surface, `A` = Aperture/Stop, `I` = Image                                         |
| **Distance** | Thickness / spacing  | Physical distance to the next surface (in mm)                                                         |
| **ROC**      | Radius of curvature  | Positive = center to the right (convex left); negative = center to the left (concave right); 0 = flat |
| **Diameter** | Clear aperture       | Effective aperture or surface diameter (in mm)                                                        |
| **Material** | Medium or glass type | Optical material (e.g., BK7, SSK4, SK1, F15, SK16, VACUUM)                                                 |
---

To match Thorlabs **LB1761** (simple BK7 biconvex singlet), we map \[R1, T, R2, D2, OD\] (mm) as:

- **R1 = 24.5**, **T = 9**, **R2 = −24.5**, **D2 = 22.2**, **OD = 3.175** (≈ f/8)

Parameter table:


---
| # | Type  | Distance (mm) | ROC (mm) | Diameter (mm) | Material |
| - | ----- | ------------- | -------- | ------------- | -------- |
| 1 | **O** | 0.000         | 0.000    | 0.000         | AIR      |
| 2 | **S** | 0.000         | 24.5     | 25.4          | BK7    |
| 3 | **S** | 9.000         | -24.5    | 25.4          | AIR      |
| 4 | **A** | 2.000         | 0.000    | 3.175         | OCCLUDER |
| 5 | **I** | 23.800        | 0.000    | 25.4          | AIR      |
---


## Reference:
- Wang, Congli, Ni Chen, and Wolfgang Heidrich. "do: A differentiable engine for deep lens design of computational imaging systems." IEEE Transactions on Computational Imaging 8 (2022): 905-916.
- Ho, Chi-Jui, et al. "A Differentiable Wave Optics Model for End-to-End Computational Imaging System Optimization." arXiv preprint arXiv:2412.09774 (2024).
## Code reference:
https://github.com/vccimaging/DiffOptics