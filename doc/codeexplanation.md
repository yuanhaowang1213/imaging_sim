## Code design:
- **Core class**: `Lensgroup` (surfaces, materials, transform, tracing).
- **Key functions**:
  - `sample_ray_from_point`, `sample_offaxis_point_axis`
  - `render` (PSF), `best_focus_D2` (spot-size/EE50 search)
  - `psf_metrics` (sum, centroid, RMS radius, EE50)
  - **Plot helpers**: layout / rays / PSF (linear & log scale).
- **Tests in `tests/test_geo.py` and `test_ray_tracing.py`**: test the design and the basic tracing function
- **Experiments in `main.py`**: `single`, `sweep_N`, `sweep_lambda`, `offaxis`, `sweep_D2` (through focus), `sweep_OD` (aperture sweep, optional refocus).

## code running:
### Experiment run: 
The main experiment is in the [main.py](../main.py) and [main_double.py](../main_doublegaussian.py)


### Plot run