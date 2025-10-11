# Single run (layout, rays, one PSF + metrics)
python3 main.py --exp single --OD 1.6
python3 main.py --exp single --OD 3.175
python3 main.py --exp single --OD 6.35
python3 main.py --exp single --OD 12.7
python3 main.py --exp single --OD 3.175


# Ray-count sweep (sampling / aliasing study)
python3 main.py --exp sweep_N --OD 6.35 --D2 20.3 --N_list 50 100 400 1600 3200 6400

# Ray-count sweep (sampling / aliasing study)
python3 main.py --exp sweep_lambda 

# Off-axis sweep (±35 mm at source plane, 36 steps total)
python3 main.py --exp offaxis --field_max_mm 35 --field_steps 36 --OD 6.35 --D2 20.3

# Through-focus sweep for D2
python3 main.py --exp sweep_D2 --D2_span 6 --D2_steps 31 --D2_sweep_span 1 --D2_sweep_steps 13

# Aperture sweep with per-aperture refocus
python3 main.py --exp sweep_OD --OD_list 1.6 3.175 6.35 12.7 

