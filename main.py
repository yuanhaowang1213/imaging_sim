from __future__ import annotations
import argparse,csv
import math
from pathlib import Path
import sys, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
from mlrt.optics import Lensgroup, Aspheric
from mlrt.materials import Material
from mlrt.basics import Ray


def build_lens(
    R1: float, T: float, R2: float, LD: float, OD: float, D2: float,
    pixel_size_mm: float, film_M: list[int], device: torch.device,
    stop_after_s2_mm: float = 0.0, add_explicit_stop: bool = True,
) -> Lensgroup:
    """
    Geometry:
      S1: AIR -> BK7 at z=0
      S2: BK7 -> AIR at z=T, radius LD
      A : optional AIR–AIR stop at z = T + stop_after_s2_mm, radius = OD/2
      Sensor plane:
      z = T + stop_after_s2_mm + D2 (used only for intersection)
    """
    print(f"pixel size is {pixel_size_mm*1e3} um")
    lens = Lensgroup(device=device, pixel_size=pixel_size_mm, film_size=film_M)

    r_lens = float(LD) * 0.5
    c1 = 0.0 if R1 == 0 else 1.0 / float(R1)
    c2 = 0.0 if R2 == 0 else 1.0 / float(R2)

    s1 = Aspheric(r=r_lens, d=0.0,    c=c1, device=device)   # at z=0
    s2 = Aspheric(r=r_lens, d=float(T), c=c2, device=device)  # at z=T
    surfaces = [s1, s2]
    materials = [Material("AIR",dispersion=False), Material("bk7",dispersion=True), Material("AIR",dispersion=False)]

    
    aperture_ind = None
    if add_explicit_stop:
        zA = float(T) + float(stop_after_s2_mm)
        a = Aspheric(r=OD*0.5, d=zA, c=0.0, device=device)     # AIR–AIR stop
        surfaces.append(a)
        materials.append(Material("AIR"))
        aperture_ind = len(surfaces) - 1

    lens.surfaces = surfaces
    lens.materials = materials
    lens.d_sensor = float(T) + float(stop_after_s2_mm) + float(D2)
    lens.aperture_ind = aperture_ind
    lens._sync()
    return lens

# simple test imaging and print the best 
def run_first(args) -> None:
    device = torch.device("cuda" if (args.cuda and torch.cuda.is_available()) else "cpu")
    pixel_size_mm = float(args.h) / float(args.M[1])
    assert 1e-4 <= pixel_size_mm <= 0.05, "pixel_size should be in mm (e.g. 0.006 for 6 µm)"
    
    lens = build_lens(
        R1=args.R1, T=args.T, R2=args.R2, LD=args.LD, OD=args.OD, D2=args.D2,
        pixel_size_mm=pixel_size_mm, film_M=args.M, device=device,
        stop_after_s2_mm=args.stop_after_s2_mm, add_explicit_stop=(not args.no_stop),
    )
    # sample rays and render
    ray = lens.sample_ray_from_point(D_mm=args.D, wavelength=args.lambda_nm, N=args.N, filter_to_stop=True)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    # layout
    ax = lens.plot_layout2d(show=False, fname=out_dir / f"{args.prefix}_layout.png")
    
    _, oss = lens.trace_to_sensor_r(ray, ignore_invalid=False)
    lens.plot_raytraces_world(oss=oss, ax=ax, show=False, fname = out_dir / f"{args.prefix}_rays.png")
    I = lens.render(ray, irr=1.0)
    if torch.max(I) > 0:
        I = I / torch.max(I) 
    lens.plot_psf(I, show=False, log=True, fname=out_dir / f"{args.prefix}_psf_log.png")
    dis = lens.best_focus_D2( D_mm = args.D, lam=500.0, N=5000, D2_guess=None, span=5.0, steps=21, use_spot=True) - args.stop_after_s2_mm
    print(f'the optimal distance between the sensor and the aperture {dis} mm')
    m = lens.psf_metrics( I)
    with open(out_dir / f"{args.prefix}_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(m.keys()))
        w.writeheader(); w.writerow(m)
    print(f"[single] saved results to {out_dir.resolve()}")

# --------------------------
# Experiment 1: N sweep (aliasing / sampling)
# --------------------------
def run_sweep_N(args) ->None:
    Ns = args.N_list or [50, 100, 400,1600,3200,6400]
    out_dir = Path(args.out_dir) / "sweep_N"; out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if (args.cuda and torch.cuda.is_available()) else "cpu")
    pixel_size_mm = float(args.h) / float(args.M[1])
    lens = build_lens(
        R1=args.R1, T=args.T, R2=args.R2, LD=args.LD, OD=args.OD, D2=args.D2+2, # off focus to see the change of sampling effect
        pixel_size_mm=pixel_size_mm, film_M=args.M, device=device,
        stop_after_s2_mm=args.stop_after_s2_mm, add_explicit_stop=(not args.no_stop),
    )
    # sample rays and render
    rows = []
    for N in Ns:
        ray = lens.sample_ray_from_point(D_mm=args.D, wavelength=args.lambda_nm, N=int(N), filter_to_stop=True)
        I = lens.render(ray, irr=1.0)
        if torch.max(I) > 0:
            I_disp = I / (torch.max(I) + 1e-12)
        lens.plot_psf(I_disp, show=False, log=True, fname=out_dir / f"{args.prefix}_psf_{N}_log.png")
        if torch.max(I) > 0:
            I_ene = I / (torch.sum(I) + 1e-12)
        m = lens.psf_metrics( I_ene) 
        m["N"] = int(N)
        rows.append(m)
    with open(out_dir / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["N","sum","cx_mm","cy_mm","r_centroid_mm","rms_radius_mm","ee50_mm"])
        w.writeheader(); [w.writerow(r) for r in rows]
    print(f"[sweep_N] done: {out_dir.resolve()}")

# --------------------------
# Experiment 2: wavelength sweep
# --------------------------
def run_sweep_lambda(args) -> None:
    lambdas = args.lambda_list or [450.0, 550.0, 650.0]
    out_dir = Path(args.out_dir) / "sweep_lambda"; out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if (args.cuda and torch.cuda.is_available()) else "cpu")
    pixel_size_mm = float(args.h) / float(args.M[0])
    lens = build_lens(
        R1=args.R1, T=args.T, R2=args.R2, LD=args.LD, OD=args.OD, D2=args.D2,
        pixel_size_mm=pixel_size_mm, film_M=args.M, device=device,
        stop_after_s2_mm=args.stop_after_s2_mm, add_explicit_stop=(not args.no_stop),
    )

    rows = []
    for lam in lambdas:
        ray = lens.sample_ray_from_point(D_mm=args.D, wavelength=float(lam), N=args.N, filter_to_stop=True)
        I = lens.render(ray, irr=1.0)
        if torch.max(I) > 0:
            I_disp = I / (torch.max(I) + 1e-12)
        lens.plot_psf(I_disp, show=False, log=True, fname=out_dir / f"{args.prefix}_psf_{lam}_log.png")
        if torch.max(I) > 0:
            I_ene = I / (torch.sum(I) + 1e-12)
        m = lens.psf_metrics( I_ene) 
        m["lambda_nm"] = float(lam)
        rows.append(m)
    with open(out_dir / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["lambda_nm","sum","cx_mm","cy_mm","r_centroid_mm","rms_radius_mm","ee50_mm"])
        w.writeheader(); [w.writerow(r) for r in rows]
    print(f"[sweep_lambda] done: {out_dir.resolve()}")

# --------------------------
# Experiment 3: off-axis source
# --------------------------
def run_offaxis(args) -> None:
    offsets = args.offaxis_list or [0.0, 1.0, 2.0]  # lateral shift in mm at the source plane
    out_dir = Path(args.out_dir) / "offaxis"; out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if (args.cuda and torch.cuda.is_available()) else "cpu")
    pixel_size_mm = float(args.h) / float(args.M[0])
    lens = build_lens(
        R1=args.R1, T=args.T, R2=args.R2, LD=args.LD, OD=args.OD, D2=args.D2,
        pixel_size_mm=pixel_size_mm, film_M=args.M, device=device,
        stop_after_s2_mm=args.stop_after_s2_mm, add_explicit_stop=(not args.no_stop),
    )

    rows = []
    for dx in offsets:
        ray = lens.sample_offaxis_point_axis( D_mm=args.D, wavelength=args.lambda_nm, N=args.N,
                                   x_off_mm=float(dx), y_off_mm=0.0, filter_to_stop=True)
        I = lens.render(ray, irr=1.0)
        if torch.max(I) > 0:
            I_disp = I / (torch.max(I) + 1e-12)
        lens.plot_psf(I_disp, show=False, log=True, fname=out_dir / f"{args.prefix}_psf_{dx:0.2f}_log.png")
        if torch.max(I) > 0:
            I_ene = I / (torch.sum(I) + 1e-12)
        m = lens.psf_metrics( I_ene) 
        m["x_off_mm"] = float(dx)
        rows.append(m)
    with open(out_dir / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["x_off_mm","sum","cx_mm","cy_mm","r_centroid_mm","rms_radius_mm","ee50_mm"])
        w.writeheader(); [w.writerow(r) for r in rows]
    print(f"[offaxis] done: {out_dir.resolve()}")

# --------------------------
#  Experiment 4 — D2 through-focus sweep
# --------------------------
def run_sweep_D2(args) -> None:
    out_dir = Path(args.out_dir) / "sweep_D2"; out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if (args.cuda and torch.cuda.is_available()) else "cpu")
    pixel_size_mm = float(args.h) / float(args.M[1])
    lens = build_lens(
        R1=args.R1, T=args.T, R2=args.R2, LD=args.LD, OD=args.OD, D2=args.D2,
        pixel_size_mm=pixel_size_mm, film_M=args.M, device=device,
        stop_after_s2_mm=args.stop_after_s2_mm, add_explicit_stop=(not args.no_stop) )
    # find best focus as center
    D2_best = lens.best_focus_D2(D_mm=args.D, lam=args.lambda_nm, N=max(2000, args.N),
                                 D2_guess=args.D2, span=args.D2_span, steps=args.D2_steps,
                                 use_spot=True)
    z2 = float(lens.surfaces[1].d)
    grid = np.linspace(D2_best - args.D2_sweep_span, D2_best + args.D2_sweep_span, args.D2_sweep_steps)

    rows = []
    for D2 in grid:
        lens.d_sensor = z2 + float(args.stop_after_s2_mm) + float(D2)
        rays = lens.sample_ray_from_point(D_mm=args.D, wavelength=args.lambda_nm, N=args.N, filter_to_stop=True)
        I = lens.render(rays, irr=1.0)
        if torch.max(I) > 0:
            I_disp = I / (torch.max(I) + 1e-12); I_ene = I / (torch.sum(I) + 1e-12)
            lens.plot_psf(I_disp, show=False, log=True, fname=out_dir / f"{args.prefix}_psf_D2_{D2:.3f}_log.png")
            m = lens.psf_metrics(I_ene); m["D2"] = float(D2)
            rows.append(m)
    with open(out_dir / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["D2","sum","cx_mm","cy_mm","r_centroid_mm","rms_radius_mm","ee50_mm"])
        w.writeheader(); [w.writerow(r) for r in rows]
    print(f"[sweep_D2] done: {out_dir.resolve()}")

# --------------------------
# Experiment 5 — Aperture OD sweep (with optional refocus)
# --------------------------
def run_sweep_OD(args) -> None:
    OD_list = args.OD_list or [args.OD/2,args.OD,  args.OD*2, args.OD*4]
    out_dir = Path(args.out_dir) / "sweep_OD"; out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if (args.cuda and torch.cuda.is_available()) else "cpu")
    pixel_size_mm = float(args.h) / float(args.M[1])

    rows = []
    for OD in OD_list:
        lens = build_lens(
            R1=args.R1, T=args.T, R2=args.R2, LD=args.LD, OD=OD, D2=args.D2,
            pixel_size_mm=pixel_size_mm, film_M=args.M, device=device,
            stop_after_s2_mm=args.stop_after_s2_mm, add_explicit_stop=(not args.no_stop),
        )
        if args.refocus_per_OD:
            D2_best = lens.best_focus_D2(D_mm=args.D, lam=args.lambda_nm, N=max(2000, args.N),
                                         D2_guess=args.D2, span=args.D2_span, steps=args.D2_steps,
                                         use_spot=True)
            z2 = float(lens.surfaces[1].d)
            lens.d_sensor = z2 + float(args.stop_after_s2_mm) + float(D2_best)

        rays = lens.sample_ray_from_point(D_mm=args.D, wavelength=args.lambda_nm, N=args.N, filter_to_stop=True)
        I = lens.render(rays, irr=1.0)
        if torch.max(I) > 0:
            I_disp = I / (torch.max(I) + 1e-12); I_ene = I / (torch.sum(I) + 1e-12)
            lens.plot_psf(I_disp, show=False, log=True, fname=out_dir / f"{args.prefix}_psf_OD_{OD:.3f}_log.png")
            m = lens.psf_metrics(I_ene); m["OD"] = float(OD)
            rows.append(m)

    with open(out_dir / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["OD","sum","cx_mm","cy_mm","r_centroid_mm","rms_radius_mm","ee50_mm"])
        w.writeheader(); [w.writerow(r) for r in rows]
    print(f"[sweep_OD] done: {out_dir.resolve()}")


# --------------------------
#  Experiment 6 — Field map (2D off-axis grid)
# --------------------------
def run_field_grid(args) -> None:
    out_dir = Path(args.out_dir) / "field_grid"; out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if (args.cuda and torch.cuda.is_available()) else "cpu")
    pixel_size_mm = float(args.h) / float(args.M[1])
    lens = build_lens(
        R1=args.R1, T=args.T, R2=args.R2, LD=args.LD, OD=args.OD, D2=args.D2,
        pixel_size_mm=pixel_size_mm, film_M=args.M, device=device,
        stop_after_s2_mm=args.stop_after_s2_mm, add_explicit_stop=(not args.no_stop),
    )
    xs = np.linspace(-args.field_max_mm, args.field_max_mm, args.field_steps)
    ys = np.linspace(-args.field_max_mm, args.field_max_mm, args.field_steps)

    rows = []
    for x in xs:
        for y in ys:
            rays = lens.sample_offaxis_point_axis( D_mm=args.D, wavelength=args.lambda_nm, N=args.N,
                                        x_off_mm=float(x), y_off_mm=float(y), filter_to_stop=True)
            I = lens.render(rays, irr=1.0)
            if torch.max(I) > 0:
                I_disp = I / (torch.max(I) + 1e-12); I_ene = I / (torch.sum(I) + 1e-12)
                tag = f"{x:+.2f}_{y:+.2f}".replace('+','p').replace('-','m')
                lens.plot_psf(I_disp, show=False, log=True, fname=out_dir / f"{args.prefix}_psf_{tag}_log.png")
                m = lens.psf_metrics(I_ene); m["x_off_mm"] = float(x); m["y_off_mm"] = float(y)
                rows.append(m)

    with open(out_dir / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["x_off_mm","y_off_mm","sum","cx_mm","cy_mm","r_centroid_mm","rms_radius_mm","ee50_mm"])
        w.writeheader(); [w.writerow(r) for r in rows]
    print(f"[field_grid] done: {out_dir.resolve()}")



# --------------------------
# Argument parsing
# --------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Biconvex lens PSF experiments (single / sweeps)."
    )
    # Geometry
    p.add_argument("--R1", type=float, default=24.5, help="Front radius R1 [mm] (convex>0)")
    p.add_argument("--T",  type=float, default=9.0,  help="Center thickness T [mm]")
    p.add_argument("--R2", type=float, default=-24.5,help="Back radius R2 [mm] (convex to sensor often negative)")
    p.add_argument("--LD", type=float, default=25.4, help="Diameter of the lens [mm]")
    p.add_argument("--OD", type=float, default=3.175, help="Aperture diameter OD [mm]")
    p.add_argument("--D2", type=float, default=20.3, help="Aperture to sensor distance [mm]")

    # Stop
    p.add_argument("--stop_after_s2_mm", type=float, default=2.0, help="Stop position after S2 [mm]")
    p.add_argument("--no_stop", action="store_true", help="Disable explicit AIR–AIR stop")

    # Source / sampling / sensor
    p.add_argument("--D",         type=float, default=1000.0, help="Object distance [mm]")
    p.add_argument("--lambda_nm", type=float, default=500.0,  help="Wavelength [nm]")
    p.add_argument("--N",         type=int,   default=3200,   help="Rays (angle-uniform)")
    p.add_argument("--M",         type=list,   default=[512,512],    help="Sensor MxM pixels")
    p.add_argument("--h",         type=float, default=3.2,    help="Sensor side length h [mm] (pixel_size=h/M)")

    # Experiment control
    p.add_argument("--exp", type=str, default="all",
                   choices=["all","single","sweep_N","sweep_lambda","offaxis"],
                   help="Which experiment to run")

    # Optional lists
    p.add_argument("--N_list", nargs="+", type=int, help="Override N sweep list, e.g., --N_list 100 500 2000")
    p.add_argument("--lambda_list", nargs="+", type=float, help="Override lambda sweep list")
    p.add_argument("--offaxis_list", nargs="+", type=float, help="Override off-axis x offsets [mm]")

    p.add_argument("--D2_span", type=float, default=5.0, help="Best-focus search span [mm]")
    p.add_argument("--D2_steps", type=int, default=21, help="Best-focus search steps")
    p.add_argument("--D2_sweep_span", type=float, default=1.0, help="Through-focus sweep ±span around best [mm]")
    p.add_argument("--D2_sweep_steps", type=int, default=13, help="Through-focus sweep steps")

    p.add_argument("--OD_list", nargs="+", type=float, help="OD sweep list [mm]")
    p.add_argument("--refocus_per_OD", action="store_true", help="Refocus (best D2) for each OD")

    p.add_argument("--field_max_mm", type=float, default=1.0, help="Off-axis grid half-width [mm] at source plane")
    p.add_argument("--field_steps", type=int, default=3, help="Off-axis grid steps per axis")

    # IO / misc
    p.add_argument("--out_dir", type=str, default="out", help="Output directory")
    p.add_argument("--prefix",  type=str, default="biconvex", help="Filename prefix for single run")
    p.add_argument("--cuda",    action="store_true", help="Use CUDA if available")

    return p.parse_args()
if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(0); np.random.seed(0)

    # run_first(args)
    # run_sweep_N(args)
    # run_sweep_lambda(args)
    run_offaxis(args)
    run_sweep_D2(args)
    run_sweep_OD(args)
    run_field_grid(args)