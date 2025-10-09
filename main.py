from __future__ import annotations
import argparse
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
    lens = Lensgroup(device=device, pixel_size=pixel_size_mm, film_size=film_M)

    r_lens = float(LD) * 0.5
    c1 = 0.0 if R1 == 0 else 1.0 / float(R1)
    c2 = 0.0 if R2 == 0 else 1.0 / float(R2)

    s1 = Aspheric(r=r_lens, d=0.0,    c=c1, device=device)   # at z=0
    s2 = Aspheric(r=r_lens, d=float(T), c=c2, device=device)  # at z=T
    surfaces = [s1, s2]
    materials = [Material("AIR"), Material("bk7"), Material("AIR")]

    
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
    lens.plot_psf(I, show=False, fname=out_dir / f"{args.prefix}_psf.png")
    dis = lens.best_focus_D2( D_mm = args.D, lam=500.0, N=5000, D2_guess=None, span=5.0, steps=21, use_spot=True) - args.stop_after_s2_mm

    
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
    p.add_argument("--D2", type=float, default=21.3, help="Aperture to sensor distance [mm]")

    # Stop
    p.add_argument("--stop_after_s2_mm", type=float, default=2.0, help="Stop position after S2 [mm]")
    p.add_argument("--no_stop", action="store_true", help="Disable explicit AIR–AIR stop")

    # Source / sampling / sensor
    p.add_argument("--D",         type=float, default=1000.0, help="Object distance [mm]")
    p.add_argument("--lambda_nm", type=float, default=500.0,  help="Wavelength [nm]")
    p.add_argument("--N",         type=int,   default=2000,   help="Rays (angle-uniform)")
    p.add_argument("--M",         type=list,   default=[256,256],    help="Sensor MxM pixels")
    p.add_argument("--h",         type=float, default=3.2,    help="Sensor side length h [mm] (pixel_size=h/M)")

    # Experiment control
    p.add_argument("--exp", type=str, default="all",
                   choices=["all","single","sweep_N","sweep_lambda","offaxis"],
                   help="Which experiment to run")

    # Optional lists
    p.add_argument("--N_list", nargs="+", type=int, help="Override N sweep list, e.g., --N_list 100 500 2000")
    p.add_argument("--lambda_list", nargs="+", type=float, help="Override lambda sweep list")
    p.add_argument("--offaxis_list", nargs="+", type=float, help="Override off-axis x offsets [mm]")

    # IO / misc
    p.add_argument("--out_dir", type=str, default="out", help="Output directory")
    p.add_argument("--prefix",  type=str, default="biconvex", help="Filename prefix for single run")
    p.add_argument("--cuda",    action="store_true", help="Use CUDA if available")

    return p.parse_args()
if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(0); np.random.seed(0)

    run_first(args)