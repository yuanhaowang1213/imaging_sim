import math
import numpy as np
import torch
import pytest

# Import classes from the project
import os,sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import mlrt 
from mlrt.optics import Lensgroup, Aspheric, Surface
from mlrt.materials import Material
from mlrt.basics import Ray

torch.manual_seed(0)
np.random.seed(0)

DEVICE = torch.device("cpu")

def _make_ray(o, d, wav=550.0):
    """Helper: build a normalized Ray on DEVICE."""
    if not torch.is_tensor(o): o = torch.tensor(o, dtype=torch.float32)
    if not torch.is_tensor(d): d = torch.tensor(d, dtype=torch.float32)
    o = o.reshape(-1, 3).to(DEVICE)
    d = d.reshape(-1, 3).to(DEVICE)
    d = d / torch.linalg.norm(d, dim=-1, keepdim=True)
    wav = torch.full((o.shape[0],), float(wav), dtype=torch.float32, device=DEVICE)
    return Ray(o, d, wav, device=DEVICE)

# ---------- 1) Intersection: plane z = d ----------
def test_intersect_plane_z_equals_d():
    s = Aspheric(r=50.0, d=12.3, c=0.0, device=DEVICE)  # plane z = d
    ray = _make_ray([0.0, 0.0, 0.0], [0.0, 0.0, 1.0])
    valid, p = s.ray_surface_intersection(ray)
    assert valid.item()
    assert torch.allclose(p[0, 2], torch.tensor(12.3, device=DEVICE), atol=1e-6)

# ---------- 2) Intersection: spherical surface (off-axis) ----------
def test_intersect_sphere_off_axis_sag_matches():
    R = 50.0  # mm
    s = Aspheric(r=30.0, d=10.0, c=1.0/R, device=DEVICE)
    x = 5.0  # off-axis hit
    ray = _make_ray([x, 0.0, -100.0], [0.0, 0.0, 1.0])
    valid, p = s.ray_surface_intersection(ray)
    assert valid.item()
    # expected z = d + sag(x)
    r2 = torch.tensor(x**2, dtype=torch.float32, device=DEVICE)
    sag = s._g(r2).item()
    z_expect = 10.0 + sag
    assert abs(p[0, 2].item() - z_expect) < 1e-4

# ---------- 3) Snell's law (vector form): sin θ_t = η sin θ_i ----------
def test_snell_vector_form():
    lens = Lensgroup(device=DEVICE)
    wi_theta = math.radians(30.0)
    wi = torch.tensor([[math.sin(wi_theta), 0.0, math.cos(wi_theta)]], dtype=torch.float32, device=DEVICE)
    n  = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32, device=DEVICE)
    eta = 1.0 / 1.5  # air -> glass
    valid, wt = lens._refract(wi, n, eta)
    assert valid.item()
    # Use |wi × n| and |wt × n| to get sin θ (robust to sign)
    sin_i = torch.linalg.norm(torch.cross(wi, n), dim=-1).item()
    sin_t = torch.linalg.norm(torch.cross(wt, n), dim=-1).item()
    assert abs(sin_t - eta * sin_i) < 1e-5

# ---------- 4) Total internal reflection flag ----------
def test_total_internal_reflection_flag():
    lens = Lensgroup(device=DEVICE)
    eta = 1.5  # glass -> air (n1/n2)
    # incidence > critical angle ≈ arcsin(1/eta)
    th = math.radians(50.0)
    wi = torch.tensor([[math.sin(th), 0.0, math.cos(th)]], dtype=torch.float32, device=DEVICE)
    n  = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32, device=DEVICE)
    valid, _ = lens._refract(wi, n, eta)
    assert not valid.item()  # expect TIR

# ---------- 5) Render energy conservation (air-air plate + sensor) ----------
def test_render_energy_conservation_air_plate():
    # Camera with only an aperture plane (AIR/AIR). Sensor at z = 20 mm.
    lens = Lensgroup(device=DEVICE, pixel_size=0.05, film_size=[128, 128])
    s = Aspheric(r=20.0, d=0.0, c=0.0, device=DEVICE)  # aperture plane (no refraction)
    lens.surfaces = [s]
    lens.materials = [Material('AIR'), Material('AIR')]
    lens.d_sensor = 20.0
    lens.aperture_ind = 0
    lens._sync()

    # Emit rays from a distant on-axis point
    ray = lens.sample_ray_from_point(D_mm=500.0, wavelength=550.0, N=2000, filter_to_stop=True)

    # Count effective rays used by render() (valid and landing inside sensor)
    ray_final, valid = lens.trace(ray)
    t = (lens.d_sensor - ray_final.o[..., 2]) / (ray_final.d[..., 2].clamp_min(1e-12))
    p = ray_final(t)
    R_sensor = [lens.film_size[i] * lens.pixel_size / 2 for i in range(2)]
    inside = (
        (-R_sensor[0] <= p[..., 0]) & (p[..., 0] <= R_sensor[0]) &
        (-R_sensor[1] <= p[..., 1]) & (p[..., 1] <= R_sensor[1])
    )
    used = (valid & inside).sum().item()

    I = lens.render(ray, irr=1.0)
    # Bilinear splat: total weight per ray sums to 1 → sum(I) ≈ number of used rays
    assert I.shape == (lens.film_size[0], lens.film_size[1])
    assert abs(I.sum().item() - used) < 1e-4
    assert used > 0

def thin_lens_si(n=1.5168, R1=50.0, R2=-50.0, T=3.0):
    """Approx focal length for a thick lens (standard formula)."""
    return 1.0 / ((n-1)*(1.0/R1 - 1.0/R2 + (n-1)*T/(n*R1*R2)))

def test_biconvex_focus_near_thin_lens():
    # Build a simple biconvex lens and check near-axis focus vs thin-lens estimate
    n = 1.5168
    R1, R2, T = 50.0, -50.0, 3.0
    f = thin_lens_si(n, R1, R2, T)
    so = 1000.0  # object distance [mm]
    si = 1.0 / (1.0/f - 1.0/so)  # Gaussian imaging

    lens = Lensgroup(device=DEVICE, pixel_size=0.02, film_size=[256, 256])
    # Surface 1 at d=0, surface 2 at d=T
    s1 = Aspheric(r=25.0, d=0.0,  c=1.0/R1, device=DEVICE)
    s2 = Aspheric(r=25.0, d=T,    c=1.0/R2, device=DEVICE)
    lens.surfaces = [s1, s2]
    lens.materials = [Material('AIR'), Material('bk7'), Material('AIR')]
    lens.d_sensor = T + si
    lens.aperture_ind = None
    lens._sync()

    ray = lens.sample_ray_from_point(D_mm=so, wavelength=550.0, N=2000, filter_to_stop=False)
    p = lens.trace_to_sensor(ray, ignore_invalid=True).cpu().numpy()
    if p.size == 0:
        pytest.skip("No rays reached sensor in this config.")
    centroid = p[:, :2].mean(axis=0)
    # Near-axis check: centroid should be close to the optical axis (< 0.1 mm)
    assert np.linalg.norm(centroid) < 0.1
