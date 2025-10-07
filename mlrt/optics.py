# optics.py — Lensgroup + Aspheric surface + tracing, sampling, and PSF rendering
from __future__ import annotations
import math
import numpy as np
import torch
import pathlib
from typing import List, Tuple

from .basics import PrettyPrinter, Transformation, Ray, rodrigues_rotation_matrix
from .materials import Material



class Lensgroup(PrettyPrinter):
    """A collection of optical surfaces with a sensor plane.

    Coordinates:
      - Object frame: surfaces arranged along +z starting at z=0.
      - World ↔ Object via rigid transform (origin/shift/theta).

    Sensor:
      - pixel_size: mm per pixel
      - film_size: [HxW] pixels
    """
    def __init__(
        self,
        origin: torch.Tensor | None = None,
        shift: torch.Tensor | None = None,
        pixel_size: float = 6.45e-3,
        lambdas: List[float] | None = None,
        film_size: List[int] | None = None,
        theta: torch.Tensor | None = None,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.origin = (origin if origin is not None else torch.zeros(3)).to(device)
        self.shift = (shift if shift is not None else torch.zeros(3)).to(device)
        self.theta = (theta if theta is not None else torch.zeros(3)).to(device)
        self.device = device

        self.surfaces: List[Surface] = []
        self.materials: List[Material] = []

        self.pixel_size = float(pixel_size)  # mm
        self.film_size = list(film_size) if film_size is not None else [512, 512]

        self.to_world = self._compute_transformation()
        self.to_object = self.to_world.inverse()

    # ---- setup / IO ----
    def load_file(self, filename: pathlib.Path) -> None:
        self.surfaces, self.materials, self.r_last, d_last = self.read_lensfile(str(filename))
        # sensor plane z
        self.d_sensor = d_last + self.surfaces[-1].d
        self._sync()

    def update(self, _x: float = 0.0, _y: float = 0.0) -> None:
        self.to_world = self._compute_transformation(_x, _y)
        self.to_object = self.to_world.inverse()

    def _sync(self) -> None:
        for s in self.surfaces:
            s.to(self.device)
        self.aperture_ind = self._find_aperture()

    def _find_aperture(self) -> int | None:
        for i in range(len(self.surfaces) - 1):
            if self.materials[i].A < 1.0003 and self.materials[i + 1].A < 1.0003:
                return i
        return None

    def _compute_transformation(self, _x: float = 0.0, _y: float = 0.0, _z: float = 0.0) -> Transformation:
        R = (
            rodrigues_rotation_matrix(torch.tensor([1, 0, 0], device=self.device, dtype=torch.float32), torch.deg2rad(self.theta[0] + _x))
            @ rodrigues_rotation_matrix(torch.tensor([0, 1, 0], device=self.device, dtype=torch.float32), torch.deg2rad(self.theta[1] + _y))
            @ rodrigues_rotation_matrix(torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32), torch.deg2rad(self.theta[2] + _z))
        )
        t = self.origin + R @ self.shift
        return Transformation(R, t)

    @staticmethod
    def read_lensfile(filename: str) -> Tuple[List["Surface"], List["Material"], float, float]:
        surfaces: List[Surface] = []
        materials: List[Material] = []
        ds: List[float] = []
        with open(filename, "r") as file:
            line_no = 0
            d_total = 0.0
            for line in file:
                if line_no < 2:  # header lines
                    line_no += 1
                    continue
                ls = line.split()
                if not ls:
                    continue
                surface_type, d, r = ls[0], float(ls[1]), float(ls[3]) / 2.0
                roc = float(ls[2])
                if roc != 0:
                    roc = 1.0 / roc
                materials.append(Material(ls[4]))

                d_total += d
                ds.append(d)

                if surface_type == "O":  # object marker
                    d_total = 0.0
                    ds.pop()
                elif surface_type == "S":  # aspheric/spherical
                    if len(ls) <= 5:
                        surfaces.append(Aspheric(r, d_total, roc))
                    else:
                        conic = float(ls[5])
                        ai = [float(v) for v in ls[6:]] if len(ls) > 6 else None
                        surfaces.append(Aspheric(r, d_total, roc, conic, ai))
                elif surface_type == "A":  # aperture stop
                    surfaces.append(Aspheric(r, d_total, roc))
                elif surface_type == "I":  # sensor plane record
                    d_total -= d
                    ds.pop()
                    materials.pop()
                    r_last = r
                    d_last = d
        return surfaces, materials, r_last, d_last

    # ---- rendering ----
    def render(self, ray: "Ray", irr: float = 1.0) -> torch.Tensor:
        """Trace rays and bilinearly accumulate onto the sensor image [H,W]."""
        ray_final, valid = self.trace(ray)
        t = (self.d_sensor - ray_final.o[..., 2]) / (ray_final.d[..., 2].clamp_min(1e-12))
        p = ray_final(t)

        R_sensor = [self.film_size[i] * self.pixel_size / 2 for i in range(2)]
        valid = valid & (
            (-R_sensor[0] <= p[..., 0]) & (p[..., 0] <= R_sensor[0]) &
            (-R_sensor[1] <= p[..., 1]) & (p[..., 1] <= R_sensor[1])
        )
        p = p[valid]
        if p.numel() == 0:
            return torch.zeros(*self.film_size, device=self.device)

        u = (p[..., 0] + R_sensor[0]) / self.pixel_size
        v = (p[..., 1] + R_sensor[1]) / self.pixel_size

        i0 = torch.clamp(torch.floor(u).long(), 0, self.film_size[0] - 1)
        j0 = torch.clamp(torch.floor(v).long(), 0, self.film_size[1] - 1)
        i1 = torch.clamp(i0 + 1, 0, self.film_size[0] - 1)
        j1 = torch.clamp(j0 + 1, 0, self.film_size[1] - 1)

        wu = (u - i0.float()).clamp(0.0, 1.0)
        wv = (v - j0.float()).clamp(0.0, 1.0)
        wl = (1.0 - wu)
        wb = (1.0 - wv)

        I = torch.zeros(*self.film_size, device=self.device)
        I.index_put_((i0, j0), wl * wb * irr, accumulate=True)
        I.index_put_((i1, j0), wu * wb * irr, accumulate=True)
        I.index_put_((i0, j1), wl * wv * irr, accumulate=True)
        I.index_put_((i1, j1), wu * wv * irr, accumulate=True)
        return I

    # ---- samplers & tracing ----
    def sample_ray_from_point(
        self,
        D_mm: float = 1000.0,
        wavelength: float = 500.0,
        N: int = 50,
        theta_max: float | None = None,
        filter_to_stop: bool = True,
        device: torch.device | None = None,
    ) -> "Ray":
        """Emit N rays from on-axis point (0,0,z_src) in an **angularly uniform** grid.

        If `theta_max` is None, estimate it from the aperture stop geometry.
        If `filter_to_stop` is True, pre-filter rays by tracing to the stop.
        """
        if device is None:
            device = self.device

        # estimate theta_max from aperture stop edge as seen from the source
        if theta_max is None:
            stop_i = getattr(self, "aperture_ind", None)
            if stop_i is None:
                stop_i = 0
            stop = self.surfaces[stop_i]
            z_stop = float(stop.d.detach().cpu().item())
            r_stop = float(stop.r)
            theta_max = math.atan2(r_stop, D_mm + z_stop)

        # angular grid (uniform in angle, not solid angle)
        n_th = int(math.sqrt(N))
        n_ph = int(math.ceil(N / max(n_th, 1)))
        N_eff = n_th * n_ph
        th = torch.linspace(0.0, float(theta_max), steps=n_th, device=device)
        ph = torch.linspace(0.0, 2 * math.pi, steps=n_ph, device=device)
        TH, PH = torch.meshgrid(th, ph, indexing="ij")
        dx = torch.sin(TH) * torch.cos(PH)
        dy = torch.sin(TH) * torch.sin(PH)
        dz = torch.cos(TH)
        d = torch.stack((dx, dy, dz), dim=-1).reshape(-1, 3)

        # source in front of first surface by D_mm
        z_front = float(self.surfaces[0].d.detach().cpu().item())
        z_src = z_front - float(D_mm)
        o = torch.tensor([0.0, 0.0, z_src], device=device).repeat(N_eff, 1)
        wav = torch.full((N_eff,), float(wavelength), device=device)
        ray = Ray(o, d, wav, device=device)

        if filter_to_stop and getattr(self, "aperture_ind", None) is not None:
            _, valid = self.trace(ray, stop_ind=self.aperture_ind)
            ray.o = ray.o[valid]
            ray.d = ray.d[valid]
            ray.wavelength = ray.wavelength[valid]
        return ray

    def trace_to_sensor(self, ray: "Ray", ignore_invalid: bool = False) -> torch.Tensor:
        ray_final, valid = self.trace(ray)
        t = (self.d_sensor - ray_final.o[..., 2]) / (ray_final.d[..., 2].clamp_min(1e-12))
        p = ray_final(t)
        if ignore_invalid:
            p = p[valid]
        else:
            if p.ndim < 2:
                return p
            p = p.reshape(-1, 3)
        return p

    def trace_to_sensor_r(self, ray: "Ray", ignore_invalid: bool = False):
        ray_final, valid, oss = self.trace_r(ray)
        t = (self.d_sensor - ray_final.o[..., 2]) / (ray_final.d[..., 2].clamp_min(1e-12))
        p = ray_final(t)
        if ignore_invalid:
            p = p[valid]
        else:
            p = p.reshape(-1, 3)

        # transform recorded points to world coords correctly and append sensor hits
        oss_world = []
        hit_idx = 0
        for path_valid, path in zip(valid.tolist(), oss):
            if not path_valid:
                oss_world.append(path)
                continue
            path_w = []
            for pt in path:
                pt_t = torch.as_tensor(pt, device=self.device, dtype=torch.float32)
                path_w.append(self.to_world.transform_point(pt_t).detach().cpu().numpy())
            path_w.append(self.to_world.transform_point(torch.as_tensor(p[hit_idx], device=self.device, dtype=torch.float32)).detach().cpu().numpy())
            hit_idx += 1
            oss_world.append(path_w)
        return p, oss_world

    def trace(self, ray: "Ray", stop_ind: int | None = None):
        # to object frame
        ray_in = self.to_object.transform_ray(ray)
        valid, ray_out = self._trace(ray_in, stop_ind=stop_ind, record=False)
        # to world
        ray_final = self.to_world.transform_ray(ray_out)
        return ray_final, valid

    def trace_r(self, ray: "Ray", stop_ind: int | None = None):
        ray_in = self.to_object.transform_ray(ray)
        valid, ray_out, oss = self._trace(ray_in, stop_ind=stop_ind, record=True)
        ray_final = self.to_world.transform_ray(ray_out)
        return ray_final, valid, oss

    # ---- internal tracing ----
    def _refract(self, wi: torch.Tensor, n: torch.Tensor, eta: torch.Tensor | float, approx: bool = False):
        """Snell refraction: wi,n normalized; eta = n1/n2. Returns (valid, wt)."""
        if not torch.is_tensor(eta):
            eta_ = torch.as_tensor(eta, device=wi.device, dtype=wi.dtype)
        else:
            eta_ = eta if eta.ndim > 0 else eta.view(1)
        eta_ = eta_[..., None]

        cosi = (wi * n).sum(dim=-1, keepdim=True)
        if approx:
            tmp = 1.0 - (eta_ ** 2) * (1.0 - cosi)
            valid = (tmp > 0.0).squeeze(-1)
            wt = tmp * n + eta_ * (wi - cosi * n)
        else:
            cost2 = 1.0 - (1.0 - cosi ** 2) * (eta_ ** 2)
            valid = (cost2 > 0.0).squeeze(-1)
            cost = torch.sqrt(cost2.clamp_min(1e-12))
            wt = cost * n + eta_ * (wi - cosi * n)
        wt = wt / torch.linalg.norm(wt, dim=-1, keepdim=True).clamp_min(1e-12)
        return valid, wt

    def _trace(self, ray: "Ray", stop_ind: int | None = None, record: bool = False):
        if stop_ind is None:
            stop_ind = len(self.surfaces) - 1
        dim = ray.o[..., 2].shape
        if record:
            oss: List[list] = [[ray.o[i, :].detach().cpu().numpy()] for i in range(dim[0])]
        valid = torch.ones(dim, device=self.device, dtype=torch.bool)

        for i in range(stop_ind + 1):
            eta = self.materials[i].ior(ray.wavelength) / self.materials[i + 1].ior(ray.wavelength)
            valid_o, p = self.surfaces[i].ray_surface_intersection(ray, valid)
            n = self.surfaces[i].normal(p[..., 0], p[..., 1])
            valid_d, d = self._refract(ray.d, -n, eta)
            valid = valid & valid_o & valid_d
            if not valid.any():
                break
            if record:
                for path, v, pp in zip(oss, valid.detach().cpu().numpy(), p.detach().cpu().numpy()):
                    if v:
                        path.append(pp)
            ray.o = p
            ray.d = d

        if record:
            return valid, ray, oss
        return valid, ray

class Surface(PrettyPrinter):
    """Base implicit surface: f(x,y,z)=g(x,y)+h(z)=0 with circular/square aperture."""
    def __init__(self, r: float, d: float | torch.Tensor, is_square: bool = False, device: torch.device = torch.device("cpu")) -> None:
        self.d = d if torch.is_tensor(d) else torch.tensor(float(d), device=device)
        self.is_square = is_square
        self.r = float(r)
        self.device = device
        # tracing controls
        self.NEWTONS_MAXITER = 10
        self.NEWTONS_TOLERANCE_TIGHT = 50e-6
        self.NEWTONS_TOLERANCE_LOOSE = 300e-6
        self.APERTURE_SAMPLING = 257

    # utility
    def length2(self, d: torch.Tensor) -> torch.Tensor:
        return (d ** 2).sum(dim=-1)

    def length(self, d: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.length2(d))

    def normalize(self, d: torch.Tensor) -> torch.Tensor:
        return d / self.length(d).clamp_min(1e-12)[..., None]

    # common methods
    def surface_with_offset(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.surface(x, y) + self.d

    def normal(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        ds_dxyz = self.surface_derivatives(x, y)
        return self.normalize(torch.stack(ds_dxyz, dim=-1))

    def surface_area(self) -> float:
        return self.r ** 2 if self.is_square else math.pi * self.r ** 2

    def mesh(self) -> torch.Tensor:
        x, y = torch.meshgrid(
            torch.linspace(-self.r, self.r, self.APERTURE_SAMPLING, device=self.device),
            torch.linspace(-self.r, self.r, self.APERTURE_SAMPLING, device=self.device),
            indexing="ij",
        )
        valid_map = self.is_valid(torch.stack((x, y), dim=-1))
        return self.surface(x, y) * valid_map

    def sdf_approx(self, p: torch.Tensor) -> torch.Tensor:
        if self.is_square:
            return torch.max(torch.abs(p) - self.r, dim=-1)[0]
        return self.length2(p) - self.r ** 2

    def is_valid(self, p: torch.Tensor) -> torch.Tensor:
        return (self.sdf_approx(p) < 0.0).bool()

    def ray_surface_intersection(self, ray: "Ray", active: torch.Tensor | None = None):
        solution_found, local = self.newtons_method(ray.maxt, ray.o, ray.d)
        valid_o = solution_found & self.is_valid(local[..., 0:2])
        if active is not None:
            valid_o = active & valid_o
        return valid_o, local

    def newtons_method(self, maxt, o, D, option: str = "implicit"):
        ox, oy, oz = (o[..., i].clone() for i in range(3))
        dx, dy, dz = (D[..., i].clone() for i in range(3))
        A = dx ** 2 + dy ** 2
        B = 2 * (dx * ox + dy * oy)
        C = ox ** 2 + oy ** 2
        t0 = (self.d - oz) / (dz.clamp_min(1e-12))
        if option == "explicit":
            t, t_delta, valid = self.newtons_method_impl(maxt, t0, dx, dy, dz, ox, oy, oz, A, B, C)
        elif option == "implicit":
            with torch.no_grad():
                t, t_delta, valid = self.newtons_method_impl(maxt, t0, dx, dy, dz, ox, oy, oz, A, B, C)
                s_dot = self.surface_and_derivatives_dot_D(t, dx, dy, dz, ox, oy, t_delta * dz, A, B, C)[1]
            t = t0 + t_delta
            t = t - (self.g(ox + t * dx, oy + t * dy) + self.h(t_delta * dz)) / s_dot
        else:
            raise ValueError(f"option={option} not supported")
        p = o + t[..., None] * D
        return valid, p

    def newtons_method_impl(self, maxt, t0, dx, dy, dz, ox, oy, oz, A, B, C):
        if oz.numel() < 2:
            oz = torch.tensor([oz.item()], device=self.device, dtype=torch.float32)
        t_delta = torch.zeros_like(oz)
        t = maxt * torch.ones_like(oz)
        residual = maxt * torch.ones_like(oz)
        it = 0
        while (torch.abs(residual) > self.NEWTONS_TOLERANCE_TIGHT).any() and (it < self.NEWTONS_MAXITER):
            it += 1
            t = t0 + t_delta
            residual, s_dot = self.surface_and_derivatives_dot_D(t, dx, dy, dz, ox, oy, t_delta * dz, A, B, C)
            t_delta = t_delta - residual / s_dot
        t = t0 + t_delta
        valid = (torch.abs(residual) < self.NEWTONS_TOLERANCE_LOOSE) & (t <= maxt)
        return t, t_delta, valid

    # virtuals
    def g(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor: ...
    def dgd(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]: ...
    def h(self, z: torch.Tensor) -> torch.Tensor: ...
    def dhd(self, z: torch.Tensor) -> torch.Tensor: ...
    def surface(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor: ...
    def reverse(self) -> None: ...

    # default implementations
    def surface_derivatives(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gx, gy = self.dgd(x, y)
        z = self.surface(x, y)
        return gx, gy, self.dhd(z)

    def surface_and_derivatives_dot_D(self, t, dx, dy, dz, ox, oy, z, A, B, C):
        x = ox + t * dx
        y = oy + t * dy
        s = self.g(x, y) + self.h(z)
        sx, sy = self.dgd(x, y)
        sz = self.dhd(z)
        return s, sx * dx + sy * dy + sz * dz

class Aspheric(Surface):
    """Aspheric surface (spherical if ai=None and k=0).

    g(x,y) = c r² / (1 + sqrt(1 - (1+k) c² r²)) + (ai₀ r⁴ + ai₁ r⁶ + ...)
    h(z)   = -z
    """
    def __init__(self, r: float, d: float, c: float = 0.0, k: float = 0.0, ai: List[float] | None = None, is_square: bool = False, device: torch.device = torch.device("cpu")) -> None:
        super().__init__(r, d, is_square, device)
        self.c = torch.tensor(c, device=device, dtype=torch.float32)
        self.k = torch.tensor(k, device=device, dtype=torch.float32)
        self.ai = None if ai is None else torch.tensor(ai, device=device, dtype=torch.float32)

    def g(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self._g(x * x + y * y)

    def dgd(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dsdr2 = 2.0 * self._dgd(x * x + y * y)
        return dsdr2 * x, dsdr2 * y

    def h(self, z: torch.Tensor) -> torch.Tensor:
        return -z

    def dhd(self, z: torch.Tensor) -> torch.Tensor:
        return -torch.ones_like(z)

    def surface(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self._g(x * x + y * y)

    def reverse(self) -> None:
        self.c = -self.c
        if self.ai is not None:
            self.ai = -self.ai

    def surface_derivatives(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dsdr2 = 2.0 * self._dgd(x * x + y * y)
        return dsdr2 * x, dsdr2 * y, -torch.ones_like(x)

    def surface_and_derivatives_dot_D(self, t, dx, dy, dz, ox, oy, z, A, B, C):
        r2 = A * t * t + B * t + C
        return self._g(r2) - z, self._dgd(r2) * (2 * A * t + B) - dz

    # private
    def _g(self, r2: torch.Tensor) -> torch.Tensor:
        tmp = r2 * self.c
        sq = torch.sqrt((1.0 - (1.0 + self.k) * tmp * self.c).clamp_min(1e-12))
        total_surface = tmp / (1.0 + sq)
        higher_surface = 0.0
        if self.ai is not None:
            hs = 0.0
            for i in reversed(range(len(self.ai))):
                hs = r2 * hs + self.ai[i]
            higher_surface = hs * (r2 ** 2)
        if not isinstance(higher_surface, torch.Tensor):
            higher_surface = torch.as_tensor(higher_surface, device=r2.device, dtype=r2.dtype)
        return total_surface + higher_surface

    def _dgd(self, r2: torch.Tensor) -> torch.Tensor:
        alpha_r2 = (1.0 + self.k) * (self.c ** 2) * r2
        tmp = torch.sqrt((1.0 - alpha_r2).clamp_min(1e-12))
        total_derivative = self.c * (1.0 + tmp - 0.5 * alpha_r2) / (tmp * (1.0 + tmp) ** 2)
        higher_derivative = 0.0
        if self.ai is not None:
            hd = 0.0
            for i in reversed(range(len(self.ai))):
                hd = r2 * hd + (i + 2) * self.ai[i]
            higher_derivative = hd
        if not isinstance(higher_derivative, torch.Tensor):
            higher_derivative = torch.as_tensor(higher_derivative, device=r2.device, dtype=r2.dtype)
        return total_derivative + higher_derivative
