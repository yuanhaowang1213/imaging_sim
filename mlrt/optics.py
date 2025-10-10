# optics.py — Lensgroup + Aspheric surface + tracing, sampling, and PSF rendering
# modified from https://github.com/vccimaging/DiffOptics/tree/main/diffoptics MIT License
from __future__ import annotations
import math
import numpy as np
import torch
import pathlib
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

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
        dispersion : bool = True,
        device: torch.device = torch.device("cpu"),
        
    ) -> None:
        self.origin = (origin if origin is not None else torch.zeros(3)).to(device)
        self.shift = (shift if shift is not None else torch.zeros(3)).to(device)
        self.theta = (theta if theta is not None else torch.zeros(3)).to(device)
        self.device = device

        # lens group
        self.surfaces: List[Surface] = []
        self.materials: List[Material] = []

        self.dispersion = dispersion

        # sensor
        self.pixel_size = float(pixel_size)  # mm
        self.film_size = list(film_size) if film_size is not None else [512, 512]

        # transformation
        self.to_world = self._compute_transformation()
        self.to_object = self.to_world.inverse()

    # ---- setup / IO ----
    def load_file(self, filename: pathlib.Path) -> None:
        self.surfaces, self.materials, self.r_last, d_last, self.aperture_ind = self.read_lensfile(str(filename), self.dispersion)
        # sensor plane z
        self.d_sensor = d_last + self.surfaces[-1].d
        self._sync()

    def update(self, _x: float = 0.0, _y: float = 0.0) -> None:
        self.to_world = self._compute_transformation(_x, _y)
        self.to_object = self.to_world.inverse()

    def _sync(self) -> None:
        for s in self.surfaces:
            s.to(self.device)
    
    # define the aperture_ind manually
        # self.aperture_ind = self._find_aperture()

    # def _find_aperture(self) -> int | None:
    #     for i in range(len(self.surfaces) - 1):
    #         if self.materials[i].A < 1.0003 and self.materials[i + 1].A < 1.0003:
    #             return i
    #     return None

    def _compute_transformation(self, _x: float = 0.0, _y: float = 0.0, _z: float = 0.0) -> Transformation:
        R = (
            rodrigues_rotation_matrix(torch.tensor([1, 0, 0], device=self.device, dtype=torch.float32), torch.deg2rad(self.theta[0] + _x))
            @ rodrigues_rotation_matrix(torch.tensor([0, 1, 0], device=self.device, dtype=torch.float32), torch.deg2rad(self.theta[1] + _y))
            @ rodrigues_rotation_matrix(torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32), torch.deg2rad(self.theta[2] + _z))
        )
        t = self.origin + R @ self.shift
        return Transformation(R, t)

    @staticmethod
    def read_lensfile(filename: str, dispersion: bool) -> Tuple[List["Surface"], List["Material"], float, float]:
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
                materials.append(Material(name=ls[4], dispersion=dispersion))

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
                    aperture_ind = len(surfaces) -1
                elif surface_type == "I":  # sensor plane record
                    d_total -= d
                    ds.pop()
                    materials.pop()
                    r_last = r
                    d_last = d
        return surfaces, materials, r_last, d_last,aperture_ind

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
        if(self.film_size[0]%2 == 0):
            u -= 0.5
        if(self.film_size[1]%2 == 0):
            v -= 0.5
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
    # off axis sampling
    def sample_offaxis_point_axis(
        self,
        D_mm: float,
        wavelength: float,
        N: int,
        x_off_mm: float,
        y_off_mm: float,
        filter_to_stop: bool = True,
    ) -> "Ray":
        """Off-axis point at (x_off,y_off,z_src). Cone axis = chief ray (toward stop center)."""
        device = self.device
        stop_i = getattr(self, "aperture_ind", None)
        stop = self.surfaces[stop_i] if stop_i is not None else self.surfaces[0]
        z_stop, r_stop = float(stop.d), float(stop.r)

        # source position
        z_front = float(self.surfaces[0].d)
        z_src = z_front - float(D_mm)
        o0 = torch.tensor([x_off_mm, y_off_mm, z_src], dtype=torch.float32, device=device)

        # chief ray axis (toward stop center)
        c = torch.tensor([0.0, 0.0, z_stop], dtype=torch.float32, device=device) - o0
        dist = float(torch.linalg.norm(c))
        axis = c / (dist + 1e-12)

        # cone half-angle so that cone just touches stop rim
        theta_max = math.atan2(r_stop, max(dist, 1e-9))

        # angle-uniform grid on that cone
        n_th = max(1, int(math.sqrt(max(1, N))))
        n_ph = max(1, int(math.ceil(N / n_th)))
        TH = torch.linspace(0.0, theta_max, steps=n_th, device=device)
        PH = torch.linspace(0.0, 2 * math.pi, steps=n_ph + 1, device=device)[:-1]
        TH, PH = torch.meshgrid(TH, PH, indexing="ij")

        # local orthonormal basis (u, v, axis)
        tmp = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=device)
        if torch.abs((tmp * axis).sum()) > 0.9:
            tmp = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=device)
        u = torch.cross(axis, tmp); u = u / (torch.linalg.norm(u) + 1e-12)
        v = torch.cross(axis, u)

        # directions on the cone
        dx = torch.sin(TH) * torch.cos(PH)
        dy = torch.sin(TH) * torch.sin(PH)
        dz = torch.cos(TH)
        d = (u[None, None, :] * dx[..., None] +
            v[None, None, :] * dy[..., None] +
            axis[None, None, :] * dz[..., None]).reshape(-1, 3)

        # build rays
        o = o0.repeat(d.shape[0], 1)
        wav = torch.full((d.shape[0],), float(wavelength), dtype=torch.float32, device=device)
        ray = Ray(o, d, wav, device=device)

        # optional pre-filter: keep only rays that pass the stop
        if filter_to_stop and stop_i is not None:
            _, valid = self.trace(ray, stop_ind=stop_i)
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
    
    def best_focus_D2(
        self,
        D_mm,
        lam: float = 500.0,
        N: int = 5000,
        D2_guess: float | None = None,
        span: float = 5.0,
        steps: int = 21,
        use_spot: bool = True,
        # new: fine pass params
        span_fine: float = 0.3,
        steps_fine: int = 25,
    ) -> float:
        """
        Two-stage search for best D2 (surface-2 → sensor distance) that minimizes blur.

        Stage 1 (coarse): sweep D2 in [D2_guess ± span] with `steps` samples.
        Stage 2 (fine):   sweep D2 in [D2_coarse_best ± span_fine] with `steps_fine` samples.

        If use_spot=True: metric = mean spot radius from ray hit points (no pixel grid).
        If use_spot=False: render to pixels, metric = EE50 radius (mm).
        """
        import numpy as np

        # surface-2 z
        z2 = float(self.surfaces[1].d)

        # default guess: current sensor offset
        if D2_guess is None:
            D2_guess = self.d_sensor - z2

        # keep original sensor z and restore later
        _orig_d_sensor = float(self.d_sensor)

        def _metric_for(D2_val: float) -> float | None:
            """Compute blur metric at a given D2. Return None if no rays hit sensor."""
            self.d_sensor = z2 + float(D2_val)
            rays = self.sample_ray_from_point(D_mm=D_mm, wavelength=lam, N=N, filter_to_stop=True)

            if use_spot:
                p = self.trace_to_sensor(rays, ignore_invalid=True).cpu().numpy()
                if p.size == 0:
                    return None
                xy = p[:, :2]
                c = xy.mean(axis=0, keepdims=True)
                r = np.linalg.norm(xy - c, axis=1)
                return float(r.mean())  # or RMS: float(np.sqrt((r**2).mean()))
            else:
                I = self.render(rays)
                img = I.detach().cpu().float().numpy()
                if img.max() <= 0:
                    return None
                # EE50 radius (center = brightest pixel)
                peak = np.unravel_index(np.argmax(img), img.shape)
                yy, xx = np.indices(img.shape)
                dx = (xx - peak[0]) * self.pixel_size
                dy = (yy - peak[1]) * self.pixel_size
                r = np.sqrt(dx * dx + dy * dy).ravel()
                w = (img.ravel() / (img.sum() + 1e-12))
                order = np.argsort(r)
                cdf = np.cumsum(w[order])
                idx = np.searchsorted(cdf, 0.5)
                return float(r[order][min(idx, len(order) - 1)])

        # ---- Stage 1: coarse sweep ----
        coarse_grid = np.linspace(D2_guess - span, D2_guess + span, steps)
        best_val = float("inf")
        best_D2 = float(coarse_grid[0])

        for D2 in coarse_grid:
            m = _metric_for(D2)
            if m is None:
                continue
            if m < best_val:
                best_val = m
                best_D2 = float(D2)

        # ---- Stage 2: fine sweep around coarse best ----
        fine_grid = np.linspace(best_D2 - span_fine, best_D2 + span_fine, steps_fine)
        best_val_f = float("inf")
        best_D2_f = float(fine_grid[0])

        for D2 in fine_grid:
            m = _metric_for(D2)
            if m is None:
                continue
            if m < best_val_f:
                best_val_f = m
                best_D2_f = float(D2)

        return best_D2_f
    
    # psf metrics 
    def psf_metrics(self, I:torch.Tensor) -> dict:
        """
        Compute simple metrics: sum, centorid offset, RMS radius, EE50.
        All in mm.
        """
        img = I.detach().cpu().float().numpy()
        H, W = img.shape
        S = img.sum()
        if S <= 0:
            return dict(sum=0.0, cx_mm=0.0, cy_mm=0.0,
                        r_centroid_mm=float("nan"),
                        rms_radius_mm=float("nan"),
                        ee50_mm=float("nan"))

        # Axis convention: axis 0 = x (size H), axis 1 = y (size W)
        xs = (np.arange(H) - H/2 + 0.5) * self.pixel_size  # mm, pixel centers
        ys = (np.arange(W) - W/2 + 0.5) * self.pixel_size  # mm
        X, Y = np.meshgrid(xs, ys, indexing="ij")          # shapes (H,W), (H,W)

        # Energy-weighted centroid (mm)
        cx = float((img * X).sum() / S)
        cy = float((img * Y).sum() / S)

        # RMS radius about centroid (mm)
        r2 = (X - cx)**2 + (Y - cy)**2
        rms = float(np.sqrt((img * r2).sum() / S))

        # Encircled-energy radius at 50% (EE50), centered at centroid
        r = np.sqrt(r2).ravel()
        w = img.ravel()
        order = np.argsort(r)
        cdf = np.cumsum(w[order]) / S
        ee50 = float(r[order][np.searchsorted(cdf, 0.5)])

        return dict(
            sum=float(S),
            cx_mm=cx, cy_mm=cy,
            r_centroid_mm=float(np.hypot(cx, cy)),
            rms_radius_mm=rms,
            ee50_mm=ee50,
        )

    # plot function
    def plot_layout2d(
        self,
        ax : Optional[Axes] = None,
        n_samples: int = 100,
        color : str= "k",
        with_sensor:bool = True,
        show: bool= True,
        fname: str =None
    ) -> plt.Axes:
        """2D layout in z–x (y=0). Draw surfaces, aperture wedges, lens edges, and sensor line."""
        created = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
            created = True

        # helper: world transform then plot
        def _plot_zx(z: torch.Tensor, x: torch.Tensor, c: str):
            p = self.to_world.transform_point(
                torch.stack(
                    (x, torch.zeros_like(x, device=self.device), z), dim=-1
                )
            ).detach().cpu().numpy()
            ax.plot(p[..., 2], p[..., 0], c)

        # helper: draw aperture wedge at a surface
        def _draw_aperture(surface, c: str):
            N = 3
            d = surface.d
            R = surface.r
            L = 0.05 * R      # wedge length [mm]
            H = 0.15 * R      # wedge height [mm]

            # two short horizontal ticks
            z = torch.linspace(d - L, d + L, N, device=self.device)
            _plot_zx(z, -R * torch.ones_like(z), c)
            _plot_zx(z,  R * torch.ones_like(z), c)
            # two short vertical ticks
            z = d * torch.ones(N, device=self.device)
            _plot_zx(z, torch.linspace( R,  R + H, N, device=self.device), c)
            _plot_zx(z, torch.linspace(-R - H, -R, N, device=self.device), c)

        # draw surfaces / aperture markers
        if len(self.surfaces) == 1:
            _draw_aperture(self.surfaces[0], color)
        else:
            # draw each surface curve (skip drawing full curve for the aperture, only wedges)
            for i, s in enumerate(self.surfaces):
                # Air–Air interface → mark as aperture
                if i < len(self.materials) - 1:
                    if self.materials[i].A < 1.0003 and self.materials[i + 1].A < 1.0003:
                        _draw_aperture(s, color)
                        continue
                # regular surface curve
                r = torch.linspace(-s.r, s.r, max(n_samples, getattr(s, "APERTURE_SAMPLING", n_samples)), device=self.device)
                z = s.surface_with_offset(r, torch.zeros_like(r))
                _plot_zx(z, r, color)

            # draw lens outer edges (between air and glass radii)
            prev_surface = None
            for i, s in enumerate(self.surfaces):
                if self.materials[i].A < 1.0003:  # AIR
                    prev_surface = s
                else:
                    if prev_surface is None:
                        prev_surface = s
                    r_prev = prev_surface.r
                    r_cur  = s.r
                    sag_prev = prev_surface.surface_with_offset(torch.tensor(r_prev, device=self.device), torch.tensor(0.0, device=self.device))
                    sag_cur  = s.surface_with_offset(torch.tensor(r_cur,  device=self.device), torch.tensor(0.0, device=self.device))
                    z = torch.stack((sag_prev, sag_cur))
                    x = torch.tensor([ r_prev,  r_cur], device=self.device)
                    _plot_zx(z,  x, color)
                    _plot_zx(z, -x, color)
                    prev_surface = s

        # draw sensor line
        if with_sensor and hasattr(self, "d_sensor"):
            x_half = max(self.film_size) * self.pixel_size / 2.0  # [mm]
            z_s = float(self.d_sensor)
            ax.plot([z_s, z_s], [-x_half, x_half], "r--", lw=1.0, label="sensor")
            ax.legend(loc="best")

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("z [mm]")
        ax.set_ylabel("x [mm]")
        ax.set_title("Layout 2D")

        if created:
            if show: 
                plt.show()
            else: 
                if fname is not None:
                    from pathlib import Path
                    Path(fname).parent.mkdir(parents=True, exist_ok=True)
                    ax.figure.savefig(fname, dpi=300, bbox_inches="tight")
            plt.close(ax.figure)
        return ax

    def plot_raytraces_world( self,
                            oss: list[list[np.ndarray]],
                            ax: Optional[Axes] = None,
                            color: str = "C0",
                            show: bool = True,
                            fname: str = None):
        if ax is None:
            ax = self.plot_layout2d(show=False)
        for path in oss:
            if len(path) < 2: 
                continue
            p = np.asarray(path)              # already world coords
            ax.plot(p[:, 2], p[:, 0], color, lw=0.8, alpha=0.8)
        if show: 
            plt.show()
        else: 
            if fname is not None:
                from pathlib import Path
                Path(fname).parent.mkdir(parents=True, exist_ok=True)
                ax.figure.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close(ax.figure)
        return ax
    def plot_psf(self, I: torch.Tensor, fname: str =None,show: bool = True,
             normalize: bool = True, cmap: str = "gray", log:bool=False) -> None:
        img = I.detach().cpu().float()
        if normalize and img.numel() > 0 and img.max() > 0:
            img = img / img.max()

        H, W = self.film_size
        x_half = W * self.pixel_size / 2.0
        y_half = H * self.pixel_size / 2.0
        extent = [-x_half, x_half, -y_half, y_half]

        fig, ax = plt.subplots(figsize=(5, 5))
        if log:
            from matplotlib.colors import LogNorm
            if(img >0).any():
                pos_min = float(img[img > 0].min().item())
                eps = max(1e-12, pos_min * 0.5)
            else:
                eps = 1e-12
            arr = img.T.numpy().copy()
            arr[arr<=eps] = eps
            ax.imshow(arr, extent=extent, origin="lower", cmap=cmap, norm=LogNorm(vmin=eps, vmax=img.max() if img.max()>0 else 1))
        else:
            ax.imshow(img.T.numpy(), extent=extent, origin="lower", cmap=cmap)

        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_title("PSF at sensor")
        cbar = fig.colorbar(ax.images[0], ax=ax)
        cbar.set_label("normalized intensity")
        if show: 
            plt.show()
        else:
            if fname is not None:
                    from pathlib import Path
                    Path(fname).parent.mkdir(parents=True, exist_ok=True)
                    fig.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close(fig)
 
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
            higher_surface = torch.as_tensor(higher_surface, device=self.c.device, dtype=self.c.dtype)
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
