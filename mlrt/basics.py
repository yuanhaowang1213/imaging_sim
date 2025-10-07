# basics.py — Minimal, typed, and device-safe primitives
from __future__ import annotations
import torch
import numpy as np
from typing import Iterable, Tuple
def rodrigues_rotation_matrix(k: torch.Tensor, theta: torch.Tensor | float) -> torch.Tensor:
    """Rodrigues rotation for axis k (3,) and angle theta (rad)."""
    kx, ky, kz = k[0], k[1], k[2]
    K = torch.tensor([[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]], device=k.device, dtype=k.dtype)
    if not torch.is_tensor(theta):
        theta = torch.tensor(theta, device=k.device, dtype=k.dtype)
    I = torch.eye(3, device=k.device, dtype=k.dtype)
    return I + torch.sin(theta) * K + (1.0 - torch.cos(theta)) * (K @ K)
class PrettyPrinter:
    def __str__(self) -> str:
        lines = [self.__class__.__name__ + ":"]
        for key, val in vars(self).items():
            if isinstance(val, (list, tuple)):
                for i, v in enumerate(val):
                    lines += f"{key}[{i}]: {v}".split("\n")
            elif isinstance(val, dict):
                pass
            elif key == key.upper() and len(key) > 5:
                # likely constants; skip to keep output short
                pass
            else:
                lines += f"{key}: {val}".split("\n")
        return "\n    ".join(lines)

    def to(self, device: torch.device = torch.device("cpu")) -> None:
        for key, val in vars(self).items():
            if torch.is_tensor(val):
                setattr(self, key, val.to(device))
            elif isinstance(val, PrettyPrinter):
                val.to(device)
            elif isinstance(val, (list, tuple)):
                lst = list(val)
                for i, v in enumerate(lst):
                    if torch.is_tensor(v):
                        lst[i] = v.to(device)
                    elif isinstance(v, PrettyPrinter):
                        v.to(device)
                setattr(self, key, type(val)(lst))

class Transformation(PrettyPrinter):
    """Rigid transformation with rotation matrix R and translation t."""
    def __init__(self, R: torch.Tensor, t: torch.Tensor) -> None:
        self.R = R
        self.t = t

    def transform_point(self, o: torch.Tensor) -> torch.Tensor:
        # o: (..., 3)
        return torch.squeeze(self.R @ o[..., None]) + self.t

    def transform_vector(self, d: torch.Tensor) -> torch.Tensor:
        # d: (..., 3)
        return torch.squeeze(self.R @ d[..., None])

    def transform_ray(self, ray: "Ray") -> "Ray":
        o = self.transform_point(ray.o)
        d = self.transform_vector(ray.d)
        return Ray(o, d, ray.wavelength, device=ray.o.device)

    def inverse(self) -> "Transformation":
        RT = self.R.T
        return Transformation(RT, -RT @ self.t)

class Ray(PrettyPrinter):
    """Geometric ray with origin o, normalized direction d, wavelength (nm).

    Attributes
    -----------
    o : torch.Tensor [..., 3]
    d : torch.Tensor [..., 3] (normalized)
    wavelength : torch.Tensor or float (nm)
    mint / maxt : float travel bounds (mm)
    """
    def __init__(
        self,
        o: torch.Tensor,
        d: torch.Tensor,
        wavelength: torch.Tensor | float,
        device: torch.device = torch.device("cpu"),
        *,
        normalize_dir: bool = True,
    ) -> None:
        self.o = o.to(device)
        self.d = d.to(device)
        if normalize_dir:
            self.d = self.d / torch.linalg.norm(self.d, dim=-1, keepdim=True).clamp_min(1e-12)
        # scalar or vector wavelength supported
        if torch.is_tensor(wavelength):
            self.wavelength = wavelength.to(device)
        else:
            self.wavelength = (
                torch.full((self.o.shape[0],), float(wavelength), device=device)
                if self.o.ndim > 1 else
                torch.tensor(float(wavelength), device=device)
            )
        self.mint = 1e-5  # [mm]
        self.maxt = 1e5   # [mm]

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return self.o + t[..., None] * self.d
