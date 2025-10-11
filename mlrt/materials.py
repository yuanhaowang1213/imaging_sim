# materials.py — Constant-index default; simple A + B/λ² if dispersion=True
from __future__ import annotations
import math
import numpy as np
import torch
from typing import Tuple

class Material:
    """Optical material with optional dispersion.

    If dispersion=False (default), n(λ) is constant (A), i.e., B=0.
    If dispersion=True, uses a simple Cauchy-like form: n(λ) = A + B / λ²,
    with (A, B) estimated from (n_D, V) at the D-line (587.5618 nm) and Abbe number V.
    """
    def __init__(self, name: str | None = None, dispersion: bool = True) -> None:
        self.name = "vacuum" if name is None else name.lower()
        self.dispersion = dispersion
        # minimal table; extend as needed
        self.MATERIAL_TABLE = {
            "vacuum": [1.0, math.inf],
            "air": [1.000293, math.inf],
            "occluder": [1.0, math.inf],  # for aperture
            "bk7": [1.51680, 64.17],

            "sk1":        [1.61030,  56.712],
            "sk16":       [1.62040,  60.306],
            "ssk4":       [1.61770,  55.116],
            "f15":        [1.60570,  37.831],

        }
        self.A, self.B = self._lookup_material()
        if not self.dispersion:
            self.B = 0.0

    def ior(self, wavelength: torch.Tensor | float) -> torch.Tensor | float:
        """Return index of refraction at wavelength (nm)."""
        if not self.dispersion:
            return self.A if not torch.is_tensor(wavelength) else torch.as_tensor(
                self.A, device=getattr(wavelength, 'device', None)
            )
        # dispersion on
        wl2 = wavelength ** 2 if torch.is_tensor(wavelength) else float(wavelength) ** 2
        return self.A + self.B / wl2

    @staticmethod
    def nV_to_AB(n: float, V: float) -> Tuple[float, float]:
        def ivs(a: float) -> float: return 1.0 / (a * a)
        C, D, F = 656.2725, 587.5618, 486.1327  # C, D, F spectral lines
        if V == 0.0 or math.isinf(V):
            return n, 0.0
        B = (n - 1.0) / V / (ivs(F) - ivs(C))
        A = n - B * ivs(D)
        return A, B

    def _lookup_material(self) -> Tuple[float, float]:
        out = self.MATERIAL_TABLE.get(self.name)
        if isinstance(out, list):
            n, V = out
        else:
            # allow literal like "1.5168/0" → n/V
            tmp = self.name.split('/')
            n, V = float(tmp[0]), float(tmp[1])
        return self.nV_to_AB(n, V)

    def __repr__(self) -> str:
        return f"Material(name={self.name}, A={self.A:.6f}, B={self.B:.6e}, dispersion={self.dispersion})"
