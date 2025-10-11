# plot_metrics.py
from __future__ import annotations
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

def _to_float_or_nan(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")

def read_metrics(csv_path: str | Path) -> list[dict]:
    """Load metrics.csv → list of dicts with numeric values where possible."""
    rows = []
    with open(csv_path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({k: _to_float_or_nan(v) for k, v in row.items()})
    return rows

def plot_metrics_from_csv(
    csv_path: str | Path,
    *,
    xkey: str | None = None,
    ykeys: list[str] = ["ee50_mm", "rms_radius_mm"],
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str = "mm",
    out_prefix: str | None = None,
) -> None:
    """
    Read a metrics.csv and plot:
      - If xkey is provided → 1D curves (each ykey vs xkey)
      - If xkey is None and file has x_off_mm & y_off_mm → 2D heatmaps

    Saves figures next to metrics.csv with PNG extension.
    """
    csv_path = Path(csv_path)
    out_dir = csv_path.parent
    out_prefix = out_prefix or csv_path.stem

    rows = read_metrics(csv_path)
    if not rows:
        print(f"[plot] no rows found in {csv_path}")
        return

    # 1D sweep
    if xkey is not None:
        xs = [row.get(xkey, np.nan) for row in rows]
        plt.figure(figsize=(6.0, 4.0))
        for yk in ykeys:
            ys = [row.get(yk, np.nan) for row in rows]
            plt.plot(xs, ys, marker="o", label=yk)
        plt.grid(True, ls=":", lw=0.7)
        plt.xlabel(xlabel or xkey)
        plt.ylabel(ylabel)
        if title: plt.title(title)
        if len(ykeys) >= 1: plt.legend()
        plt.tight_layout()
        out_file = out_dir / f"{out_prefix}_vs_{xkey}.png"
        plt.savefig(out_file, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[plot] saved {out_file}")
        return

    # 2D field heatmaps (requires x_off_mm & y_off_mm)
    has_field = ("x_off_mm" in rows[0]) and ("y_off_mm" in rows[0])
    if not has_field:
        print("[plot] No xkey provided and this CSV does not look like a 2D field grid.")
        return

    xs = sorted({row["x_off_mm"] for row in rows})
    ys = sorted({row["y_off_mm"] for row in rows})
    x_index = {v: i for i, v in enumerate(xs)}
    y_index = {v: i for i, v in enumerate(ys)}

    def _heatmap(vkey: str):
        Z = np.full((len(ys), len(xs)), np.nan, dtype=float)
        for row in rows:
            ix = x_index[row["x_off_mm"]]
            iy = y_index[row["y_off_mm"]]
            Z[iy, ix] = row.get(vkey, np.nan)

        plt.figure(figsize=(5.6, 4.6))
        im = plt.imshow(
            Z, origin="lower",
            extent=[min(xs), max(xs), min(ys), max(ys)],
            aspect="auto"
        )
        plt.colorbar(im, label=vkey)
        plt.xlabel("x_off_mm")
        plt.ylabel("y_off_mm")
        if title: plt.title(title + f" – {vkey}")
        plt.tight_layout()
        out_file = out_dir / f"{out_prefix}_{vkey}_field_heatmap.png"
        plt.savefig(out_file, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[plot] saved {out_file}")

    for v in ykeys:
        _heatmap(v)
# sweep N
plot_metrics_from_csv("out/sweep_N/metrics.csv",
                      xkey="N",
                      ykeys=[ "rms_radius_mm"],
                      title="PSF metrics vs N",
                      xlabel="N (rays)")

# sweep lambda
plot_metrics_from_csv("out/sweep_lambda/metrics.csv",
                      xkey="lambda_nm",
                      ykeys=[ "rms_radius_mm","ee50_mm"],
                      title="PSF metrics vs Wavelength",
                      xlabel="Wavelength (nm)")
# sweep 1D offaxis
plot_metrics_from_csv("out/offaxis/metrics.csv",
                      xkey="x_off_mm",
                      ykeys=["rms_radius_mm"],
                      title="Off-axis PSF metrics",
                      xlabel="Source x-offset (mm)")

# sweep d2 
plot_metrics_from_csv("out/sweep_D2/metrics.csv",
                      xkey="D2",
                      ykeys=[ "rms_radius_mm","ee50_mm"],
                      title="Through-focus metrics",
                      xlabel="D2 (mm)")
# sweep OD (aperture diameter)
plot_metrics_from_csv(
    "out/sweep_OD/metrics.csv",
    xkey="OD",
    ykeys=[ "rms_radius_mm","ee50_mm"],
    title="PSF metrics vs Aperture Diameter",
    xlabel="Aperture OD (mm)"
)

## Double Gaussian
# sweep lambda (aperture diameter)
plot_metrics_from_csv(
    "out_gaussian/sweep_lambda/metrics.csv",
    xkey="lambda_nm",
    ykeys=[ "rms_radius_mm"],
    title="PSF metrics vs Aperture Diameter",
    xlabel="Aperture OD (mm)"
)

# sweep 1D offaxis
plot_metrics_from_csv(
    "out_gaussian/offaxis/metrics.csv",
    xkey="x_off_mm",
    ykeys=[ "rms_radius_mm"],
    title="PSF metrics vs Aperture Diameter",
    xlabel="Aperture OD (mm)"
)

# # sweep field
# plot_metrics_from_csv("out/field_grid/metrics.csv",
#                       xkey=None,  # 
#                       ykeys=[ "rms_radius_mm","ee50_mm"],
#                       title="Field map")
