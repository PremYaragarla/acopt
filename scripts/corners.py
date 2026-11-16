"""List corners (s ranges, radius, direction) so you can place drift segments.

    python scripts/corners.py configs/vallelunga_club_f2004.yaml
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from acopt.drift import DriftEquilibrium  # noqa: E402
from acopt.session import load_track  # noqa: E402
from acopt.vehicle import VehicleModel, VehicleParams  # noqa: E402

cfgp = Path(sys.argv[1]); root = cfgp.parent.parent
cfg = yaml.safe_load(cfgp.read_text())
tr = load_track(cfg["track"], root)
de = DriftEquilibrium(VehicleModel(VehicleParams.from_yaml(root / cfg["vehicle"])))
print(tr.summary())
k, s = tr.kappa, tr.s
inside = np.abs(k) > 1 / 150
edges = np.flatnonzero(np.diff(inside.astype(int)))
if inside[0]:
    edges = np.r_[edges[-1], edges[:-1]] if len(edges) % 2 else edges
print(f"{'#':>2} {'s_start':>8} {'s_end':>8} {'dir':>5} {'Rmin':>6}  max drift angle")
for j, (a, b) in enumerate(zip(edges[::2], edges[1::2])):
    seg = slice(a + 1, b + 1)
    i = a + 1 + np.argmax(np.abs(k[seg]))
    R = 1 / k[i]
    bmax = np.degrees(de.max_feasible_beta(k[i]))
    print(f"{j+1:>2} {s[a+1]:8.0f} {s[b]:8.0f} {'left' if R > 0 else 'right':>5} {abs(R):6.0f}  {bmax:4.0f} deg")
