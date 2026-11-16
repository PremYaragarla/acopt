"""Compute the minimum-lap-time line + controls offline and plot it.

    python scripts/plan_line.py configs/vallelunga_club_f2004.yaml [--ds 3] [--png out/line.png]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from acopt.planner import plan_min_lap_time  # noqa: E402
from acopt.session import load_track  # noqa: E402
from acopt.vehicle import VehicleParams  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("config")
ap.add_argument("--ds", type=float)
ap.add_argument("--png", default=None)
a = ap.parse_args()
cfgp = Path(a.config)
root = cfgp.parent.parent
cfg = yaml.safe_load(cfgp.read_text())
p = VehicleParams.from_yaml(root / cfg["vehicle"])
tr = load_track(cfg["track"], root)
print(tr.summary())
rc = cfg.get("raceline", {})
rl = plan_min_lap_time(tr, p, ds=a.ds or rc.get("ds", 4.0), margin=rc.get("margin", 0.4),
                       window_m=rc.get("window_m"), overlap_m=rc.get("overlap_m", 600.0))
rl.meta["track_length"] = tr.length
out = root / rc.get("cache", f"out/{tr.name}_{p.name}.npz")
out.parent.mkdir(parents=True, exist_ok=True)
rl.save(out)
print(f"saved {out}  lap {rl.lap_time:.3f} s  vmax {rl.col('vx').max()*3.6:.0f} km/h  "
      f"vmin {rl.col('vx').min()*3.6:.0f} km/h")

png = a.png or str(out.with_suffix(".png"))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, ax = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.1, 1]})
s = tr.s
for side in (tr.w_left, -tr.w_right):
    ax[0].plot(*tr.to_world(s, side), color="0.4", lw=1)
X, Y = rl.world(tr)
sc = ax[0].scatter(X, Y, c=rl.col("vx") * 3.6, s=6, cmap="viridis")
fig.colorbar(sc, ax=ax[0], label="km/h")
ax[0].set_aspect("equal"); ax[0].set_title(f"{tr.name}: optimal line, {rl.lap_time:.2f} s")
ax[1].plot(rl.s, rl.col("vx") * 3.6, label="speed km/h")
ax[1].plot(rl.s, rl.col("T") * 100, label="throttle(+)/brake(-) %")
ax[1].plot(rl.s, np.degrees(rl.col("delta")) * 5, label="steer deg x5")
ax[1].set_xlabel("s [m]"); ax[1].legend(); ax[1].grid(alpha=0.3)
fig.tight_layout(); fig.savefig(png, dpi=110)
print("plot:", png)
