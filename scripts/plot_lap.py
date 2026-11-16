"""Plot a logged lap: path vs. reference, speed, sideslip tracking, controls.

    python scripts/plot_lap.py configs/sim_test.yaml out/logs_sim/lap001.npy [--png out/lap.png]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from acopt.session import load_track  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("config")
ap.add_argument("log")
ap.add_argument("--png")
a = ap.parse_args()
cfgp = Path(a.config); root = cfgp.parent.parent
tr = load_track(yaml.safe_load(cfgp.read_text())["track"], root)
L = list(np.load(a.log, allow_pickle=True))
g = lambda k: np.array([l.get(k, np.nan) for l in L], dtype=float)
s, n, nref, vx, vref = g("s"), g("n"), g("n_ref"), g("vx"), g("v_ref")
beta, bref, mode = np.degrees(g("beta")), np.degrees(g("beta_ref")), g("mode")
order = np.argsort(s)

fig = plt.figure(figsize=(15, 8))
ax0 = fig.add_subplot(1, 2, 1)
for side in (tr.w_left, -tr.w_right):
    ax0.plot(*tr.to_world(tr.s, side), color="0.5", lw=1)
ax0.plot(*tr.to_world(s[order], nref[order]), "--", color="0.2", lw=1, label="reference")
sc = ax0.scatter(*tr.to_world(s, n), c=np.abs(beta), s=4, cmap="inferno", vmin=0, vmax=25)
fig.colorbar(sc, ax=ax0, label="|sideslip| deg", shrink=0.7)
ax0.set_aspect("equal"); ax0.legend(loc="lower right"); ax0.set_title(Path(a.log).name)

ax1 = fig.add_subplot(3, 2, 2)
ax1.plot(s[order], vref[order] * 3.6, "k--", lw=1, label="ref")
ax1.plot(s[order], vx[order] * 3.6, lw=1, label="actual"); ax1.set_ylabel("km/h"); ax1.legend()
ax2 = fig.add_subplot(3, 2, 4, sharex=ax1)
ax2.plot(s[order], bref[order], "k--", lw=1, label="ref")
ax2.plot(s[order], beta[order], lw=1, label="actual")
ax2.fill_between(s[order], -30, 30, where=mode[order] > 0.5, color="C1", alpha=0.12, label="drift mode")
ax2.set_ylim(-30, 30); ax2.set_ylabel("sideslip deg"); ax2.legend(loc="upper left")
ax3 = fig.add_subplot(3, 2, 6, sharex=ax1)
ax3.plot(s[order], np.degrees(g("delta_cmd"))[order], lw=1, label="steer deg (road wheel)")
ax3.plot(s[order], 20 * g("T_cmd")[order], lw=1, label="throttle/brake x20")
ax3.set_xlabel("s [m]"); ax3.legend(loc="upper left")
for ax in (ax1, ax2, ax3):
    ax.grid(alpha=0.3)
fig.tight_layout()
out = a.png or str(Path(a.log).with_suffix(".png"))
fig.savefig(out, dpi=110)
print("saved", out)
