"""Turn a session's logs into a report: a PNG with the key plots and a text summary.

    python scripts/report.py                      # latest session of configs/sim_test.yaml
    python scripts/report.py configs/vallelunga_club_f2004.yaml
    python scripts/report.py configs/sim_test.yaml out/logs_sim/20260101_120000 --png my.png

Reads lapNNN.csv + session.json written by scripts/run.py or scripts/ui.py.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from acopt.session import ConfigError, load_config, load_track  # noqa: E402
from acopt.telemetry import lap_metrics, load_lap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("config", nargs="?", default=str(ROOT / "configs" / "sim_test.yaml"))
ap.add_argument("session_dir", nargs="?", help="folder with lapNNN.csv (default: the latest one)")
ap.add_argument("--png")
a = ap.parse_args()

try:
    cfg, root = load_config(a.config)
    track = load_track(cfg["track"], root)
except ConfigError as e:
    sys.exit(f"error: {e}")
if a.session_dir:
    d = Path(a.session_dir)
else:
    ptr = root / cfg.get("log_dir", "out/logs") / "latest.txt"
    if not ptr.is_file():
        sys.exit(f"error: no sessions recorded yet for this config ({ptr} missing). Run scripts/run.py or scripts/ui.py first.")
    d = Path(ptr.read_text().strip())
files = sorted(d.glob("lap[0-9]*.csv"))
if not files:
    sys.exit(f"error: no lapNNN.csv files in {d}")

laps = [load_lap(f) for f in files]
sj = d / "session.json"
if sj.is_file():
    mets = json.loads(sj.read_text())["laps"]
else:
    mets = [dict(lap_metrics(l, l[-1]["lap_t"], None, track.length), lap=i) for i, l in enumerate(laps)]
mets = mets[:len(laps)]
done = [i for i, m in enumerate(mets) if m.get("complete")] or list(range(len(laps)))
best = min(done, key=lambda i: mets[i]["lap_time"])
first = done[0]
g = lambda log, k, default=np.nan: np.array([l.get(k, default) for l in log], dtype=float)

# ---------- text summary ----------
lines = [f"Session {d.name}: {len(laps)} lap(s) on {track.name} ({track.length:.0f} m)", "",
         f"{'lap':>3} {'time s':>8} {'plan s':>8} {'top km/h':>8} {'line err m':>10} {'drift err deg':>13} {'interv.':>7} {'solve ms p95':>12}"]
for m in mets:
    de = m.get("drift_beta_err_rms_deg")
    lines.append(f"{m['lap']:>3} {m['lap_time']:>8.3f} {(m.get('plan_time') or float('nan')):>8.3f} {m['v_max_kmh']:>8.0f} "
                 f"{m['n_err_rms_m']:>10.2f} {('-' if de is None else format(de, '.2f')):>13} {m['interventions']:>7} "
                 f"{(m.get('solve_ms_p95') or float('nan')):>12.0f}" + ("" if m.get("complete") else "  (partial)"))
lines += ["", f"Best lap: {mets[best]['lap_time']:.3f} s (lap {mets[best]['lap']})"]
if len(done) > 1:
    lines.append(f"First to best: {mets[first]['lap_time'] - mets[best]['lap_time']:+.3f} s improvement")
lines.append(f"Supervisor interventions in total: {sum(m['interventions'] for m in mets)}")
summary = "\n".join(lines)
(d / "report.txt").write_text(summary + "\n")
print(summary)

# ---------- figure ----------
BLUE, ORANGE, AQUA, REF = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(4, 2, width_ratios=[1.05, 1], hspace=0.42, wspace=0.14)
B = laps[best]
s, n, nref = g(B, "s"), g(B, "n"), g(B, "n_ref")
order = np.argsort(s)

ax = fig.add_subplot(gs[:3, 0])
for side in (track.w_left, -track.w_right):
    ax.plot(*track.to_world(track.s, side), color="0.55", lw=1)
ax.plot(*track.to_world(s[order], nref[order]), "--", color=REF, lw=1, label="reference line")
sc = ax.scatter(*track.to_world(s, n), c=3.6 * g(B, "vx"), s=5, cmap="Blues", vmin=0)
fig.colorbar(sc, ax=ax, label="speed (km/h)", shrink=0.6, pad=0.01)
drift = g(B, "mode", 0.0) > 0.5
if drift.any():
    ax.scatter(*track.to_world(s[drift], n[drift]), s=22, facecolors="none", edgecolors=ORANGE, lw=0.6,
               label="drift segment")
ax.set_aspect("equal"); ax.axis("off"); ax.legend(loc="upper left", frameon=False)
ax.set_title(f"Best lap ({mets[best]['lap_time']:.3f} s, lap {mets[best]['lap']}): driven line coloured by speed", loc="left")

ax = fig.add_subplot(gs[3, 0])
x = [mets[i]["lap"] for i in done]
ax.plot(x, [mets[i]["lap_time"] for i in done], "-o", color=BLUE, lw=2, label="driven")
plan = [mets[i].get("plan_time") for i in done]
if all(p is not None for p in plan):
    ax.plot(x, plan, "--", color=AQUA, lw=1.5, label="predicted by the plan")
ax.set_xticks(x); ax.set_xlabel("lap"); ax.set_ylabel("lap time (s)"); ax.legend(frameon=False)
ax.set_title("Lap time by lap", loc="left"); ax.grid(alpha=0.25)


def trace(ax, key, scale, ref_key=None, ylabel=""):
    if ref_key:
        ax.plot(s[order], scale * g(B, ref_key)[order], "--", color=REF, lw=1, label="reference")
    if first != best:
        Fl = laps[first]; so = np.argsort(g(Fl, "s"))
        ax.plot(g(Fl, "s")[so], scale * g(Fl, key)[so], color=ORANGE, lw=1, label=f"lap {mets[first]['lap']}")
    ax.plot(s[order], scale * g(B, key)[order], color=BLUE, lw=1.4, label=f"lap {mets[best]['lap']} (best)")
    if drift.any():
        lo, hi = ax.get_ylim()
        ax.fill_between(s[order], lo, hi, where=drift[order], color=ORANGE, alpha=0.10, lw=0)
        ax.set_ylim(lo, hi)
    ax.set_ylabel(ylabel); ax.grid(alpha=0.25); ax.set_xlim(0, track.length)


ax1 = fig.add_subplot(gs[0, 1]); trace(ax1, "vx", 3.6, "v_ref", "speed (km/h)")
ax1.legend(frameon=False, ncol=3, loc="upper right"); ax1.set_title("Over lap distance (shaded: drift segments)", loc="left")
ax2 = fig.add_subplot(gs[1, 1], sharex=ax1); trace(ax2, "beta", 180 / np.pi, "beta_ref", "sideslip (deg)")
ax3 = fig.add_subplot(gs[2, 1], sharex=ax1)
ax3.axhline(0, color=REF, lw=1, ls="--")
ax3.plot(s[order], (n - nref)[order], color=BLUE, lw=1.4); ax3.set_ylabel("off line (m)"); ax3.grid(alpha=0.25)
ax4 = fig.add_subplot(gs[3, 1], sharex=ax1); trace(ax4, "T_cmd", 100, None, "throttle / brake (%)")
ax4.set_xlabel("distance from start line (m)")
out = Path(a.png) if a.png else d / "report.png"
fig.savefig(out, dpi=110, bbox_inches="tight")
print("\nreport:", out, "and", d / "report.txt")
