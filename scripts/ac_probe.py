"""Calibrate frame conventions against the running game (Windows, AC on track).

Drive the car YOURSELF for ~30 s: some straight running, a few left and right
corners. The probe records shared memory and estimates:
  heading_sign / heading_offset  (physics.heading vs. direction of travel)
  yaw_rate_sign                  (localAngularVel[1] vs. d(heading)/dt)
  sign_z / swap_sides            (track spline vs. where the car actually is)
  steer_sign                     (steerAngle sign vs. turning direction)
and writes configs/ac_calibration.yaml.

    python scripts/ac_probe.py configs/vallelunga_club_f2004.yaml --seconds 30
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from acopt.ac.shared_memory import ACSharedMemory, wstr  # noqa: E402
from acopt.session import load_track  # noqa: E402
from acopt.track import wrap  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("config")
ap.add_argument("--seconds", type=float, default=30)
a = ap.parse_args()
cfgp = Path(a.config); root = cfgp.parent.parent
cfg = yaml.safe_load(cfgp.read_text())

sm = ACSharedMemory()
st = sm.static()
print("car:", wstr(st.carModel), " track:", wstr(st.track))
rows, t0, last = [], time.perf_counter(), None
print(f"recording {a.seconds:.0f} s - drive normally...")
while time.perf_counter() - t0 < a.seconds:
    ph, gr = sm.physics(), sm.graphics()
    if ph.packetId != last:
        last = ph.packetId
        rows.append((time.perf_counter(), ph.heading, ph.velocity[0], ph.velocity[2],
                     ph.localAngularVel[1], ph.steerAngle, gr.carCoordinates[0], gr.carCoordinates[2],
                     ph.speedKmh))
    time.sleep(0.002)
R = np.array(rows)
t, hd, vx, vz, wy, st_, X, Z, kmh = R.T
mv = kmh > 30
cal = {}
for sz in (-1.0, 1.0):
    trav = np.arctan2(sz * vz, vx)
    for hs in (-1.0, 1.0):
        off = np.angle(np.mean(np.exp(1j * (trav[mv] - hs * hd[mv]))))
        err = np.abs(wrap(trav[mv] - (hs * hd[mv] + off)))
        if "err" not in cal or np.median(err) < cal["err"]:
            cal.update(err=float(np.median(err)), sign_z=sz, heading_sign=hs, heading_offset=float(off))
psi = cal["heading_sign"] * hd + cal["heading_offset"]
dpsi = np.gradient(np.unwrap(psi), t)
cal["yaw_rate_sign"] = float(np.sign(np.corrcoef(dpsi[mv], wy[mv])[0, 1]))
cal["steer_sign"] = float(np.sign(np.corrcoef(dpsi[mv], st_[mv])[0, 1]))
print(f"heading fit median error {np.degrees(cal.pop('err')):.2f} deg (should be < ~3)")

# track check: is the car inside the spline's corridor, and on the expected side?
tc = dict(cfg["track"]); tc["sign_z"] = cal["sign_z"]
tr = load_track(tc, root)
ns, idx = [], None
for x, z in zip(X[mv][::10], Z[mv][::10]):
    s, n, _, idx = tr.to_frenet(x, cal["sign_z"] * z, hint=idx)
    ns.append((n, *tr.bounds_at(s)))
ns = np.array(ns)
inside = np.mean((ns[:, 0] < ns[:, 1]) & (-ns[:, 0] < ns[:, 2]))
print(f"{100*inside:.0f}% of samples inside the track corridor (expect ~100% if you stayed on track)")
cal.update(brake_gamma=1.0, stale_s=0.1)
out = root / "configs/ac_calibration.yaml"
out.write_text(yaml.safe_dump(cal, sort_keys=False))
print("wrote", out, cal)
print("Also set track.sign_z in your session config to", cal["sign_z"])
