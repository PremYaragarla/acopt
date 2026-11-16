"""Fit vehicle parameters to logged laps (from scripts/run.py logs).

    python scripts/identify.py configs/vallelunga_club_f2004.yaml out/logs_ac/<session>/lap00*.csv --save configs/f2004_identified.yaml
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from acopt.telemetry import load_lap  # noqa: E402
from acopt.sysid import identify, prepare, DEFAULT_FIT  # noqa: E402
from acopt.vehicle import VehicleParams  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("config")
ap.add_argument("logs", nargs="+")
ap.add_argument("--fit", default=",".join(DEFAULT_FIT))
ap.add_argument("--save")
a = ap.parse_args()
cfgp = Path(a.config); root = cfgp.parent.parent
cfg = yaml.safe_load(cfgp.read_text())
p = VehicleParams.from_yaml(root / cfg["vehicle"])
XB, Y = [], []
for f in a.logs:
    x, y = prepare(load_lap(f))
    XB.append(x); Y.append(y)
p2 = identify(p, np.vstack(XB), np.vstack(Y), fit=tuple(a.fit.split(",")))
if a.save:
    p2.to_yaml(a.save)
    print("saved", a.save)
