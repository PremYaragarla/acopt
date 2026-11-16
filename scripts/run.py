"""Run the controller. Same script for the simulator and Assetto Corsa:

    python scripts/run.py configs/sim_test.yaml --laps 6
    python scripts/run.py configs/vallelunga_club_f2004.yaml --laps 10

While it runs, tune live from another terminal (scripts/tune.py) or just edit
and save the profile YAML named in the config.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from acopt.session import Session  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("config")
ap.add_argument("--laps", type=int, default=5)
ap.add_argument("--profile", help="override the profile YAML")
ap.add_argument("--no-live", action="store_true")
ap.add_argument("--no-learn", action="store_true")
ap.add_argument("--dt", type=float, default=0.01, help="control tick [s]")
a = ap.parse_args()

ov = {}
if a.profile:
    ov["profile"] = a.profile
if a.no_learn:
    ov["learning"] = {"enabled": False}
S = Session(a.config, ov)
print("[run] Ctrl+C to stop (vJoy is centred on exit)")
try:
    times = S.run(laps=a.laps, live=not a.no_live, dt=a.dt)
except KeyboardInterrupt:
    times = S.lap_times
print("lap times:", times)
