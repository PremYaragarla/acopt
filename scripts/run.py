"""Run the controller headless (no dashboard). Same script for the simulator and
Assetto Corsa:

    python scripts/run.py configs/sim_test.yaml --laps 6
    python scripts/run.py configs/vallelunga_club_f2004.yaml --laps 10

For the web dashboard use scripts/ui.py instead. While this runs you can tune live
from another terminal (scripts/tune.py) or edit and save the profile YAML.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from acopt.profile import ProfileError  # noqa: E402
from acopt.session import ConfigError, Session  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("config")
ap.add_argument("--laps", type=int, default=5)
ap.add_argument("--profile", help="override the profile YAML")
ap.add_argument("--no-live", action="store_true")
ap.add_argument("--no-learn", action="store_true")
ap.add_argument("--dt", type=float, default=0.01, help="control tick [s]")
ap.add_argument("--pace", type=float, default=None, help="simulator speed, x real time (0 = flat out)")
a = ap.parse_args()

ov = {}
if a.profile:
    ov["profile"] = a.profile
if a.no_learn:
    ov["learning"] = {"enabled": False}
try:
    S = Session(a.config, ov)
except (ConfigError, ProfileError) as e:
    sys.exit(f"error: {e}")
print("[run] Ctrl+C to stop (controls are centred on exit)")
try:
    times = S.run(laps=a.laps, live=not a.no_live, dt=a.dt, pace=a.pace)
except KeyboardInterrupt:
    times = S.lap_times
finally:
    S.close()
print("lap times:", times)
if S.lap_times:
    print("logs + session.json:", S.out_dir)
