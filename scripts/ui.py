"""Web dashboard: watch the car, start/stop runs, tune drift segments live, download data.

    python scripts/ui.py                                   # simulator (no game needed)
    python scripts/ui.py configs/vallelunga_club_f2004.yaml   # Assetto Corsa

Then open the printed address in a browser. Ctrl+C here stops everything and
centres the controls.
"""
import argparse
import sys
import time
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from acopt.dashboard import App, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("config", nargs="?", default=str(ROOT / "configs" / "sim_test.yaml"))
ap.add_argument("--profile", help="override the profile YAML")
ap.add_argument("--port", type=int, default=8765)
ap.add_argument("--host", default="127.0.0.1",
                help="0.0.0.0 to reach the dashboard from another device (no password: trusted networks only)")
ap.add_argument("--no-browser", action="store_true")
ap.add_argument("--autostart", type=int, metavar="LAPS", help="start driving as soon as the session is ready")
a = ap.parse_args()

ov = {"env": {"pace": 1.0}}          # simulator runs at real time so you can watch it
if a.profile:
    ov["profile"] = a.profile
app = App(a.config, ov).build_async()
srv, url = serve(app, a.host, a.port)
print(f"[ui] dashboard: {url}   (Ctrl+C to quit)")
if not a.no_browser:
    try:
        webbrowser.open(url)
    except Exception:  # noqa: BLE001
        pass
try:
    started = False
    while True:
        time.sleep(0.25)
        if a.autostart and not started and app.session is not None:
            started = True
            print("[ui] autostart:", app.start_run(a.autostart))
except KeyboardInterrupt:
    print("\n[ui] stopping...")
finally:
    app.shutdown()
    srv.shutdown()
