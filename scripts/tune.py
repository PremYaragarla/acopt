"""Live tuning client.

    python scripts/tune.py status
    python scripts/tune.py drift T3 --beta 22 [--start 940 --end 1110] [--line 0.5] [--speed-scale 0.95]
    python scripts/tune.py race  T3              # turn a segment back into race mode
    python scripts/tune.py remove T3
    python scripts/tune.py weights drift beta=2.0 n=1.0
    python scripts/tune.py speed 0.95
    python scripts/tune.py load profiles/drift_demo.yaml
    python scripts/tune.py pause | resume | stop
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from acopt.live import send  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("action")
ap.add_argument("args", nargs="*")
ap.add_argument("--beta", type=float)
ap.add_argument("--start", type=float)
ap.add_argument("--end", type=float)
ap.add_argument("--line")
ap.add_argument("--blend", type=float)
ap.add_argument("--speed-scale", type=float)
ap.add_argument("--port", type=int, default=47800)
a = ap.parse_args()

act = a.action
if act in ("drift", "race"):
    cmd = {"cmd": "segment", "name": a.args[0], "mode": act}
    for k, v in (("beta_deg", a.beta), ("s_start", a.start), ("s_end", a.end),
                 ("blend_m", a.blend), ("speed_scale", a.speed_scale)):
        if v is not None:
            cmd[k] = v
    if a.line is not None:
        cmd["line"] = a.line if a.line in ("race", "center") else float(a.line)
elif act == "remove":
    cmd = {"cmd": "remove", "name": a.args[0]}
elif act == "weights":
    cmd = {"cmd": "weights", "mode": a.args[0]}
    cmd.update({kv.split("=")[0]: float(kv.split("=")[1]) for kv in a.args[1:]})
elif act == "speed":
    cmd = {"cmd": "speed_scale", "value": float(a.args[0])}
elif act == "load":
    cmd = {"cmd": "load_profile", "path": str(Path(a.args[0]).resolve())}
else:
    cmd = {"cmd": act}
print(json.dumps(send(cmd, port=a.port), indent=1))
