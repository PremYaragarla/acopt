"""
Web dashboard: a small HTTP server (standard library only) in front of a Session.

    App       owns the Session and its run thread; builds the session in the
              background so the page is up while the race line is being planned.
    serve()   starts the HTTP server. Open http://127.0.0.1:8765 in a browser.

API (JSON unless noted)
    GET  /                    the dashboard page
    GET  /api/static          track geometry, car, config (fetched once)
    GET  /api/state           live state; ?since=<tick seq>&ev=<event seq> for increments
    GET  /api/reference       reference line / speed / sideslip around the lap
    GET  /api/lap?n=3         one finished lap resampled over distance
    GET  /api/lap.csv?n=3     full-rate CSV of one lap (download)
    GET  /api/session.json    session summary: all lap metrics + profile (download)
    POST /api/command         {"cmd": "start", "laps": 5} | pause | resume | stop |
                              segment | remove | weights | speed_scale | pace | kick | ...

The server binds to localhost by default. It has no authentication: only use
--host 0.0.0.0 on a network you trust (e.g. to watch from a tablet next to the rig).
"""
from __future__ import annotations

import json
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from .live import execute
from .telemetry import TICK_FIELDS, TelemetryHub, dumps

WEB = Path(__file__).parent / "web"


def find_corners(track, k_min=1 / 150.0, min_len=15.0):
    """Corner list from centreline curvature: s range, direction, minimum radius."""
    k, s, L = track.kappa, track.s, track.length
    inside = np.abs(k) > k_min
    if inside.all() or not inside.any():
        return []
    start = int(np.flatnonzero(~inside)[0])          # begin on a straight so nothing wraps
    idx = (np.arange(len(s)) + start) % len(s)
    ins = inside[idx]
    out, a = [], None
    for j, v in enumerate(ins):
        if v and a is None:
            a = j
        if a is not None and (not v or j == len(ins) - 1):
            seg = idx[a:j] if not v else idx[a:j + 1]
            a = None
            if len(seg) * track.ds < min_len:
                continue
            i = seg[np.argmax(np.abs(k[seg]))]
            out.append(dict(s_start=round(float(s[seg[0]]), 0), s_end=round(float(s[seg[-1]]), 0),
                            s_apex=round(float(s[i]), 0), dir="left" if k[i] > 0 else "right",
                            radius=round(float(1 / abs(k[i])), 0)))
    out.sort(key=lambda c: c["s_start"])
    for n, c in enumerate(out):
        c["name"] = f"T{n + 1}"
    return out


class App:
    def __init__(self, cfg_path, overrides=None):
        self.cfg_path, self.overrides = cfg_path, overrides or {}
        self.hub = TelemetryHub()
        self.session = None
        self.error = None
        self._thread = None
        self._lock = threading.Lock()
        self._static = None
        self._ref = (None, None)

    # ---- lifecycle ----
    def build_async(self):
        threading.Thread(target=self._build, daemon=True, name="acopt-build").start()
        return self

    def _build(self):
        from .session import Session
        try:
            self.session = Session(self.cfg_path, self.overrides, hub=self.hub)
        except Exception as e:  # noqa: BLE001 - shown in the page instead of a dead terminal
            self.error = f"{type(e).__name__}: {e}"
            self.hub.set_phase("error", str(e))
            traceback.print_exc()

    def start_run(self, laps=5, pace=None, learning=None, dt=0.01):
        with self._lock:
            s = self.session
            if s is None:
                return {"ok": False, "error": "session is still loading" if not self.error else self.error}
            if s.running or (self._thread and self._thread.is_alive()):
                return {"ok": False, "error": "already running"}
            laps = int(laps)
            if not 1 <= laps <= 1000:
                return {"ok": False, "error": "laps must be between 1 and 1000"}
            if learning is not None:
                s.learning = bool(learning)
            if pace is not None:
                s.pace = float(max(0.0, min(float(pace), 20.0)))
            s.controller.paused = False

            def job():
                try:
                    s.run(laps=laps, dt=dt)
                except Exception as e:  # noqa: BLE001 - already reported as an event
                    self.hub.set_phase("ready", f"run aborted: {e}")
                    traceback.print_exc()
            self._thread = threading.Thread(target=job, daemon=True, name="acopt-run")
            self._thread.start()
            return {"ok": True}

    def command(self, cmd: dict) -> dict:
        if not isinstance(cmd, dict):
            return {"ok": False, "error": "expected a JSON object"}
        if cmd.get("cmd") == "start":
            return self.start_run(cmd.get("laps", 5), cmd.get("pace"), cmd.get("learning"))
        if self.session is None:
            return {"ok": False, "error": self.error or "session is still loading"}
        if cmd.get("cmd") in ("save_profile", "load_profile") and not cmd.get("path"):
            if not self.session.profile_path:
                return {"ok": False, "error": "the config names no profile file"}
            cmd = dict(cmd, path=str(self.session.profile_path))
        r = execute(self.session, cmd)
        if r.get("ok") and cmd.get("cmd") == "save_profile" and self.session.watcher:
            self.session.watcher.touch()
        return r

    def shutdown(self):
        if self.session:
            self.session.close()
            if self._thread:
                self._thread.join(timeout=3.0)

    # ---- data for the page ----
    def static(self):
        s = self.session
        if s is None:
            return dict(ready=False, error=self.error, phase=self.hub.phase, phase_msg=self.hub.phase_msg)
        if self._static is None:
            tr = s.track
            step = max(1, int(round(max(3.0, tr.length / 3000.0) / tr.ds)))
            ss = tr.s[::step]
            rd = lambda a: np.round(a, 2).tolist()
            lx, ly = tr.to_world(ss, tr.interp(tr.w_left, ss))
            rx, ry = tr.to_world(ss, -tr.interp(tr.w_right, ss))
            p = s.params
            self._static = dict(
                ready=True, tick_fields=TICK_FIELDS,
                config=str(s.cfg_path.name), env=s.env_type, car=p.name, track=tr.name,
                length=round(tr.length, 1), s=rd(ss), left=[rd(lx), rd(ly)], right=[rd(rx), rd(ry)],
                corners=find_corners(tr), delta_max_deg=round(float(np.degrees(p.delta_max)), 2),
                car_len=round(p.lf + p.lr + 1.2, 2), car_width=p.car_width,
                profile_path=str(s.profile_path) if s.profile_path else None,
                can_kick=hasattr(s.env, "kick"), realtime=bool(s.env.realtime))
        return self._static

    def reference(self):
        s = self.session
        if s is None:
            return dict(version=-1)
        ver = s.profile.version
        if self._ref[0] != ver:
            ref = s.profile.reference(step=max(1, int(round(max(2.0, s.track.length / 2600.0) / s.track.ds))))
            X, Y = s.track.to_world(np.array(ref["s"]), np.array(ref["n"]))
            ref.update(version=ver, X=np.round(X, 2).tolist(), Y=np.round(Y, 2).tolist())
            self._ref = (ver, ref)
        return self._ref[1]

    def state(self, since=0, ev=0):
        snap = self.hub.snapshot(since, ev)
        s = self.session
        snap.update(ready=s is not None, error=self.error)
        if s is None:
            return snap
        hz = None
        h = s.controller.horizon() if s.running else None
        if h is not None:
            X, Y = s.track.to_world(h[0], h[1])
            hz = [np.round(X, 2).tolist(), np.round(Y, 2).tolist()]
        prof = s.profile.as_dict()
        snap.update(running=s.running, paused=s.controller.paused, lap_no=s.lap_no,
                    plan_time=round(s.raceline.lap_time, 3), pace=s.pace, learning=s.learning,
                    horizon=hz, log_dir=str(s.out_dir),
                    profile=dict(version=s.profile.version, speed_scale=prof.get("speed_scale", 1.0),
                                 segments=prof.get("segments", []), weights=prof.get("weights", {}),
                                 warnings=list(s.profile.warnings)))
        return snap

    def lap_csv_path(self, n: int):
        s = self.session
        if s is None:
            return None
        p = s.out_dir / f"lap{int(n):03d}.csv"
        return p if p.is_file() else None


class Handler(BaseHTTPRequestHandler):
    app: App = None
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):   # keep the terminal for the controller's own messages
        pass

    def _send(self, code, body: bytes, ctype="application/json", extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, dumps(obj).encode())

    def do_GET(self):  # noqa: N802
        try:
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            app = self.app
            if u.path in ("/", "/index.html"):
                return self._send(200, (WEB / "index.html").read_bytes(), "text/html; charset=utf-8")
            if u.path == "/api/static":
                return self._json(app.static())
            if u.path == "/api/state":
                return self._json(app.state(int(q.get("since", 0)), int(q.get("ev", 0))))
            if u.path == "/api/reference":
                return self._json(app.reference())
            if u.path == "/api/lap":
                tr = app.hub.trace(int(q.get("n", -1)))
                return self._json(tr if tr else {"error": "no such lap"}, 200 if tr else 404)
            if u.path == "/api/lap.csv":
                p = app.lap_csv_path(int(q.get("n", -1)))
                if p is None:
                    return self._json({"error": "no such lap"}, 404)
                return self._send(200, p.read_bytes(), "text/csv",
                                  {"Content-Disposition": f'attachment; filename="{p.name}"'})
            if u.path == "/api/session.json":
                if app.session is None:
                    return self._json({"error": "not ready"}, 404)
                body = app.session.write_summary().read_bytes() if app.hub.laps else \
                    dumps(app.hub.session_summary()).encode()
                return self._send(200, body, "application/json",
                                  {"Content-Disposition": 'attachment; filename="session.json"'})
            return self._json({"error": "not found"}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            try:
                self._json({"error": f"{type(e).__name__}: {e}"}, 500)
            except OSError:
                pass

    def do_POST(self):  # noqa: N802
        try:
            if urlparse(self.path).path != "/api/command":
                return self._json({"error": "not found"}, 404)
            # a web page on another origin must not be able to drive the car
            origin = self.headers.get("Origin")
            if origin and urlparse(origin).netloc != self.headers.get("Host"):
                return self._json({"ok": False, "error": "cross-origin request refused"}, 403)
            n = int(self.headers.get("Content-Length", 0))
            if n > 65536:
                return self._json({"ok": False, "error": "request too large"}, 413)
            try:
                cmd = json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                return self._json({"ok": False, "error": "body is not valid JSON"}, 400)
            return self._json(self.app.command(cmd))
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            try:
                self._json({"ok": False, "error": f"{type(e).__name__}: {e}"}, 500)
            except OSError:
                pass


def serve(app: App, host="127.0.0.1", port=8765):
    """Start the HTTP server in a daemon thread. Returns (server, url)."""
    handler = type("BoundHandler", (Handler,), {"app": app})
    last = None
    for p in range(port, port + 20):         # next free port if the default is taken
        try:
            srv = ThreadingHTTPServer((host, p), handler)
            break
        except OSError as e:
            last = e
    else:
        raise OSError(f"no free port in {port}..{port + 19}: {last}")
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True, name="acopt-http").start()
    shown = "127.0.0.1" if host in ("0.0.0.0", "") else host
    return srv, f"http://{shown}:{srv.server_address[1]}"
