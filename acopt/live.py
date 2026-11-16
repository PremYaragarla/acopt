"""
Live tuning while the car is driving.

* CommandServer: UDP JSON on localhost (default 127.0.0.1:47800).
  Each datagram is one command; the reply is JSON {ok, ...}.
* ProfileWatcher: polls a profile YAML and applies it whenever you save it.

Commands
  {"cmd": "segment", "name": "T3", "beta_deg": 25}          # create/update a segment
  {"cmd": "segment", "name": "T3", "mode": "drift", "s_start": 940, "s_end": 1110, "beta_deg": 20}
  {"cmd": "remove", "name": "T3"}
  {"cmd": "weights", "mode": "drift", "beta": 2.0, "n": 1.0} # MPC weights (race|drift|common)
  {"cmd": "speed_scale", "value": 0.95}
  {"cmd": "load_profile", "path": "profiles/drift_demo.yaml"}
  {"cmd": "save_profile", "path": "profiles/my.yaml"}
  {"cmd": "pause"} / {"cmd": "resume"} / {"cmd": "stop"}
  {"cmd": "pace", "value": 1.0}        # simulator speed (x real time, 0 = as fast as possible)
  {"cmd": "kick", "yaw_rate": 1.5}     # simulator only: disturbance to test recovery
  {"cmd": "status"}

Profile rebuilds (drift equilibria) run in a worker thread, so a command never
stalls the control loop; the MPC switches to the new references atomically.
"""
from __future__ import annotations

import json
import os
import queue
import socket
import threading
import time
from pathlib import Path


SLOW = ("segment", "remove", "weights", "speed_scale", "load_profile", "save_profile")


def execute(session, cmd: dict) -> dict:
    """Run one command synchronously and return a JSON-able reply {ok, ...}.
    Used by the web dashboard directly and by the UDP server (slow ones via its worker).
    Never raises: a bad command is reported, not propagated into the control loop."""
    try:
        if not isinstance(cmd, dict) or "cmd" not in cmd:
            return {"ok": False, "error": "expected a JSON object with a 'cmd' field"}
        c = cmd["cmd"]
        s = session
        if c == "status":
            return {"ok": True, **s.status()}
        if c == "pause":
            s.controller.paused = True
        elif c == "resume":
            s.controller.paused = False
        elif c == "stop":
            s.request_stop()
        elif c == "pace":
            s.pace = float(max(0.0, min(float(cmd.get("value", 0)), 20.0)))
        elif c == "learning":
            s.learning = bool(cmd.get("value", True))
        elif c == "kick":
            if not hasattr(s.env, "kick"):
                return {"ok": False, "error": "kick is only available in the simulator"}
            s.env.kick(yaw_rate=float(cmd.get("yaw_rate", 1.5)), vy=float(cmd.get("vy", 0.0)))
        elif c in SLOW:
            t0 = time.time()
            apply_command(s.profile, cmd, root=getattr(s, "root", None))
            reply = {"ok": True, "profile_version": s.profile.version,
                     "warnings": list(s.profile.warnings), "ms": round(1e3 * (time.time() - t0))}
            if c == "save_profile":
                reply["path"] = cmd["path"]
            s.hub.event("live", f"{c} {_brief(cmd)} -> profile v{s.profile.version}")
            for w in s.profile.warnings:
                s.hub.event("profile", w, "warn")
            return reply
        else:
            return {"ok": False, "error": f"unknown cmd {c!r}"}
        s.hub.event("live", f"{c} {_brief(cmd)}".strip())
        return {"ok": True}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e) or repr(e)}


def _brief(cmd):
    return " ".join(f"{k}={v}" for k, v in cmd.items() if k != "cmd")


class CommandServer:
    def __init__(self, session, host="127.0.0.1", port=47800):
        self.session = session
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(self.addr)            # OSError if the port is taken: caller decides
        self.sock.settimeout(0.2)
        self.jobs: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self.threads = [threading.Thread(target=self._rx, daemon=True),
                        threading.Thread(target=self._worker, daemon=True)]

    def start(self):
        for t in self.threads:
            t.start()
        print(f"[live] listening on udp://{self.addr[0]}:{self.addr[1]}")
        return self

    def stop(self):
        self._stop.set()
        try:
            self.sock.close()
        except OSError:
            pass

    def _rx(self):
        while not self._stop.is_set():
            try:
                data, peer = self.sock.recvfrom(65536)
            except socket.timeout:
                continue
            except OSError:
                break                         # socket closed
            try:
                cmd = json.loads(data.decode())
                reply = self.handle_fast(cmd)
            except Exception as e:  # never let a bad packet kill the loop
                reply = {"ok": False, "error": repr(e)}
            try:
                self.sock.sendto(json.dumps(reply, default=str).encode(), peer)
            except OSError:
                pass

    def handle_fast(self, cmd):
        """Cheap commands run immediately; profile rebuilds are queued."""
        if isinstance(cmd, dict) and cmd.get("cmd") in SLOW:
            self.jobs.put(cmd)
            return {"ok": True, "queued": cmd["cmd"]}
        return execute(self.session, cmd)

    def _worker(self):
        while not self._stop.is_set():
            try:
                cmd = self.jobs.get(timeout=0.2)
            except queue.Empty:
                continue
            r = execute(self.session, cmd)
            if not r.get("ok"):
                self.session.hub.event("live", f"command failed: {cmd} ({r.get('error')})", "warn")


def apply_command(profile, cmd, root=None):
    c = cmd["cmd"]

    def resolve(p):
        p = Path(p)
        return p if p.is_absolute() or root is None else Path(root) / p

    if c == "segment":
        if not cmd.get("name"):
            raise ValueError("segment: 'name' is required")
        fields = {k: v for k, v in cmd.items() if k not in ("cmd", "name")}
        profile.set_segment(str(cmd["name"]), **fields)
    elif c == "remove":
        profile.remove_segment(cmd["name"])
    elif c == "weights":
        mode = cmd.get("mode", "drift")
        profile.update({"weights": {mode: {k: float(v) for k, v in cmd.items() if k not in ("cmd", "mode")}}})
    elif c == "speed_scale":
        profile.update({"speed_scale": float(cmd["value"])})
    elif c == "load_profile":
        new = profile.load_yaml(resolve(cmd["path"]))
        new.setdefault("segments", [])
        profile.update(new)
    elif c == "save_profile":
        profile.save_yaml(resolve(cmd["path"]))


class ProfileWatcher(threading.Thread):
    def __init__(self, profile, path, period=0.5, on_event=None):
        super().__init__(daemon=True)
        self.profile, self.path, self.period = profile, path, period
        self.mtime = os.path.getmtime(path) if os.path.exists(path) else 0
        self.on_event = on_event or (lambda msg, level="info": print("[live]", msg))
        self._stop_ev = threading.Event()

    def stop(self):
        self._stop_ev.set()

    def touch(self):
        """Ignore the next change (the profile was just saved by us)."""
        self.mtime = os.path.getmtime(self.path) if os.path.exists(self.path) else 0

    def run(self):
        while not self._stop_ev.wait(self.period):
            try:
                mt = os.path.getmtime(self.path)
                if mt != self.mtime:
                    self.mtime = mt
                    self.profile.update(self.profile.load_yaml(self.path))
                    self.on_event(f"reloaded {self.path} -> profile v{self.profile.version}")
            except Exception as e:  # noqa: BLE001 - a half-saved file must not kill the watcher
                self.on_event(f"profile reload failed: {e}", "warn")


def send(cmd: dict, host="127.0.0.1", port=47800, timeout=1.0):
    """Client helper (used by scripts/tune.py)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(timeout)
    s.sendto(json.dumps(cmd).encode(), (host, port))
    try:
        return json.loads(s.recv(65536).decode())
    except socket.timeout:
        return {"ok": False, "error": "no reply"}
