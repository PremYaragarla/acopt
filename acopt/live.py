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
  {"cmd": "pause"} / {"cmd": "resume"} / {"cmd": "stop"}
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


class CommandServer:
    def __init__(self, session, host="127.0.0.1", port=47800):
        self.session = session
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(self.addr)
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

    def _rx(self):
        while not self._stop.is_set():
            try:
                data, peer = self.sock.recvfrom(65536)
            except socket.timeout:
                continue
            try:
                cmd = json.loads(data.decode())
                reply = self.handle_fast(cmd)
            except Exception as e:  # never let a bad packet kill the loop
                reply = {"ok": False, "error": repr(e)}
            self.sock.sendto(json.dumps(reply).encode(), peer)

    def handle_fast(self, cmd):
        """Cheap commands run immediately; profile rebuilds are queued."""
        s = self.session
        c = cmd.get("cmd")
        if c == "status":
            return {"ok": True, **s.status()}
        if c == "pause":
            s.controller.paused = True; return {"ok": True}
        if c == "resume":
            s.controller.paused = False; return {"ok": True}
        if c == "stop":
            s.request_stop(); return {"ok": True}
        if c in ("segment", "remove", "weights", "speed_scale", "load_profile"):
            self.jobs.put(cmd)
            return {"ok": True, "queued": c}
        return {"ok": False, "error": f"unknown cmd {c}"}

    def _worker(self):
        while not self._stop.is_set():
            try:
                cmd = self.jobs.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                apply_command(self.session.profile, cmd)
            except Exception as e:
                print("[live] command failed:", cmd, e)


def apply_command(profile, cmd):
    c = cmd["cmd"]
    t0 = time.time()
    if c == "segment":
        fields = {k: v for k, v in cmd.items() if k not in ("cmd", "name")}
        profile.set_segment(cmd["name"], **fields)
    elif c == "remove":
        profile.remove_segment(cmd["name"])
    elif c == "weights":
        mode = cmd.get("mode", "drift")
        profile.update({"weights": {mode: {k: float(v) for k, v in cmd.items() if k not in ("cmd", "mode")}}})
    elif c == "speed_scale":
        profile.update({"speed_scale": float(cmd["value"])})
    elif c == "load_profile":
        profile.update(profile.load_yaml(cmd["path"]))
    print(f"[live] applied {c} -> profile v{profile.version} ({1e3*(time.time()-t0):.0f} ms)")


class ProfileWatcher(threading.Thread):
    def __init__(self, profile, path, period=0.5):
        super().__init__(daemon=True)
        self.profile, self.path, self.period = profile, path, period
        self.mtime = os.path.getmtime(path) if os.path.exists(path) else 0

    def run(self):
        while True:
            time.sleep(self.period)
            try:
                mt = os.path.getmtime(self.path)
                if mt != self.mtime:
                    self.mtime = mt
                    self.profile.update(self.profile.load_yaml(self.path))
                    print(f"[live] reloaded {self.path} -> v{self.profile.version}")
            except Exception as e:
                print("[live] profile reload failed:", e)


def send(cmd: dict, host="127.0.0.1", port=47800, timeout=1.0):
    """Client helper (used by scripts/tune.py)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(timeout)
    s.sendto(json.dumps(cmd).encode(), (host, port))
    try:
        return json.loads(s.recv(65536).decode())
    except socket.timeout:
        return {"ok": False, "error": "no reply"}
