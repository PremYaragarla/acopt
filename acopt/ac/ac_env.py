"""
Assetto Corsa environment: shared memory in, virtual joystick (vJoy) out.

Setup (once):
  1. Install vJoy (one device, axes X, Y, Z, 8 buttons) and `pip install pyvjoy`.
  2. In AC Controls: steering = vJoy X, throttle = vJoy Y, brake = vJoy Z,
     gear up/down = buttons 1/2. Steering gamma 1.0, filter 0, speed
     sensitivity 0, deadzones 0. Set the controller steering lock to the
     car's lock (steer_lock_wheel_deg * 2) so the axis maps 1:1.
  3. Assists: auto-shifter ON initially (or enable `auto_shift` below),
     auto-clutch ON, ABS/TC as you like (the model assumes OFF; leave TC off
     for drifting).
  4. Run scripts/ac_probe.py and copy the calibrated signs into configs/ac.yaml.
"""
from __future__ import annotations

import time

import numpy as np

from ..env import Env, Measurement, Command
from ..track import wrap
from .shared_memory import ACSharedMemory, wstr


class VJoyOut:
    def __init__(self, device_id=1):
        try:
            import pyvjoy  # Windows only
            self.j = pyvjoy.VJoyDevice(device_id)
        except Exception as e:  # noqa: BLE001 - ImportError, vJoy driver missing, device busy...
            raise RuntimeError(
                f"Cannot open vJoy device {device_id} ({type(e).__name__}: {e}). Install the vJoy "
                "driver, enable one device with axes X/Y/Z and 8 buttons in 'Configure vJoy', "
                "run `pip install pyvjoy`, and make sure no other program owns the device.") from e
        self.MAX = 0x8000
        self._btn = {}

    def axes(self, steer: float, gas: float, brake: float):
        """steer in [-1,1], gas/brake in [0,1]."""
        a = lambda v: int(np.clip((v + 1) * 0.5, 0, 1) * (self.MAX - 1)) + 1
        p = lambda v: int(np.clip(v, 0, 1) * (self.MAX - 1)) + 1
        self.j.data.wAxisX = a(steer)
        self.j.data.wAxisY = p(gas)
        self.j.data.wAxisZ = p(brake)
        self.j.update()

    def button(self, n: int, pressed: bool):
        if self._btn.get(n) != pressed:
            self.j.set_button(n, int(pressed))
            self._btn[n] = pressed


class ACEnv(Env):
    realtime = True

    def __init__(self, vehicle_params, cal: dict, vjoy_device=1, auto_shift=None,
                 sm=None, out=None, clock=time.perf_counter, sleep=time.sleep, realtime=True):
        """sm / out / clock / sleep can be injected (see ac/mock.py) so the whole AC I/O
        path - frame conversion, extrapolation, staleness, axis mapping - is testable
        without the game."""
        self.sm = sm if sm is not None else ACSharedMemory()
        self.out = out if out is not None else VJoyOut(vjoy_device)
        self.clock, self._sleep, self.realtime = clock, sleep, realtime
        self.p = vehicle_params
        self.cal = dict(sign_z=-1.0, heading_sign=1.0, heading_offset=0.0, yaw_rate_sign=1.0,
                        steer_sign=1.0, brake_gamma=1.0, stale_s=0.1)
        self.cal.update(cal or {})
        self.auto_shift = auto_shift      # e.g. dict(up_rpm=18000, down_rpm=12500)
        self._last_pid = None
        self._last_change = self.clock()
        self._gpos = None                 # (t, X, Y) from the graphics page
        self._shift_t = 0.0

    # --- frame conversion -------------------------------------------------
    def _heading(self, ph):
        return wrap(self.cal["heading_sign"] * ph.heading + self.cal["heading_offset"])

    def read(self) -> Measurement:
        now = self.clock()
        ph, gr = self.sm.physics(), self.sm.graphics()
        if ph.packetId != self._last_pid:
            if self._last_pid is not None and ph.packetId < self._last_pid:
                self._gpos = None             # session restarted / car teleported to the pits
            self._last_pid, self._last_change = ph.packetId, now
        fresh = (now - self._last_change) < self.cal["stale_s"] and gr.status == 2  # 2 = AC_LIVE
        sz = self.cal["sign_z"]
        VX, VY = ph.velocity[0], sz * ph.velocity[2]
        Xg, Yg = gr.carCoordinates[0], sz * gr.carCoordinates[2]
        # graphics page updates at render rate: extrapolate with world velocity
        if self._gpos is None or (Xg, Yg) != self._gpos[1:]:
            self._gpos = (now, Xg, Yg)
        dtp = min(now - self._gpos[0], 0.1)   # never extrapolate a frozen page far
        X, Y = Xg + VX * dtp, Yg + VY * dtp
        psi = self._heading(ph)
        c, s = np.cos(psi), np.sin(psi)
        vx, vy = c * VX + s * VY, -s * VX + c * VY
        r = self.cal["yaw_rate_sign"] * ph.localAngularVel[1]
        delta = self.cal["steer_sign"] * ph.steerAngle * self.p.delta_max  # steerAngle is normalised
        T = ph.gas - ph.brake
        return Measurement(t=now, X=X, Y=Y, psi=psi, vx=vx, vy=vy, r=r, delta=delta, T=T,
                           tyres_out=ph.numberOfTyresOut, lap=gr.completedLaps, fresh=fresh,
                           extra=dict(rpm=ph.rpms, gear=ph.gear, speed_kmh=ph.speedKmh,
                                      accG=tuple(ph.accG), npos=gr.normalizedCarPosition))

    def write(self, cmd: Command):
        steer = self.cal["steer_sign"] * cmd.delta / self.p.delta_max
        gas = max(cmd.T, 0.0)
        brake = max(-cmd.T, 0.0) ** (1.0 / self.cal["brake_gamma"])
        self.out.axes(steer, gas, brake)
        if self.auto_shift:
            self._shift()

    def _shift(self):
        ph = self.sm.physics()
        now = self.clock()
        up = ph.rpms > self.auto_shift["up_rpm"] and now - self._shift_t > 0.25
        dn = ph.rpms < self.auto_shift["down_rpm"] and ph.gear > 2 and now - self._shift_t > 0.25
        self.out.button(1, up)
        self.out.button(2, dn)
        if up or dn:
            self._shift_t = now

    def step(self, dt):
        self._sleep(max(dt, 0))   # AC runs on its own clock; caller paces the loop

    def describe(self) -> dict:
        """Car / track the game reports (static page), for display and sanity checks."""
        try:
            st = self.sm.static()
            return dict(car=wstr(st.carModel), track=wstr(st.track), ac_version=wstr(st.acVersion))
        except Exception:  # noqa: BLE001
            return {}

    def is_live(self) -> bool:
        try:
            return self.sm.graphics().status == 2
        except Exception:  # noqa: BLE001
            return False

    def wait_live(self, stop=None, timeout=None, poll=0.2):
        """Block until AC is in a live session and physics packets are flowing.
        Returns the first fresh Measurement, or None if stopped / timed out."""
        t0 = self.clock()
        while not (stop is not None and stop.is_set()):
            m = self.read()
            if m.fresh and self._last_pid:
                return m
            if timeout is not None and self.clock() - t0 > timeout:
                return None
            self._sleep(poll)
        return None

    def neutral(self):
        self.out.axes(0.0, 0.0, 0.0)
        self.out.button(1, False)
        self.out.button(2, False)

    def close(self):
        try:
            self.neutral()
        finally:
            self.sm.close()
