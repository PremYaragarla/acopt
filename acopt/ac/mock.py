"""
Stand-in for Assetto Corsa's shared memory + vJoy, driven by the built-in simulator.

It speaks AC's conventions (Y-up world with the ground plane in x/z, a heading with
its own sign and zero, normalised steering, separate gas/brake axes, a graphics page
that only refreshes at render rate) so that `ACEnv` - the exact code that talks to
the game - can be exercised end to end on any machine:

    env: {type: ac_mock, calibration: {...}}

This proves the plumbing (frame conversion, extrapolation, axis mapping, staleness
handling). It does NOT prove the struct layouts or sign conventions of your AC
install: scripts/ac_probe.py does that.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from ..env import Command
from ..sim import SimEnv


class MockAC:
    def __init__(self, sim: SimEnv, cal: dict, vehicle_params, render_hz: float = 60.0):
        self.sim, self.p = sim, vehicle_params
        self.cal = dict(sign_z=-1.0, heading_sign=1.0, heading_offset=0.0, yaw_rate_sign=1.0,
                        steer_sign=1.0, brake_gamma=1.0)
        self.cal.update(cal or {})
        self.render_dt = 1.0 / render_hz
        self.packet = 0
        self.status = 2                      # AC_LIVE; set to 1/3 to emulate replay/pause
        self.frozen = False                  # True: physics page stops updating (game hang)
        self._steer = self._gas = self._brake = 0.0
        self._gfx_t, self._gfx_xy = -1e9, (0.0, 0.0)
        self.buttons = {}
        self._refresh_graphics(force=True)

    # ---- what ACEnv reads ----
    def physics(self):
        X, Y, psi, vx, vy, r, d, T = self.sim.state
        c = self.cal
        VX = vx * np.cos(psi) - vy * np.sin(psi)
        VY = vx * np.sin(psi) + vy * np.cos(psi)
        heading = c["heading_sign"] * (psi - c["heading_offset"])   # psi = hs*heading + off
        tyres_out = self.sim.read().tyres_out
        return SimpleNamespace(
            packetId=self.packet, gas=self._gas, brake=self._brake, gear=3, rpms=12000,
            steerAngle=c["steer_sign"] * d / self.p.delta_max, speedKmh=3.6 * float(np.hypot(vx, vy)),
            velocity=(VX, 0.0, c["sign_z"] * VY), accG=(0.0, 0.0, 0.0), heading=heading,
            localAngularVel=(0.0, c["yaw_rate_sign"] * r, 0.0), numberOfTyresOut=tyres_out)

    def graphics(self):
        X, Y = self._gfx_xy
        return SimpleNamespace(status=self.status, carCoordinates=(X, 0.0, self.cal["sign_z"] * Y),
                               completedLaps=self.sim.lap, normalizedCarPosition=0.0)

    def static(self):
        raise RuntimeError("mock has no static page")

    def close(self):
        pass

    # ---- what ACEnv writes (vJoy) ----
    def axes(self, steer, gas, brake):
        self._steer, self._gas, self._brake = (float(np.clip(steer, -1, 1)),
                                               float(np.clip(gas, 0, 1)), float(np.clip(brake, 0, 1)))

    def button(self, n, pressed):
        self.buttons[n] = pressed

    # ---- time ----
    def clock(self):
        return self.sim.t

    def advance(self, dt):
        if dt <= 0:
            dt = self.sim.dt
        c = self.cal
        delta = c["steer_sign"] * self._steer * self.p.delta_max
        T = self._gas - self._brake ** c["brake_gamma"]
        self.sim.write(Command(delta, T))
        self.sim.step(dt)
        if not self.frozen:
            self.packet += 1
            self._refresh_graphics()

    def _refresh_graphics(self, force=False):
        if force or self.sim.t - self._gfx_t >= self.render_dt - 1e-9:
            self._gfx_t = self.sim.t
            self._gfx_xy = (float(self.sim.state[0]), float(self.sim.state[1]))
