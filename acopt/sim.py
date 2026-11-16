"""
Offline simulator: lets you develop the whole stack without the game.

The 'truth' car is the same model structure with DIFFERENT parameters (and
optionally a grip map that varies around the lap), plus actuator lag,
transport delay and sensor noise. If the controller survives this, it is far
more likely to survive Assetto Corsa's physics.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from .env import Env, Measurement, Command
from .track import Track
from .vehicle import VehicleModel, VehicleParams


class SimEnv(Env):
    realtime = False

    def __init__(self, track: Track, truth: VehicleParams, dt: float = 0.002,
                 tau_delta: float = 0.03, tau_T: float = 0.05, delay_s: float = 0.02,
                 noise: float = 1.0, grip_map=None, seed: int = 0):
        self.track, self.dt = track, dt
        self.model = VehicleModel(truth)
        self.p = truth
        self.tau_d, self.tau_T = tau_delta, tau_T
        self.delay = deque(maxlen=max(1, int(round(delay_s / dt))))
        self.noise = noise
        self.grip_map = grip_map        # callable s -> mu scale
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self, s0=0.0, n0=0.0, v0=30.0, **kw):
        X, Y = self.track.to_world(s0, n0)
        self.state = np.array([X, Y, self.track.psi_at(s0), v0, 0.0, 0.0, 0.0, 0.0])
        self.t = 0.0
        self.cmd = Command(0.0, 0.0)
        self.delay.clear()
        self._s, self._idx = s0, None
        self.lap, self._s_prev = 0, s0
        return self.read()

    def _deriv(self, st, dcmd, Tcmd, mu):
        X, Y, psi, vx, vy, r, d, T = st
        vx_dot, vy_dot, r_dot = np.array(self.model.f_body([max(vx, 1.0), vy, r, d, T], mu)).ravel()
        if vx < 1.0:  # crude low-speed handling
            vy_dot, r_dot = -vy * 5, -r * 5
        return np.array([vx * np.cos(psi) - vy * np.sin(psi),
                         vx * np.sin(psi) + vy * np.cos(psi), r,
                         vx_dot, vy_dot, r_dot,
                         (dcmd - d) / self.tau_d, (Tcmd - T) / self.tau_T])

    def step(self, dt):
        nsub = max(1, int(round(dt / self.dt)))
        for _ in range(nsub):
            self.delay.append((self.cmd.delta, self.cmd.T))
            dcmd, Tcmd = self.delay[0]
            dcmd = np.clip(dcmd, -self.p.delta_max, self.p.delta_max)
            Tcmd = np.clip(Tcmd, -1, 1)
            mu = self.grip_map(self._s) if self.grip_map else 1.0
            h, st = self.dt, self.state
            k1 = self._deriv(st, dcmd, Tcmd, mu)
            k2 = self._deriv(st + h / 2 * k1, dcmd, Tcmd, mu)
            k3 = self._deriv(st + h / 2 * k2, dcmd, Tcmd, mu)
            k4 = self._deriv(st + h * k3, dcmd, Tcmd, mu)
            self.state = st + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            self.state[3] = max(self.state[3], 0.0)
            self.t += h
        s, n, _, self._idx = self.track.to_frenet(self.state[0], self.state[1], hint=self._idx)
        if self._s_prev > 0.8 * self.track.length and s < 0.2 * self.track.length:
            self.lap += 1
        self._s_prev, self._s, self._n = s, s, n

    def read(self):
        X, Y, psi, vx, vy, r, d, T = self.state
        z = self.noise
        s, n, _, self._idx = self.track.to_frenet(X, Y, hint=self._idx)
        wl, wr = self.track.bounds_at(s)
        hw = 0.5 * self.p.car_width
        over = max(n + hw - wl, -n + hw - wr)       # metres the outer edge is beyond the line
        tyres_out = 0 if over < 0 else (2 if over < hw else 4)   # 4: centre of car past the line
        return Measurement(
            t=self.t, X=X + z * 0.02 * self.rng.normal(), Y=Y + z * 0.02 * self.rng.normal(),
            psi=psi + z * 0.002 * self.rng.normal(),
            vx=vx + z * 0.05 * self.rng.normal(), vy=vy + z * 0.03 * self.rng.normal(),
            r=r + z * 0.005 * self.rng.normal(), delta=d, T=T,
            tyres_out=tyres_out, lap=self.lap,
            extra=dict(true_s=s, true_n=n))

    def write(self, cmd: Command):
        self.cmd = cmd
