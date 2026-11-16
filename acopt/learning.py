"""
Lap-to-lap learning: the "iterate to the optimum" loop.

After each lap the learner compares what happened with what the plan assumed
and updates a per-metre GRIP MAP mu(s):
  * incidents (tyres out, supervisor fallback, big tracking error) -> lower mu there
  * the car was at the tyre limit (large slip angles) -> blend toward the grip
    the car actually used there (measured |a| / available normal accel)
  * clean, sub-limit sections -> raise mu a little (explore)
The planner is then re-solved with the new map (warm-started from the last
line), and the profile swaps to the new race line at the next lap.

Converges when the lap time stops improving and laps are clean.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter1d

from .track import Track
from .vehicle import VehicleModel


@dataclass
class LearnCfg:
    bin_m: float = 5.0
    down_step: float = 0.04          # mu cut around an incident
    incident_spread_m: float = 60.0  # how far before the incident to cut (cause precedes effect)
    up_step: float = 0.01            # optimistic increase on clean sub-limit bins
    tyres_out_limit: int = 2         # AC invalidates a lap above 2 tyres out
    blend_measured: float = 0.6
    sat_frac: float = 0.85           # |alpha| / alpha_peak above which a bin counts as at-limit
    mu_min: float = 0.6
    mu_max: float = 1.25
    n_err_incident: float = 1.5
    smooth_m: float = 15.0
    converge_dt: float = 0.03        # s improvement below which we call it converged
    converge_laps: int = 2


@dataclass
class LapRecord:
    lap: int
    lap_time: float
    incidents: int
    mu_mean: float


class GripLearner:
    def __init__(self, track: Track, model: VehicleModel, cfg: LearnCfg = LearnCfg(), mu0=1.0):
        self.track, self.model, self.cfg = track, model, cfg
        self.nb = int(np.ceil(track.length / cfg.bin_m))
        self.s_bins = (np.arange(self.nb) + 0.5) * cfg.bin_m
        self.mu = np.full(self.nb, float(mu0))
        self.history: list[LapRecord] = []

    def mu_map(self):
        return self.s_bins, self.mu.copy()

    def update(self, log: list[dict], lap_time: float) -> np.ndarray:
        c, p = self.cfg, self.model.p
        if len(log) < 10:
            return self.mu
        s = np.array([l["s"] for l in log])
        b = np.clip((s / c.bin_m).astype(int), 0, self.nb - 1)
        n_err = np.abs(np.array([l["n"] - l["n_ref"] for l in log]))
        out = np.array([l.get("tyres_out", 0) > c.tyres_out_limit for l in log])
        fb = np.array([l.get("state", "MPC") != "MPC" for l in log])
        drift = np.array([l.get("mode", 0) > 0.05 for l in log])
        vx = np.array([l["vx"] for l in log]); vy = np.array([l["vy"] for l in log])
        r = np.array([l["r"] for l in log])
        dl = np.array([l["delta_cmd"] for l in log]); T = np.array([l["T_cmd"] for l in log])

        # measured body accelerations (from the log) vs. what the model predicts at the
        # same states with the grip the plan assumed there  ->  grip ratio
        t = np.array([l["t"] for l in log])
        dt = np.maximum(np.gradient(t), 1e-4)
        vxd = gaussian_filter1d(np.gradient(vx) / dt, 3)
        vyd = gaussian_filter1d(np.gradient(vy) / dt, 3)
        ax, ay = vxd - r * vy, vyd + r * vx
        dA = np.array([l.get("delta_act", l["delta_cmd"]) for l in log])
        TA = np.array([l.get("T_act", l["T_cmd"]) for l in log])
        mu_plan = self.mu[b]                       # grip the plan assumed here
        XB = np.vstack([np.maximum(vx, 1), vy, r, dA, TA])
        pred = np.array(self.model.f_body.map(len(vx))(XB, mu_plan[None, :])).reshape(3, -1)
        ax_p, ay_p = pred[0] - r * vy, pred[1] + r * vx
        a_f = dA - np.arctan2(vy + p.lf * r, np.maximum(vx, 1))
        a_r = -np.arctan2(vy - p.lr * r, np.maximum(vx, 1))
        sat = (np.maximum(np.abs(a_f), np.abs(a_r)) > c.sat_frac * p.alpha_peak) & ~drift
        sat &= np.hypot(ax_p, ay_p) > 5.0
        ratio = np.hypot(ax, ay) / np.maximum(np.hypot(ax_p, ay_p), 1e-3)
        mu_used = mu_plan * ratio

        inc = (out | fb | (n_err > c.n_err_incident)) & ~drift
        new = self.mu.copy()
        # 1) at-limit bins: move toward what the tyres actually delivered
        for k in np.unique(b[sat]):
            meas = np.median(mu_used[(b == k) & sat])
            new[k] = (1 - c.blend_measured) * new[k] + c.blend_measured * np.clip(meas, c.mu_min, c.mu_max)
        # 2) incidents: cut grip over the approach to the incident
        inc_bins = np.zeros(self.nb, bool)
        inc_bins[np.unique(b[inc])] = True
        spread = int(c.incident_spread_m / c.bin_m)
        if inc_bins.any():
            ahead = np.zeros(self.nb, bool)
            for k in np.where(inc_bins)[0]:
                ahead[(k - np.arange(spread + 1)) % self.nb] = True
            new[ahead] *= (1 - c.down_step)
        # 3) clean & below limit: explore upward
        visited = np.zeros(self.nb, bool); visited[np.unique(b)] = True
        sat_bins = np.zeros(self.nb, bool); sat_bins[np.unique(b[sat])] = True
        clean = visited & ~maximum_filter1d(inc_bins, spread, mode="wrap") & ~sat_bins
        new[clean] *= (1 + c.up_step)
        new = np.clip(gaussian_filter1d(new, c.smooth_m / c.bin_m, mode="wrap"), c.mu_min, c.mu_max)
        self.mu = new
        self.history.append(LapRecord(len(self.history) + 1, lap_time, int(inc.sum() > 0) * int(np.unique(b[inc]).size),
                                      float(new.mean())))
        return new

    def converged(self) -> bool:
        h = self.history
        c = self.cfg
        if len(h) < c.converge_laps + 1:
            return False
        last = h[-(c.converge_laps + 1):]
        clean = all(r.incidents == 0 for r in last[1:])
        gains = [last[i].lap_time - last[i + 1].lap_time for i in range(len(last) - 1)]
        return clean and all(abs(g) < c.converge_dt for g in gains)
