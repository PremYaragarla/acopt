"""
Reference profiles: what the MPC should track, as functions of s.

A profile = the optimal race line + any number of segments that override it.
Segment modes:
  race   - follow the race line (optionally with a speed scale or line offset)
  drift  - hold a sideslip angle |beta| through the segment; speed and
           feed-forward come from drift equilibria on the chosen line
Segments blend in/out over `blend_m` metres so mode changes are smooth.
Everything here is rebuilt in milliseconds, so it can change while driving.

YAML example:
  speed_scale: 0.97
  segments:
    - name: hairpin_drift
      mode: drift
      s_start: 420
      s_end: 560
      beta_deg: 18
      line: race          # race | center | <metres offset from race line>
      speed_scale: 1.0
      blend_m: 25
"""
from __future__ import annotations

import copy
import threading
from pathlib import Path

import numpy as np
import yaml
from scipy.ndimage import gaussian_filter1d

from .drift import DriftEquilibrium
from .planner import RaceLine
from .track import Track
from .vehicle import IX

DEFAULT_WEIGHTS = {
    # errors are normalised: n [m], v [m/s], beta [deg], r [0.1 rad/s], xi [deg],
    # ff_delta [deg], ff_T [0.1], ddelta [rad/s], dT [1/s]; time [s]
    "race":  dict(n=4.0, v=1.0, beta=0.05, r=0.2, xi=0.3, time=0.0,
                  ff_delta=0.02, ff_T=0.2, ddelta=5.0, dT=0.5),
    "drift": dict(n=1.5, v=0.8, beta=1.5, r=1.0, xi=0.0, time=0.0,
                  ff_delta=0.01, ff_T=0.1, ddelta=5.0, dT=0.5),
    "common": dict(slack=1e4, terminal=5.0),
}


def _smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


class ReferenceProfile:
    FIELDS = ("mode", "n", "v", "beta", "r", "xi", "delta", "T", "mu")

    def __init__(self, track: Track, raceline: RaceLine, model, cfg: dict | None = None):
        self.track, self.model = track, model
        self.raceline = raceline
        self.eq = DriftEquilibrium(model)
        self.cfg = {"speed_scale": 1.0, "segments": [], "weights": copy.deepcopy(DEFAULT_WEIGHTS)}
        self._eq_cache = {}
        self._lock = threading.Lock()        # guards the arrays the controller reads
        self._ulock = threading.Lock()       # serialises rebuilds (live server, watcher, learner)
        self.version = 0
        self.warnings: list[str] = []
        self.update(cfg or {})

    # ---------------- config handling ----------------
    @staticmethod
    def load_yaml(path):
        return yaml.safe_load(Path(path).read_text()) or {}

    def update(self, patch: dict):
        """Merge a (partial) config and rebuild. Thread-safe; returns new version."""
        with self._ulock:
            return self._update(patch)

    def _update(self, patch):
        cfg = copy.deepcopy(self.cfg)
        for k, v in patch.items():
            if k == "weights":
                for mode, wd in v.items():
                    cfg["weights"].setdefault(mode, {}).update(wd)
            else:
                cfg[k] = copy.deepcopy(v)
        arrays, warnings = self._build(cfg)
        with self._lock:
            self.cfg, self.arrays, self.warnings = cfg, arrays, warnings
            self.version += 1
        for w in warnings:
            print("[profile] WARNING:", w)
        return self.version

    def set_segment(self, name, **fields):
        segs = copy.deepcopy(self.cfg["segments"])
        for sg in segs:
            if sg.get("name") == name:
                sg.update(fields)
                break
        else:
            segs.append(dict(name=name, **fields))
        return self.update({"segments": segs})

    def remove_segment(self, name):
        return self.update({"segments": [s for s in self.cfg["segments"] if s.get("name") != name]})

    def set_raceline(self, rl: RaceLine):
        self.raceline = rl
        return self.update({})

    # ---------------- build ----------------
    def _eq_table(self, beta_deg, mu):
        key = (round(beta_deg, 2), round(mu, 3))
        if key not in self._eq_cache:
            self._eq_cache[key] = self.eq.table(np.radians(beta_deg), mu=mu)
        return self._eq_cache[key]

    def _build(self, cfg):
        tr, rl = self.track, self.raceline
        s = tr.s
        L = tr.length
        I = lambda col: np.interp(s, rl.s, rl.X[:, IX[col]], period=L)
        race = dict(n=I("n"), v=I("vx") * cfg.get("speed_scale", 1.0), r=I("r"), xi=I("xi"),
                    delta=I("delta"), T=I("T"),
                    beta=np.arctan2(I("vy"), I("vx")),
                    mu=np.interp(s, rl.s, rl.mu, period=L))
        out = {k: v.copy() for k, v in race.items()}
        out["mode"] = np.zeros_like(s)
        warnings = []
        for sg in cfg.get("segments", []):
            a, b = float(sg["s_start"]) % L, float(sg["s_end"]) % L
            blend = float(sg.get("blend_m", 20.0))
            rel = (s - a) % L
            span = (b - a) % L
            inside = rel <= span
            w = np.where(inside,
                         _smoothstep(rel / blend) * _smoothstep((span - rel) / blend), 0.0)
            mode = sg.get("mode", "race")
            # line choice
            line = sg.get("line", "race")
            if line == "race":
                n_line = race["n"]
            elif line == "center":
                n_line = np.zeros_like(s)
            else:
                n_line = race["n"] + float(line)
            wl, wr = tr.w_left, tr.w_right
            hw = 0.5 * self.model.p.car_width + 0.5
            n_line = np.clip(n_line, -wr + hw, wl - hw)
            if mode == "race":
                v_seg = race["v"] * float(sg.get("speed_scale", 1.0))
                for k, val in (("n", n_line), ("v", v_seg)):
                    out[k] = (1 - w) * out[k] + w * val
                continue
            if mode != "drift":
                warnings.append(f"segment {sg.get('name')}: unknown mode {mode}")
                continue
            # path curvature of the chosen line (from its world geometry)
            X, Y = tr.to_world(s, n_line)
            dX, dY = np.gradient(X), np.gradient(Y)
            ddX, ddY = np.gradient(dX), np.gradient(dY)
            kap = (dX * ddY - dY * ddX) / np.power(dX ** 2 + dY ** 2, 1.5)
            kap = gaussian_filter1d(kap, 8.0 / tr.ds, mode="wrap")
            kmin = 1 / 250.0
            sign = np.sign(np.where(np.abs(kap) < 1e-6, 1e-6, kap))
            kap_eq = np.where(np.abs(kap) < kmin, sign * kmin, kap)
            beta_deg = float(sg.get("beta_deg", 15.0))
            mu_seg = float(np.mean(race["mu"][inside])) if inside.any() else 1.0
            # feasibility projection: per station take the largest drift angle <= requested
            # that has an equilibrium with steering margin left for correction
            lock_margin = float(sg.get("lock_margin", 0.85)) * self.model.p.delta_max
            V = np.full_like(s, np.nan); dff = np.zeros_like(s); Tff = np.zeros_like(s)
            beta_used = np.zeros_like(s)
            for b in np.arange(beta_deg, 3.9, -2.0):
                tab = self._eq_table(float(b), mu_seg)
                Vb, db, Tb, okb = DriftEquilibrium.lookup(tab, kap_eq)
                take = np.isnan(V) & okb & (np.abs(db) <= lock_margin)
                V[take], dff[take], Tff[take], beta_used[take] = Vb[take], db[take], Tb[take], b
            miss = np.isnan(V)
            core = w > 0.9                      # the fully-drifting part of the segment
            if core.any() and miss[core].any():
                warnings.append(f"segment {sg.get('name')}: no drift equilibrium on "
                                f"{100*miss[core].mean():.0f}% of the segment; using race speed there")
            V[miss] = race["v"][miss]; beta_used[miss] = 0.0
            if core.any() and (beta_used[core] < beta_deg - 1e-6).any():
                lo = beta_used[core & ~miss].min() if (core & ~miss).any() else 0
                warnings.append(f"segment {sg.get('name')}: {beta_deg:.0f} deg not sustainable everywhere "
                                f"(steering lock / power); reduced to >= {lo:.0f} deg where needed")
            beta_used = gaussian_filter1d(beta_used, 10.0 / tr.ds, mode="wrap")
            V = gaussian_filter1d(V, 10.0 / tr.ds, mode="wrap")
            if "speed" in sg and sg["speed"] not in (None, "auto"):
                V = np.full_like(V, float(sg["speed"]))
            V = V * float(sg.get("speed_scale", 1.0))
            beta_ref = -sign * np.radians(beta_used)
            drift = dict(n=n_line, v=V * np.cos(beta_ref), beta=beta_ref, r=V * kap,
                         xi=-beta_ref, delta=dff, T=Tff, mu=race["mu"])
            for k in drift:
                out[k] = (1 - w) * out[k] + w * drift[k]
            out["mode"] = np.maximum(out["mode"], w)
        return out, warnings

    # ---------------- query ----------------
    def sample(self, s_query):
        with self._lock:
            arr = self.arrays
        return {k: self.track.interp(arr[k], s_query) for k in self.FIELDS}

    def weights(self):
        with self._lock:
            return copy.deepcopy(self.cfg["weights"])
