"""
Minimum-lap-time planner (offline / between laps).

Direct multiple shooting in the SPATIAL domain: stations every ds metres,
RK4 in s, periodic boundary conditions, objective = lap time.
Optimises the line AND the controls jointly, under the full vehicle model and
actuator rate limits. A per-station grip map mu(s) lets the learning loop
reshape the plan lap after lap; warm-starting makes re-plans fast.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from pathlib import Path

import casadi as ca
import numpy as np

from .track import Track
from .vehicle import VehicleModel, VehicleParams, NX, NU, IX


@dataclass
class RaceLine:
    s: np.ndarray
    X: np.ndarray        # (N, NX) states at stations
    U: np.ndarray        # (N, NU)
    t: np.ndarray        # cumulative time at stations
    lap_time: float
    mu: np.ndarray       # grip map used
    meta: dict

    def col(self, name):
        return self.X[:, IX[name]]

    def save(self, path):
        """Atomic write (a crash mid-save must not leave a corrupt cache behind)."""
        import json
        path = Path(path)
        tmp = path.with_name(path.stem + ".tmp.npz")
        meta = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in self.meta.items()}
        np.savez(tmp, s=self.s, X=self.X, U=self.U, t=self.t, lap_time=self.lap_time,
                 mu=self.mu, meta=np.array(json.dumps(meta, default=str)))
        tmp.replace(path)

    @staticmethod
    def load(path):
        import json
        d = np.load(path, allow_pickle=False)
        try:
            meta = json.loads(str(d["meta"]))
        except (ValueError, KeyError):
            meta = {}
        meta["loaded_from"] = str(path)
        rl = RaceLine(d["s"], d["X"], d["U"], d["t"], float(d["lap_time"]), d["mu"], meta)
        if rl.X.ndim != 2 or rl.X.shape[1] != NX or not np.all(np.isfinite(rl.X)):
            raise ValueError("race line cache has an unexpected shape or non-finite values")
        return rl

    def world(self, track: Track):
        return track.to_world(self.s, self.col("n"))


def initial_guess(track: Track, p: VehicleParams, s, mu):
    """Curvature-limited speed with forward/backward acceleration passes."""
    k = np.abs(track.kappa_at(s)) + 1e-4
    g = p.g
    # steady cornering with downforce: v^2 k = mu (g + a v^2)  ->  v^2 = mu g / (k - mu a)
    a_aero = 0.5 * p.rho * p.ClA / p.m
    mu_e = 0.8 * mu * p.mu
    den = k - mu_e * a_aero
    v = np.where(den > 1e-5, np.sqrt(mu_e * g / np.maximum(den, 1e-5)), 85.0)
    v = np.minimum(v, 85.0)
    ds = np.diff(np.concatenate([s, [s[0] + track.length]]))
    for _ in range(2):
        for i in range(len(v)):            # accel
            j = (i + 1) % len(v)
            v[j] = min(v[j], np.sqrt(v[i] ** 2 + 2 * 8.0 * ds[i]))
        for i in range(len(v) - 1, -1, -1):  # brake
            j = (i + 1) % len(v)
            v[i] = min(v[i], np.sqrt(v[j] ** 2 + 2 * 25.0 * ds[i]))
    X = np.zeros((len(s), NX))
    X[:, IX["vx"]] = np.maximum(v, p.v_min + 1)
    kk = track.kappa_at(s)
    X[:, IX["r"]] = X[:, IX["vx"]] * kk
    X[:, IX["delta"]] = np.clip((p.lf + p.lr) * kk, -p.delta_max, p.delta_max)
    return X, np.zeros((len(s), NU))


def plan_min_lap_time(track: Track, params: VehicleParams, ds: float = 4.0,
                      mu_map=None, warm: RaceLine | None = None, margin: float = 0.4,
                      beta_max_deg: float = 8.0, reg: float = 1e-3, substeps: int = 2,
                      max_iter: int = 1500, verbose: bool = True) -> RaceLine:
    model = VehicleModel(params)
    N = int(round(track.length / ds))
    ds = track.length / N
    s = np.arange(N) * ds
    kap = track.kappa_at(s)
    wl, wr = track.bounds_at(s)
    mu = np.ones(N) if mu_map is None else np.interp(s, mu_map[0], mu_map[1], period=track.length)

    opti = ca.Opti()
    X = opti.variable(NX, N)
    U = opti.variable(NU, N)
    # scaling: optimise in normalised units for conditioning
    sc = np.array([1.0, 0.1, 30.0, 2.0, 0.5, 0.1, 0.5])
    usc = np.array([0.5, 2.0])
    Xs = ca.diag(ca.DM(sc)) @ X
    Us = ca.diag(ca.DM(usc)) @ U

    T_total = 0
    hw = 0.5 * params.car_width + margin
    for k in range(N):
        kn = (k + 1) % N
        xk, uk = Xs[:, k], Us[:, k]
        xn, dt = model.rk4_space(xk, uk, ds, kap[k], kap[kn], mu[k], substeps)
        opti.subject_to(Xs[:, kn] == xn)
        T_total = T_total + dt
        n, xi, vx, vy, r, d, Tc = [xk[i] for i in range(NX)]
        opti.subject_to(opti.bounded(-wr[k] + hw, n, wl[k] - hw))
        opti.subject_to(vx >= params.v_min + 2)
        opti.subject_to(opti.bounded(-params.delta_max, d, params.delta_max))
        opti.subject_to(opti.bounded(-1, Tc, 1))
        opti.subject_to(opti.bounded(-params.ddelta_max, uk[0], params.ddelta_max))
        opti.subject_to(opti.bounded(-params.dT_max, uk[1], params.dT_max))
        # race line stays on the stable side of the tyre curve (robust to model error)
        opti.subject_to(opti.bounded(-np.radians(beta_max_deg) * vx, vy, np.radians(beta_max_deg) * vx))
        dg = model.diag(xk, mu[k])
        a_pk = params.alpha_peak
        opti.subject_to(opti.bounded(-1.1 * a_pk, dg[2], 1.1 * a_pk))
        opti.subject_to(opti.bounded(-1.1 * a_pk, dg[3], 1.1 * a_pk))
        opti.subject_to(opti.bounded(-1.0, dg[4], 1.0))
        opti.subject_to(opti.bounded(-1.0, dg[5], 1.0))
    J = T_total + reg * ca.sumsqr(U) * ds / N
    opti.minimize(J)

    if warm is not None and len(warm.s) > 3:
        X0 = np.stack([np.interp(s, warm.s, warm.X[:, i], period=track.length) for i in range(NX)], 1)
        U0 = np.stack([np.interp(s, warm.s, warm.U[:, i], period=track.length) for i in range(NU)], 1)
    else:
        X0, U0 = initial_guess(track, params, s, mu)
    opti.set_initial(X, (X0 / sc).T)
    opti.set_initial(U, (U0 / usc).T)
    opti.solver("ipopt", {"print_time": False, "expand": True},
                {"max_iter": max_iter, "print_level": 5 if verbose else 0, "tol": 1e-6,
                 "acceptable_tol": 1e-4, "mu_strategy": "adaptive",
                 "warm_start_init_point": "yes" if warm is not None else "no",
                 "linear_solver": "mumps"})
    t0 = time.time()
    try:
        sol = opti.solve()
        status = "optimal"
    except RuntimeError:
        sol = opti.debug
        status = sol.stats().get("return_status", "failed")
    Xv = (np.array(sol.value(X)).T) * sc
    Uv = (np.array(sol.value(U)).T) * usc
    # recompute timeline
    t = np.zeros(N)
    for k in range(N - 1):
        _, dt = model.rk4_space(Xv[k], Uv[k], ds, kap[k], kap[k + 1], mu[k], substeps)
        t[k + 1] = t[k] + float(dt)
    lap = float(sol.value(T_total))
    meta = dict(status=status, solve_s=round(time.time() - t0, 2), ds=ds, car=params.name,
                track=track.name, margin=margin)
    if verbose:
        print(f"[planner] {status}: lap {lap:.3f} s, {N} stations, {meta['solve_s']} s")
    return RaceLine(s, Xv, Uv, t, lap, mu, meta)
