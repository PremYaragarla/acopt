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


SC = np.array([1.0, 0.1, 30.0, 2.0, 0.5, 0.1, 0.5])   # state scaling for conditioning
USC = np.array([0.5, 2.0])


def _solve_ocp(model, params, ds, kap, wl, wr, mu, X0, U0, periodic, x_start=None, x_end=None,
               margin=0.4, beta_max_deg=8.0, reg=1e-3, substeps=2, max_iter=1500,
               print_level=0, warm=False):
    """Minimum-time OCP over M intervals of length ds.
    periodic: M stations, state M wraps to state 0 (kap/wl/wr/mu have M entries).
    open:     M+1 stations (arrays have M+1 entries); x_start / x_end optionally pin the ends.
    Returns (X, U, time, status) with X of shape (M or M+1, NX), U of shape (M, NU)."""
    M = len(U0)
    nX = M if periodic else M + 1
    opti = ca.Opti()
    X = opti.variable(NX, nX)
    U = opti.variable(NU, M)
    Xs = ca.diag(ca.DM(SC)) @ X
    Us = ca.diag(ca.DM(USC)) @ U
    T_total = 0
    hw = 0.5 * params.car_width + margin
    a_pk = params.alpha_peak
    b_max = np.radians(beta_max_deg)

    def path(k):
        xk = Xs[:, k]
        n, xi, vx, vy, r, d, Tc = [xk[i] for i in range(NX)]
        lo, hi = -wr[k] + hw, wl[k] - hw
        if lo > hi:                       # track narrower than the car + margins: use the middle
            lo = hi = 0.5 * (lo + hi)
        opti.subject_to(opti.bounded(lo - 1e-3, n, hi + 1e-3))
        opti.subject_to(opti.bounded(params.v_min + 2, vx, 110.0))
        opti.subject_to(opti.bounded(-params.delta_max, d, params.delta_max))
        opti.subject_to(opti.bounded(-1, Tc, 1))
        # race line stays on the stable side of the tyre curve (robust to model error)
        opti.subject_to(opti.bounded(-b_max * vx, vy, b_max * vx))
        dg = model.diag(xk, mu[k])
        opti.subject_to(opti.bounded(-1.1 * a_pk, dg[2], 1.1 * a_pk))
        opti.subject_to(opti.bounded(-1.1 * a_pk, dg[3], 1.1 * a_pk))
        opti.subject_to(opti.bounded(-1.0, dg[4], 1.0))
        opti.subject_to(opti.bounded(-1.0, dg[5], 1.0))

    pin_start, pin_end = x_start is not None, x_end is not None
    for k in range(M):
        kn = (k + 1) % M if periodic else k + 1
        xk, uk = Xs[:, k], Us[:, k]
        xn, dt = model.rk4_space(xk, uk, ds, kap[k], kap[kn], mu[k], substeps)
        opti.subject_to(Xs[:, kn] == xn)
        T_total = T_total + dt
        opti.subject_to(opti.bounded(-params.ddelta_max, uk[0], params.ddelta_max))
        opti.subject_to(opti.bounded(-params.dT_max, uk[1], params.dT_max))
        if not (k == 0 and pin_start):
            path(k)
    if not periodic and not pin_end:
        path(M)
    if pin_start:
        opti.subject_to(Xs[:, 0] == ca.DM(x_start))
    if pin_end:
        opti.subject_to(Xs[:, M] == ca.DM(x_end))
    opti.minimize(T_total + reg * ca.sumsqr(U) * ds / M)
    X0 = np.array(X0, float)
    if pin_start:
        X0[0] = x_start
    opti.set_initial(X, (X0 / SC).T)
    opti.set_initial(U, (np.asarray(U0) / USC).T)
    opti.solver("ipopt", {"print_time": False, "expand": True},
                {"max_iter": max_iter, "print_level": print_level, "tol": 1e-6,
                 "acceptable_tol": 1e-4, "mu_strategy": "adaptive",
                 "warm_start_init_point": "yes" if warm else "no",
                 "linear_solver": "mumps"})
    try:
        sol = opti.solve()
        status = "optimal"
    except RuntimeError:
        sol = opti.debug
        status = sol.stats().get("return_status", "failed")
    Xv = np.array(sol.value(X)).T * SC
    Uv = np.array(sol.value(U)).T * USC
    return Xv, Uv, float(sol.value(T_total)), status


def plan_min_lap_time(track: Track, params: VehicleParams, ds: float = 4.0,
                      mu_map=None, warm: RaceLine | None = None, margin: float = 0.4,
                      beta_max_deg: float = 8.0, reg: float = 1e-3, substeps: int = 2,
                      max_iter: int = 1500, verbose: bool = True,
                      window_m: float | None = None, overlap_m: float = 600.0,
                      max_stations: int = 1200, progress=None) -> RaceLine:
    """Whole lap in one periodic problem for short tracks. Long tracks (more than
    `max_stations` stations, or when `window_m` is given) are planned in overlapping
    windows and stitched - memory and time then grow linearly with track length."""
    model = VehicleModel(params)
    N = int(round(track.length / ds))
    ds = track.length / N
    s = np.arange(N) * ds
    kap = track.kappa_at(s)
    wl, wr = track.bounds_at(s)
    mu = np.ones(N) if mu_map is None else np.interp(s, mu_map[0], mu_map[1], period=track.length)
    if warm is not None and len(warm.s) > 3:
        X0 = np.stack([np.interp(s, warm.s, warm.X[:, i], period=track.length) for i in range(NX)], 1)
        U0 = np.stack([np.interp(s, warm.s, warm.U[:, i], period=track.length) for i in range(NU)], 1)
    else:
        X0, U0 = initial_guess(track, params, s, mu)
    kw = dict(margin=margin, beta_max_deg=beta_max_deg, reg=reg, substeps=substeps, max_iter=max_iter,
              warm=warm is not None)
    t0 = time.time()
    windowed = window_m is not None or N > max_stations
    if not windowed:
        Xv, Uv, lap, status = _solve_ocp(model, params, ds, kap, wl, wr, mu, X0, U0, True,
                                         print_level=5 if verbose else 0, **kw)
        nwin = 1
    else:
        Xv, Uv, status, nwin = _plan_windowed(model, params, ds, kap, wl, wr, mu, X0, U0,
                                              window_m or 2000.0, overlap_m, kw, progress, verbose)
    # recompute timeline (and, for a stitched plan, the lap time) by integrating the result
    t = np.zeros(N + 1)
    for k in range(N):
        _, dt = model.rk4_space(Xv[k], Uv[k], ds, kap[k], kap[(k + 1) % N], mu[k], substeps)
        t[k + 1] = t[k] + float(dt)
    if windowed:
        lap = float(t[N])
    meta = dict(status=status, solve_s=round(time.time() - t0, 2), ds=ds, car=params.name,
                track=track.name, margin=margin, windows=nwin)
    if verbose:
        print(f"[planner] {status}: lap {lap:.3f} s, {N} stations, {nwin} window(s), {meta['solve_s']} s")
    return RaceLine(s, Xv, Uv, t[:N], lap, mu, meta)


def _plan_windowed(model, params, ds, kap, wl, wr, mu, X0, U0, window_m, overlap_m, kw, progress, verbose):
    """Receding windows around the lap. Window i is solved from its start (pinned to where
    window i-1 left the car) to `overlap_m` beyond its end, so braking for what comes next
    is already in the plan; only the part before the overlap is kept. The first window
    starts unpinned, so it is solved again at the end, pinned at both ends, to close the lap."""
    N = len(kap)
    nwin = max(2, int(np.ceil(N * ds / window_m)))
    edges = np.linspace(0, N, nwin + 1).round().astype(int)
    ov = min(int(round(overlap_m / ds)), int(edges[1]) - 1)
    X, U = np.array(X0, float), np.array(U0, float)
    worst = "optimal"

    def window(i, x_start, x_end=None):
        a, b = int(edges[i]), int(edges[i + 1])
        e = b if x_end is not None else b + ov
        idx = np.arange(a, e + 1) % N
        Xw, Uw, _, st = _solve_ocp(model, params, ds, kap[idx], wl[idx], wr[idx], mu[idx],
                                   X[idx], U[idx[:-1]], False, x_start=x_start, x_end=x_end, **kw)
        return Xw, Uw, st, a, b

    # first pass of window 0 starts from the (conservative) initial guess; the closing
    # solve below replaces it with the state the car really arrives in
    x_start = X[0].copy()
    for i in range(nwin):
        if progress:
            progress(i, nwin + 1)
        Xw, Uw, st, a, b = window(i, x_start)
        if verbose:
            print(f"[planner] window {i + 1}/{nwin}: s {a * ds:.0f}-{b * ds:.0f} m, {st}", flush=True)
        if st != "optimal":
            worst = st if np.all(np.isfinite(Xw)) else "failed"
        m = b - a
        X[a:b], U[a:b] = Xw[:m], Uw[:m]
        x_start = Xw[m]                       # state at the next window's first station
    # close the lap: first window again, from where the last one arrives to where the second starts
    if progress:
        progress(nwin, nwin + 1)
    b = int(edges[1])
    Xw, Uw, st, _, _ = window(0, x_start, x_end=X[b])
    if st == "optimal" and np.all(np.isfinite(Xw)):
        X[0:b], U[0:b] = Xw[:b], Uw[:b]
    else:
        # could not reach the second window's entry state exactly: let the end go free instead
        Xw, Uw, st, _, _ = window(0, x_start)
        if np.all(np.isfinite(Xw)) and st in ("optimal", "Solve_Succeeded", "Solved_To_Acceptable_Level"):
            X[0:b], U[0:b] = Xw[:b], Uw[:b]
        elif worst == "optimal":
            worst = st
    if verbose:
        print(f"[planner] lap closure: {st}", flush=True)
    return X, U, worst, nwin
