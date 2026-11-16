"""
Real-time nonlinear MPC in the spatial domain.

Why spatial: references (race line, drift angle, speed) are naturally functions
of track position, and stations move with the car, so every per-station
quantity - curvature, track limits, references, grip, mode blend - is a solver
PARAMETER. Weights are parameters too. Nothing is ever re-compiled while
driving: editing a drift angle, a weight or a whole profile takes effect on the
next solve (~20-50 ms later).

Decision variables: states at N+1 stations, rate inputs on N intervals,
one slack per station on the track limits (keeps the problem feasible after a
moment, e.g. when the car is already off line).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import casadi as ca
import numpy as np

from .vehicle import VehicleModel, NX, NU, IX

PER_STATION = ("kappa", "mu", "n_lo", "n_hi", "n", "v", "beta", "r", "xi", "delta", "T", "mode")
WKEYS = ("n", "v", "beta", "r", "xi", "time", "ff_delta", "ff_T", "ddelta", "dT")
NPS = len(PER_STATION)
D2R = np.pi / 180


@dataclass
class MPCResult:
    X: np.ndarray       # (N+1, NX)
    U: np.ndarray       # (N, NU)
    s: np.ndarray       # station s (absolute, may exceed track length)
    t: np.ndarray       # time at stations from now
    ok: bool
    status: str
    solve_ms: float
    cost: float
    max_slack: float
    defect: float = 0.0

    def command_at(self, tau):
        """Actuator command (delta, T) and rates tau seconds after the solve's x0."""
        tau = np.clip(tau, 0, self.t[-1])
        d = np.interp(tau, self.t, self.X[:, IX["delta"]])
        T = np.interp(tau, self.t, self.X[:, IX["T"]])
        k = min(np.searchsorted(self.t, tau, side="right") - 1, len(self.U) - 1)
        return float(d), float(T), self.U[max(k, 0)]


class SpatialMPC:
    def __init__(self, model: VehicleModel, N: int = 30, substeps: int = 2,
                 solver: str = "sqp", max_iter: int = 40, max_cpu_s: float = 0.08,
                 sqp_iter: int = 2):
        self.model, self.N = model, N
        p = model.p
        nx, nu = NX, NU
        X = ca.MX.sym("X", nx, N + 1)
        U = ca.MX.sym("U", nu, N)
        S = ca.MX.sym("S", N + 1)
        x0 = ca.MX.sym("x0", nx)
        ds = ca.MX.sym("ds")
        PS = ca.MX.sym("PS", NPS, N + 1)
        Wr = ca.MX.sym("Wr", len(WKEYS))
        Wd = ca.MX.sym("Wd", len(WKEYS))
        Wc = ca.MX.sym("Wc", 2)  # slack, terminal

        # one shooting interval as a compiled function (keeps the graph small)
        xs, us, dss = ca.SX.sym("x", nx), ca.SX.sym("u", nu), ca.SX.sym("ds")
        ka, kb, mus = ca.SX.sym("ka"), ca.SX.sym("kb"), ca.SX.sym("mu")
        xn, dt = model.rk4_space(xs, us, dss, ka, kb, mus, substeps)
        F = ca.Function("F", [xs, us, dss, ka, kb, mus], [xn, dt])
        # stage cost as a compiled function
        ps_s = ca.SX.sym("ps", NPS)
        wr_s, wd_s = ca.SX.sym("wr", len(WKEYS)), ca.SX.sym("wd", len(WKEYS))
        P = dict(zip(PER_STATION, [ps_s[i] for i in range(NPS)]))
        m = P["mode"]
        W = (1 - m) * wr_s + m * wd_s
        Wd_ = dict(zip(WKEYS, [W[i] for i in range(len(WKEYS))]))
        n, xi, vx, vy, r, dl, T = [xs[i] for i in range(nx)]
        beta = ca.atan(vy / vx)
        sdot = (vx * ca.cos(xi) - vy * ca.sin(xi)) / (1 - n * P["kappa"])
        cost = (Wd_["n"] * (n - P["n"]) ** 2
                + Wd_["v"] * (vx - P["v"]) ** 2
                + Wd_["beta"] * ((beta - P["beta"]) / D2R) ** 2
                + Wd_["r"] * ((r - P["r"]) / 0.1) ** 2
                + Wd_["xi"] * ((xi - P["xi"]) / D2R) ** 2
                + Wd_["ff_delta"] * ((dl - P["delta"]) / D2R) ** 2
                + Wd_["ff_T"] * ((T - P["T"]) / 0.1) ** 2
                + Wd_["time"] * dss / sdot * 10.0
                + Wd_["ddelta"] * us[0] ** 2 + Wd_["dT"] * us[1] ** 2)
        L = ca.Function("L", [xs, us, dss, ps_s, wr_s, wd_s], [cost])

        g, lbg, ubg = [], [], []
        J = 0
        g.append(X[:, 0] - x0); lbg += [0] * nx; ubg += [0] * nx
        tcum = [ca.MX(0)]
        for k in range(N):
            xn_k, dt_k = F(X[:, k], U[:, k], ds, PS[0, k], PS[0, k + 1], PS[1, k])
            g.append(X[:, k + 1] - xn_k); lbg += [0] * nx; ubg += [0] * nx
            tcum.append(tcum[-1] + dt_k)
            J += L(X[:, k], U[:, k], ds, PS[:, k], Wr, Wd)
        J += Wc[1] * L(X[:, N], ca.DM.zeros(nu), ds, PS[:, N], Wr, Wd)
        for k in range(N + 1):
            nk = X[IX["n"], k]
            g.append(nk - PS[2, k] + S[k]); lbg.append(0); ubg.append(ca.inf)
            g.append(PS[3, k] + S[k] - nk); lbg.append(0); ubg.append(ca.inf)
        J += Wc[0] * (ca.sum1(S) + ca.sumsqr(S))

        w = ca.vertcat(ca.vec(X), ca.vec(U), S)
        par = ca.vertcat(x0, ds, ca.vec(PS), Wr, Wd, Wc)
        self.nw, self.npar = w.shape[0], par.shape[0]
        # bounds
        xl = np.array([-np.inf, -np.pi / 2, p.v_min, -np.inf, -np.inf, -p.delta_max, -1.0])
        xu = np.array([np.inf, np.pi / 2, 100.0, np.inf, np.inf, p.delta_max, 1.0])
        ul = np.array([-p.ddelta_max, -p.dT_max]); uu = -ul
        self.lbw = np.concatenate([np.tile(xl, N + 1), np.tile(ul, N), np.zeros(N + 1)])
        self.ubw = np.concatenate([np.tile(xu, N + 1), np.tile(uu, N), np.full(N + 1, np.inf)])
        self.lbg, self.ubg = np.array(lbg, float), np.array(ubg, float)
        nlp = {"x": w, "p": par, "f": J, "g": ca.vertcat(*g)}
        self.solver_name = solver
        ipopt_opts = {"expand": True, "print_time": False,
                      "ipopt": {"print_level": 0, "max_iter": max_iter, "max_cpu_time": max_cpu_s,
                                "tol": 1e-4, "acceptable_tol": 1e-2, "acceptable_iter": 2,
                                "warm_start_init_point": "yes", "warm_start_bound_push": 1e-6,
                                "warm_start_mult_bound_push": 1e-6, "mu_init": 1e-3,
                                "linear_solver": "mumps", "sb": "yes"}}
        sqp_opts = {"expand": True, "print_time": False, "print_header": False, "print_status": False,
                    "print_iteration": False, "max_iter": sqp_iter,
                    "qpsol": "qrqp", "qpsol_options": {"print_iter": False, "print_header": False,
                                                        "error_on_fail": False},
                    "error_on_fail": False, "convexify_strategy": "regularize"}
        if solver == "ipopt":
            self.solver = ca.nlpsol("mpc", "ipopt", nlp, ipopt_opts)
        elif solver == "sqp":
            # fast real-time-iteration style primary solver
            self.solver = ca.nlpsol("mpc", "sqpmethod", nlp, sqp_opts)
        else:
            raise ValueError(solver)
        # robust (slower) rescue solver for hard moments: interior point, time-capped
        self.rescue = ca.nlpsol("mpc_rescue", "ipopt", nlp, ipopt_opts)
        self.T_fn = ca.Function("tl", [w, par], [ca.vertcat(*tcum)])
        self.g_fn = ca.Function("gv", [w, par], [ca.vertcat(*g)])
        # acceptable dynamics defect per state: n, xi, vx, vy, r, delta, T
        self.defect_scale = np.array([0.10, 0.02, 0.3, 0.3, 0.05, 0.01, 0.05])
        self._w0_store = None
        self._lam_g = self._lam_x = None
        self.last: MPCResult | None = None

    # ------------------------------------------------------------------
    def _unpack(self, w):
        N = self.N
        nX = NX * (N + 1)
        X = w[:nX].reshape(N + 1, NX)
        U = w[nX:nX + NU * N].reshape(N, NU)
        S = w[nX + NU * N:]
        return X, U, S

    def _pack(self, X, U, S):
        return np.concatenate([X.ravel(), U.ravel(), S])

    def solve(self, x0, s0, ds, track, profile, weights=None, mu_scale=1.0,
              rescue=False) -> MPCResult:
        """mu_scale: online grip estimate; scales the model grip AND the speed references
        (v ~ sqrt(mu) for grip-limited driving), so a slippery car slows down at once."""
        N = self.N
        ss = s0 + ds * np.arange(N + 1)
        ref = profile.sample(ss)
        wl, wr = track.bounds_at(ss)
        hw = 0.5 * self.model.p.car_width
        PS = np.zeros((NPS, N + 1))
        PS[0] = track.kappa_at(ss)
        PS[1] = ref["mu"] * mu_scale
        PS[2] = -wr + hw
        PS[3] = wl - hw
        for i, k in enumerate(("n", "v", "beta", "r", "xi", "delta", "T", "mode")):
            PS[4 + i] = ref[k]
        vs = np.sqrt(min(mu_scale, 1.0))
        PS[5] *= vs           # v ref
        PS[7] *= vs           # r ref = v * kappa
        Wt = weights or profile.weights()
        Wr = np.array([Wt["race"][k] for k in WKEYS])
        Wd = np.array([Wt["drift"][k] for k in WKEYS])
        Wc = np.array([Wt["common"]["slack"], Wt["common"]["terminal"]])
        par = np.concatenate([x0, [ds], PS.T.ravel(), Wr, Wd, Wc])

        w0 = self._warm(x0, ss, ref)
        args = dict(x0=w0, p=par, lbx=self.lbw, ubx=self.ubw, lbg=self.lbg, ubg=self.ubg)
        solver = self.rescue if rescue else self.solver
        if self._lam_g is not None and (rescue or self.solver_name == "ipopt"):
            args.update(lam_g0=self._lam_g, lam_x0=self._lam_x)
        t0 = time.perf_counter()
        sol = solver(**args)
        ms = (time.perf_counter() - t0) * 1e3
        st = solver.stats()
        status = st.get("return_status", "?")
        w = np.array(sol["x"]).ravel()
        # accept any iterate (RTI / time-capped) whose dynamics defects are small:
        # what matters in real time is a consistent, constraint-respecting plan
        gv = np.array(self.g_fn(w, par)).ravel()
        viol = np.maximum(self.lbg - gv, 0) + np.maximum(gv - self.ubg, 0)
        ndyn = NX * (self.N + 1)
        dyn = viol[:ndyn].reshape(self.N + 1, NX) / self.defect_scale
        # the first part of the plan is what gets executed: judge it strictly
        head = dyn[: max(3, self.N // 3)]
        self.last_defect = float(np.max(head)) if np.all(np.isfinite(dyn)) else np.inf
        ok = bool(np.isfinite(self.last_defect) and self.last_defect < 1.0)
        X, U, S = self._unpack(w)
        tl = np.array(self.T_fn(w, par)).ravel()
        finite = np.all(np.isfinite(w)) and np.all(np.diff(tl) > 0)
        ok = bool(ok and finite)
        if finite:
            self._w0_store = (ss, X, U)
            self._lam_g = np.array(sol["lam_g"]).ravel()
            self._lam_x = np.array(sol["lam_x"]).ravel()
        res = MPCResult(X, U, ss, tl, ok, status, ms, float(sol["f"]), float(np.max(S)),
                        self.last_defect)
        self.last = res
        return res

    def _warm(self, x0, ss, ref):
        N = self.N
        st = getattr(self, "_w0_store", None)
        if st is not None:
            ps, PX, PU = st
            if ps[0] - 1 <= ss[0] <= ps[-1]:
                X = np.stack([np.interp(ss, ps, PX[:, i]) for i in range(NX)], 1)
                U = np.stack([np.interp(ss[:-1], ps[:-1], PU[:, i]) for i in range(NU)], 1)
                X[0] = x0
                return self._pack(X, U, np.zeros(N + 1))
        # cold start: roll the reference
        X = np.zeros((N + 1, NX))
        X[:] = x0
        X[:, IX["n"]] = np.linspace(x0[0], ref["n"][-1], N + 1)
        X[:, IX["vx"]] = np.maximum(np.linspace(x0[2], ref["v"][-1], N + 1), self.model.p.v_min + 0.5)
        self._lam_g = self._lam_x = None
        return self._pack(X, np.zeros((N, NU)), np.zeros(N + 1))

    def reset(self):
        self._w0_store = None
        self._lam_g = self._lam_x = None
