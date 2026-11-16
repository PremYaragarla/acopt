"""
Steady-state drift equilibria.

For a path curvature kappa and a sideslip beta, find (V, delta, T) such that
vx_dot = vy_dot = r_dot = 0 with vx = V cos(beta), vy = V sin(beta), r = V*kappa.
Drift (rear saturated, counter-steer) and grip cornering are both equilibria of
the same model; drift solutions are the ones with |beta| large and the rear
axle past its peak. The table built here gives the MPC a speed reference and
feed-forward (delta, T) for any drift angle you ask for, live.

Sign convention: a LEFT-hand corner (kappa > 0) is drifted with beta < 0
(velocity points right of the nose). Profiles give |beta|; sign is applied here.
"""
from __future__ import annotations

import casadi as ca
import numpy as np

from .vehicle import VehicleModel


class DriftEquilibrium:
    def __init__(self, model: VehicleModel):
        self.model = model
        p = model.p
        z = ca.SX.sym("z", 3)                   # V, delta, T
        prm = ca.SX.sym("prm", 3)               # kappa, beta, mu
        V, d, T = z[0], z[1], z[2]
        k, b, mu = prm[0], prm[1], prm[2]
        xb = ca.vertcat(V * ca.cos(b), V * ca.sin(b), V * k, d, T)
        res = self.model.f_body(xb, mu)
        scale = ca.DM([1 / 5.0, 1 / 5.0, 1 / 2.0])
        J = ca.sumsqr(res * scale) + 1e-6 * T ** 2
        self.solver = ca.nlpsol("drift_eq", "ipopt", {"x": z, "p": prm, "f": J},
                                {"print_time": False, "ipopt.print_level": 0,
                                 "ipopt.max_iter": 200, "ipopt.tol": 1e-10})
        self.lbx = [p.v_min, -p.delta_max, -1.0]
        self.ubx = [90.0, p.delta_max, 1.0]
        self.res_fn = ca.Function("res", [z, prm], [res])

    def solve(self, kappa: float, beta_abs: float, mu: float = 1.0):
        """beta_abs in rad (magnitude). Returns dict(V, delta, T, beta, r, ok, residual)."""
        beta = -np.sign(kappa) * abs(beta_abs) if kappa != 0 else -abs(beta_abs)
        R = 1 / max(abs(kappa), 1e-4)
        sk = np.sign(kappa) if kappa != 0 else 1.0
        dmax = self.ubx[1]
        cands = []
        # counter-steer guesses first; stop as soon as that branch is found
        for d0 in (-sk * min(0.8 * abs(beta_abs), 0.9 * dmax), -sk * 0.05, sk * 0.05):
            for V0 in (np.sqrt(9.81 * R * 0.8), 25.0, 40.0, 15.0):
                for T0 in (0.4, 0.8):
                    sol = self.solver(x0=[min(max(V0, 6), 80), d0, T0], p=[kappa, beta, mu],
                                      lbx=self.lbx, ubx=self.ubx)
                    z = np.array(sol["x"]).ravel()
                    res = np.array(self.res_fn(z, [kappa, beta, mu])).ravel()
                    cands.append((np.max(np.abs(res)), z, res))
                    if cands[-1][0] < 0.02 and z[1] * sk < 0:
                        break
                else:
                    continue
                break
            if any(c[0] < 0.02 and c[1][1] * sk < 0 for c in cands):
                break
        good = [c for c in cands if c[0] < 0.02]
        if good:
            # several equilibria can exist; the classic drift is the counter-steer branch
            _, z, res = min(good, key=lambda c: c[1][1] * sk)
        else:
            _, z, res = min(cands, key=lambda c: c[0])
        V, d, T = z
        return dict(V=V, delta=d, T=T, beta=beta, r=V * kappa, ok=bool(good), residual=res)

    def max_feasible_beta(self, kappa: float, mu: float = 1.0, step_deg=2.0, upto=45):
        """Largest drift angle (rad) with a valid equilibrium at this curvature."""
        last, seen = 0.0, False
        for b in np.arange(6.0, upto + 1e-9, step_deg):
            ok = self.solve(kappa, np.radians(b), mu)["ok"]
            if ok:
                seen, last = True, np.radians(b)
            elif seen:
                break
        return last

    def table(self, beta_abs: float, kappas=None, mu: float = 1.0):
        """Equilibrium vs |kappa| for one drift angle. Returns arrays for interpolation."""
        if kappas is None:
            kappas = 1 / np.geomspace(12, 250, 18)
        rows = []
        for k in kappas:
            e = self.solve(float(k), beta_abs, mu)
            rows.append((k, e["V"], e["delta"], e["T"], e["ok"]))
        a = np.array(rows, float)
        return dict(kappa=a[:, 0], V=a[:, 1], delta=a[:, 2], T=a[:, 3], ok=a[:, 4].astype(bool),
                    beta_abs=beta_abs)

    @staticmethod
    def lookup(tab, kappa):
        """Signed lookup: mirror delta for right-hand corners."""
        ka = np.abs(kappa)
        order = np.argsort(tab["kappa"])
        kk = tab["kappa"][order]
        V = np.interp(ka, kk, tab["V"][order])
        d = np.interp(ka, kk, tab["delta"][order]) * np.sign(kappa + 1e-12)
        T = np.interp(ka, kk, tab["T"][order])
        ok = np.interp(ka, kk, tab["ok"][order].astype(float)) > 0.5
        return V, d, T, ok
