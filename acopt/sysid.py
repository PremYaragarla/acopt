"""
System identification from driving logs (sim or Assetto Corsa).

Fits a chosen subset of VehicleParams so the model's body accelerations match
the measured ones:  min sum || f_body(x, u; p) - [vx_dot, vy_dot, r_dot] ||^2
with a prior pulling toward the starting values (keeps unobservable params sane).

Kunos cars ship encrypted data (data.acd), so for the F2004 you identify the
model from telemetry instead of reading car.ini/tyres.ini. Collect a few laps
with varied driving (steady corners, braking zones, some slides) for good
excitation - the fallback/manual driving modes are fine for that.
"""
from __future__ import annotations

import casadi as ca
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

from .vehicle import VehicleModel, VehicleParams, NP

DEFAULT_FIT = ("mu", "B_f", "B_r", "C_f", "C_r", "ClA", "CdA", "Iz", "P_max", "aero_bal")


def prepare(logs: list[dict], dt_target=0.01, window_s=0.15):
    t = np.array([l["t"] for l in logs])
    keep = np.r_[True, np.diff(t) > 1e-6]
    t = t[keep]
    get = lambda k: np.array([l[k] for l in logs])[keep]
    vx, vy, r = get("vx"), get("vy"), get("r")
    d, T = get("delta_act") if "delta_act" in logs[0] else get("delta_cmd"), \
        get("T_act") if "T_act" in logs[0] else get("T_cmd")
    tu = np.arange(t[0], t[-1], dt_target)
    I = lambda a: np.interp(tu, t, a)
    vx, vy, r, d, T = map(I, (vx, vy, r, d, T))
    w = max(5, int(window_s / dt_target) | 1)
    D = lambda a: savgol_filter(a, w, 3, deriv=1, delta=dt_target)
    S = lambda a: savgol_filter(a, w, 3)
    XB = np.c_[S(vx), S(vy), S(r), S(d), S(T)]
    Y = np.c_[D(vx), D(vy), D(r)]
    m = XB[:, 0] > 8.0
    return XB[m], Y[m]


def identify(params: VehicleParams, XB, Y, fit=DEFAULT_FIT, prior_weight=0.05,
             max_points=4000, verbose=True) -> VehicleParams:
    model = VehicleModel(params)
    if len(XB) > max_points:
        idx = np.linspace(0, len(XB) - 1, max_points).astype(int)
        XB, Y = XB[idx], Y[idx]
    names = VehicleParams.DYN
    fi = [names.index(k) for k in fit]
    p0 = params.vec()
    Fm = model.f_body_P.map(len(XB))
    xb = ca.DM(XB.T)
    scale = np.array([5.0, 3.0, 1.0])

    def resid(z):
        P = p0.copy()
        P[fi] = p0[fi] * z
        pred = np.array(Fm(xb, 1.0, ca.repmat(ca.DM(P), 1, len(XB)))).T
        r = ((pred - Y) / scale).ravel()
        return np.concatenate([r / np.sqrt(len(XB)), prior_weight * (z - 1)])

    z0 = np.ones(len(fi))
    sol = least_squares(resid, z0, bounds=(0.3 * z0, 3.0 * z0), x_scale="jac", verbose=0)
    Pn = p0.copy()
    Pn[fi] = p0[fi] * sol.x
    out = params.with_vec(Pn)
    if verbose:
        r0, r1 = np.linalg.norm(resid(z0)), np.linalg.norm(resid(sol.x))
        print(f"[sysid] residual {r0:.4f} -> {r1:.4f}")
        for k, z in zip(fit, sol.x):
            print(f"   {k:10s} {getattr(params, k):10.4g} -> {getattr(out, k):10.4g}  (x{z:.3f})")
    return out
