"""
Vehicle model: planar dynamic bicycle with
  * Pacejka-style lateral tyre curves per axle
  * friction-ellipse combined slip (throttle-on oversteer -> drift is in the model)
  * aerodynamic downforce / drag, longitudinal load transfer, tyre load sensitivity
  * power-limited drive, brake force with bias
  * steering and pedal as *states*, driven by rate inputs (actuator limits are hard)

Everything is written once in CasADi and reused by the planner, the MPC, the
drift-equilibrium solver, the simulator and system identification.

Conventions (right-handed, planar):
  body x forward, body y LEFT, yaw r positive counter-clockwise,
  steering delta positive = turn left, beta = atan(vy/vx).
  Frenet: s along track, n lateral offset (LEFT positive), xi = heading error.

State x (7):  [n, xi, vx, vy, r, delta, T]    T in [-1, 1]: +throttle / -brake
Input u (2):  [delta_dot, T_dot]             (time rates)
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields
from pathlib import Path

import casadi as ca
import numpy as np
import yaml

NX, NU = 7, 2
IX = dict(n=0, xi=1, vx=2, vy=3, r=4, delta=5, T=6)


@dataclass
class VehicleParams:
    name: str = "unnamed"
    # mass / geometry
    m: float = 650.0
    Iz: float = 900.0
    lf: float = 1.68
    lr: float = 1.37
    h_cg: float = 0.28
    car_width: float = 1.80
    g: float = 9.81
    # aero
    rho: float = 1.20
    ClA: float = 4.2
    CdA: float = 1.30
    aero_bal: float = 0.45          # fraction of downforce on front axle
    Crr: float = 0.012
    # tyres (per axle, lumped)
    mu: float = 1.70
    B_f: float = 13.0
    C_f: float = 1.50
    B_r: float = 13.0
    C_r: float = 1.50
    load_sens: float = 0.08         # mu drops this fraction per +100% load
    sat_k: float = 8.0              # sharpness of longitudinal force saturation
    # powertrain / brakes
    P_max: float = 650e3            # W at the wheels
    F_drive_max: float = 20e3       # N, low-gear torque limit at the wheels
    F_brake_max: float = 40e3       # N, total brake force at T = -1
    brake_bias: float = 0.56        # front fraction
    # actuators (road-wheel angle!)
    delta_max: float = 0.30         # rad
    ddelta_max: float = 1.5         # rad/s
    dT_max: float = 6.0             # 1/s
    v_min: float = 4.0              # m/s, model validity floor
    # mapping to the game (used by the I/O layer only)
    steer_lock_wheel_deg: float = 360.0   # full-lock steering-wheel angle in AC (lock to lock / 2)
    steer_ratio: float = 12.0
    extra: dict = field(default_factory=dict)

    # parameters that enter the dynamics, in a fixed order
    DYN = ("m", "Iz", "lf", "lr", "h_cg", "g", "rho", "ClA", "CdA", "aero_bal", "Crr",
           "mu", "B_f", "C_f", "B_r", "C_r", "load_sens", "sat_k",
           "P_max", "F_drive_max", "F_brake_max", "brake_bias")

    def vec(self) -> np.ndarray:
        return np.array([getattr(self, k) for k in self.DYN], dtype=float)

    def with_vec(self, v) -> "VehicleParams":
        d = asdict(self)
        for k, val in zip(self.DYN, v):
            d[k] = float(val)
        return VehicleParams(**d)

    @staticmethod
    def from_yaml(path) -> "VehicleParams":
        raw = yaml.safe_load(Path(path).read_text())
        known = {f.name for f in fields(VehicleParams)}
        kw = {k: v for k, v in raw.items() if k in known}
        kw["extra"] = {k: v for k, v in raw.items() if k not in known}
        return VehicleParams(**kw)

    def to_yaml(self, path):
        d = asdict(self)
        extra = d.pop("extra")
        d.update(extra)
        Path(path).write_text(yaml.safe_dump(d, sort_keys=False))

    @property
    def alpha_peak(self) -> float:
        """Slip angle at peak lateral force (rear) - used for stability margins."""
        return float(np.tan(np.pi / (2 * self.C_r)) / self.B_r)


NP = len(VehicleParams.DYN)


def _softplus(x, eps=1e-4):
    return 0.5 * (x + ca.sqrt(x * x + eps))


def _sat(x, k):
    """Smooth unit saturation, ~linear to 0.8, -> +-1."""
    return x / (1 + ca.fabs(x) ** k + 1e-9) ** (1.0 / k)


def tyre_forces(vx, vy, r, delta, T, P, mu_scale=1.0):
    """Returns dict of forces / diagnostics. All CasADi-compatible."""
    (m, Iz, lf, lr, h, g, rho, ClA, CdA, abal, Crr, mu, Bf, Cf, Br, Cr, ls, k,
     Pmax, Fdmax, Fbmax, bb) = [P[i] for i in range(NP)]
    L = lf + lr
    v2 = vx * vx
    Fdown = 0.5 * rho * ClA * v2
    Fdrag = 0.5 * rho * CdA * v2 + Crr * m * g

    Tp, Tn = _softplus(T), _softplus(-T)
    vpos = _softplus(vx - 1.0) + 1.0
    F_drive = Tp / (1.0 / Fdmax + vpos / Pmax)          # smooth min(Fdmax, P/v)
    F_brake = Tn * Fbmax
    Fx_f_cmd = -bb * F_brake
    Fx_r_cmd = F_drive - (1 - bb) * F_brake

    ax_est = (Fx_f_cmd + Fx_r_cmd - Fdrag) / m
    Fz_f = m * g * lr / L - m * ax_est * h / L + abal * Fdown
    Fz_r = m * g * lf / L + m * ax_est * h / L + (1 - abal) * Fdown
    Fz_f = _softplus(Fz_f - 200.0) + 200.0
    Fz_r = _softplus(Fz_r - 200.0) + 200.0
    Fz0_f, Fz0_r = m * g * lr / L, m * g * lf / L
    mu_f = mu * mu_scale * (1 - ls * (Fz_f / Fz0_f - 1))
    mu_r = mu * mu_scale * (1 - ls * (Fz_r / Fz0_r - 1))
    Fmax_f, Fmax_r = mu_f * Fz_f, mu_r * Fz_r

    # longitudinal: saturate at the friction limit (lock-up / wheelspin)
    rx_f = _sat(Fx_f_cmd / Fmax_f, k)
    rx_r = _sat(Fx_r_cmd / Fmax_r, k)
    Fx_f, Fx_r = rx_f * Fmax_f, rx_r * Fmax_r

    a_f = delta - ca.atan2(vy + lf * r, vpos)
    a_r = -ca.atan2(vy - lr * r, vpos)
    ell_f = ca.sqrt(_softplus(1 - rx_f * rx_f, 1e-6) + 1e-4)
    ell_r = ca.sqrt(_softplus(1 - rx_r * rx_r, 1e-6) + 1e-4)
    Fy_f = Fmax_f * ca.sin(Cf * ca.atan(Bf * a_f)) * ell_f
    Fy_r = Fmax_r * ca.sin(Cr * ca.atan(Br * a_r)) * ell_r
    return dict(Fx_f=Fx_f, Fx_r=Fx_r, Fy_f=Fy_f, Fy_r=Fy_r, Fz_f=Fz_f, Fz_r=Fz_r,
                Fdrag=Fdrag, Fdown=Fdown, a_f=a_f, a_r=a_r,
                rx_f_cmd=Fx_f_cmd / Fmax_f, rx_r_cmd=Fx_r_cmd / Fmax_r, vpos=vpos)


def body_dynamics(vx, vy, r, delta, T, P, mu_scale=1.0):
    F = tyre_forces(vx, vy, r, delta, T, P, mu_scale)
    m, Iz, lf, lr = P[0], P[1], P[2], P[3]
    c, s = ca.cos(delta), ca.sin(delta)
    Fxb = F["Fx_r"] + F["Fx_f"] * c - F["Fy_f"] * s - F["Fdrag"]
    Fyb = F["Fy_r"] + F["Fy_f"] * c + F["Fx_f"] * s
    Mz = lf * (F["Fy_f"] * c + F["Fx_f"] * s) - lr * F["Fy_r"]
    vx_dot = Fxb / m + r * vy
    vy_dot = Fyb / m - r * vx
    r_dot = Mz / Iz
    return vx_dot, vy_dot, r_dot, F, Fxb / m, Fyb / m


class VehicleModel:
    """Compiled CasADi functions for one parameter set (P may also stay symbolic)."""

    def __init__(self, params: VehicleParams):
        self.p = params
        x = ca.SX.sym("x", NX)
        u = ca.SX.sym("u", NU)
        kappa = ca.SX.sym("kappa")
        mu_s = ca.SX.sym("mu_s")
        P = ca.SX.sym("P", NP)
        n, xi, vx, vy, r, delta, T = [x[i] for i in range(NX)]
        vx_dot, vy_dot, r_dot, F, ax, ay = body_dynamics(vx, vy, r, delta, T, P, mu_s)
        sdot = (vx * ca.cos(xi) - vy * ca.sin(xi)) / (1 - n * kappa)
        xdot = ca.vertcat(vx * ca.sin(xi) + vy * ca.cos(xi),
                          r - kappa * sdot,
                          vx_dot, vy_dot, r_dot, u[0], u[1])
        # time-domain Frenet dynamics (+ sdot)
        self.f_time_P = ca.Function("f_time_P", [x, u, kappa, mu_s, P], [xdot, sdot])
        diag = ca.vertcat(ax, ay, F["a_f"], F["a_r"], F["rx_f_cmd"], F["rx_r_cmd"],
                          F["Fz_f"], F["Fz_r"], F["Fdown"])
        self.diag_P = ca.Function("diag_P", [x, mu_s, P], [diag])
        self.DIAG = ("ax", "ay", "alpha_f", "alpha_r", "rx_f", "rx_r", "Fz_f", "Fz_r", "Fdown")
        # body-only dynamics for sim / sysid / equilibria
        xb = ca.SX.sym("xb", 5)  # vx vy r delta T
        bd = body_dynamics(xb[0], xb[1], xb[2], xb[3], xb[4], P, mu_s)
        self.f_body_P = ca.Function("f_body_P", [xb, mu_s, P], [ca.vertcat(*bd[:3])])
        self.set_params(params)

    def set_params(self, params: VehicleParams):
        self.p = params
        Pv = ca.DM(params.vec())
        x = ca.SX.sym("x", NX); u = ca.SX.sym("u", NU)
        k = ca.SX.sym("k"); mu = ca.SX.sym("mu")
        xd, sd = self.f_time_P(x, u, k, mu, Pv)
        self.f_time = ca.Function("f_time", [x, u, k, mu], [xd, sd])
        self.diag = ca.Function("diag", [x, mu], [self.diag_P(x, mu, Pv)])
        xb = ca.SX.sym("xb", 5)
        self.f_body = ca.Function("f_body", [xb, mu], [self.f_body_P(xb, mu, Pv)])

    def f_space(self, x, u, kappa, mu_s=1.0):
        """dx/ds and dt/ds (symbolic-friendly)."""
        xd, sd = self.f_time(x, u, kappa, mu_s)
        return xd / sd, 1.0 / sd

    def rk4_space(self, x, u, ds, k0, k1, mu_s=1.0, substeps=1):
        """Integrate dx/ds over ds with curvature linearly varying k0->k1. Returns x+, dt."""
        h = ds / substeps
        t = 0
        for i in range(substeps):
            ka = k0 + (k1 - k0) * (i / substeps)
            km = k0 + (k1 - k0) * ((i + 0.5) / substeps)
            kb = k0 + (k1 - k0) * ((i + 1) / substeps)
            f1, t1 = self.f_space(x, u, ka, mu_s)
            f2, t2 = self.f_space(x + h / 2 * f1, u, km, mu_s)
            f3, t3 = self.f_space(x + h / 2 * f2, u, km, mu_s)
            f4, t4 = self.f_space(x + h * f3, u, kb, mu_s)
            x = x + h / 6 * (f1 + 2 * f2 + 2 * f3 + f4)
            t = t + h / 6 * (t1 + 2 * t2 + 2 * t3 + t4)
        return x, t

    def diagnostics(self, x, mu_s=1.0) -> dict:
        d = np.array(self.diag(x, mu_s)).ravel()
        return dict(zip(self.DIAG, d))
