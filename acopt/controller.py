"""
Controller = state estimation + latency compensation + MPC + safety supervisor.

Supervisor states
  MPC        normal operation
  MPC_SAFE   excursion (off line, too much slip): same MPC with a grip haircut,
             back to MPC after `recover_s` of stable driving
  FALLBACK   solver down and last plan expired: geometric path follower
  SAFE_STOP  spin or loss of telemetry: brake, wheel straight
  RECOVER    after a stop: creep back to the line, then MPC_SAFE
  PAUSED     live "pause" command
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .env import Command, Measurement
from .mpc import SpatialMPC, MPCResult
from .track import Track, wrap
from .vehicle import VehicleModel, IX, NX


@dataclass
class SupervisorCfg:
    beta_margin_deg: float = 15.0      # allowed |beta| above the reference before abort
    beta_spin_deg: float = 65.0
    max_solver_fails: int = 3
    solver_retries: int = 1
    rescue_with_ipopt: bool = True     # retry with the robust interior-point solver
    max_plan_age_s: float = 0.5        # how long a stale-but-valid plan may be executed
    offtrack_m: float = 2.5
    stale_s: float = 0.25
    recover_s: float = 1.0
    fallback_speed_scale: float = 0.6
    safe_mu_scale: float = 0.85        # grip haircut in MPC_SAFE (speed refs drop ~8%)
    lookahead_s: float = 0.6


@dataclass
class GripEstCfg:
    enabled: bool = True
    gain: float = 1.5                  # 1/s, adaptation speed
    min_util: float = 0.55             # only adapt when |alpha| > min_util * alpha_peak
    lo: float = 0.6
    hi: float = 1.05
    lp_hz: float = 8.0                 # low-pass for differentiated accelerations


@dataclass
class ControllerCfg:
    dt_mpc: float = 0.05               # re-solve period
    latency_s: float = 0.03            # measurement->actuation delay to compensate
    tau_delta: float = 0.03            # first-order steering lag of the game/input chain [s]
    tau_T: float = 0.05                # first-order pedal lag [s]
    use_measured_actuators: bool = False   # True: take delta/T from telemetry instead of observer
    ds_min: float = 1.0
    ds_max: float = 3.5
    horizon_time: float = 1.2          # target horizon in seconds (sets ds)
    sup: SupervisorCfg = field(default_factory=SupervisorCfg)
    grip: GripEstCfg = field(default_factory=GripEstCfg)


class Controller:
    def __init__(self, track: Track, model: VehicleModel, profile, mpc: SpatialMPC,
                 cfg: ControllerCfg = ControllerCfg()):
        self.track, self.model, self.profile, self.mpc, self.cfg = track, model, profile, mpc, cfg
        self.idx = None
        self.state = "MPC"
        self.fails = 0
        self.t_stable = 0.0
        self.last_cmd = Command(0.0, 0.0)
        self.res: MPCResult | None = None
        self.t_plan = -1e9
        self.t_last_meas = None
        self.paused = False
        self.info = {}
        self.act = np.zeros(2)            # observer of ACTUAL (lagged) delta, T
        self.mu_hat = 1.0                 # online grip scale (global, fast)
        self._prev = None                 # (t, vx, vy, r) for differentiation
        self._acc = np.zeros(3)
        self.t_act = None

    # ---------------- estimation ----------------
    def frenet_state(self, m: Measurement):
        s, n, xi, self.idx = self.track.to_frenet(m.X, m.Y, m.psi, hint=self.idx)
        # actuator observer: the model's delta/T are the ACTUAL wheel angle / pedal
        if self.cfg.use_measured_actuators:
            self.act[:] = (m.delta, m.T)
        else:
            if self.t_act is not None:
                h = max(m.t - self.t_act, 0.0)
                c = self.last_cmd
                self.act[0] += (c.delta - self.act[0]) * (1 - np.exp(-h / max(self.cfg.tau_delta, 1e-3)))
                self.act[1] += (c.T - self.act[1]) * (1 - np.exp(-h / max(self.cfg.tau_T, 1e-3)))
            self.t_act = m.t
        x = np.array([n, xi, max(m.vx, self.model.p.v_min), m.vy, m.r,
                      self.act[0], self.act[1]])
        return s, x

    def predict(self, s, x, tau, u=(0.0, 0.0), steps=3):
        """Forward-integrate Frenet time dynamics by tau (latency compensation)."""
        if tau <= 0:
            return s, x
        h = tau / steps
        for _ in range(steps):
            k = float(self.track.kappa_at(s))
            xd, sd = self.model.f_time(x, u, k, 1.0)
            x = x + h * np.array(xd).ravel()
            s = s + h * float(sd)
        x[IX["delta"]] = np.clip(x[IX["delta"]], -self.model.p.delta_max, self.model.p.delta_max)
        x[IX["T"]] = np.clip(x[IX["T"]], -1, 1)
        x[IX["vx"]] = max(x[IX["vx"]], self.model.p.v_min)
        return s, x

    def update_grip(self, m: Measurement, x, mu_ref):
        """Online grip estimate: compare measured body accelerations with the model's
        prediction at the planned grip, only when the tyres are working hard enough for
        grip to be observable. Integral-type update, clipped, low-passed."""
        g = self.cfg.grip
        if self._prev is not None:
            dt = m.t - self._prev[0]
            if dt > 1e-4:
                raw = np.array([(m.vx - self._prev[1]) / dt, (m.vy - self._prev[2]) / dt,
                                (m.r - self._prev[3]) / dt])
                a = 1 - np.exp(-2 * np.pi * g.lp_hz * dt)
                self._acc += a * (raw - self._acc)
        self._prev = (m.t, m.vx, m.vy, m.r)
        if not g.enabled or m.vx < 10 or self._prev is None:
            return
        p = self.model.p
        d = self.model.diagnostics(x, mu_ref * self.mu_hat)
        util = max(abs(d["alpha_f"]), abs(d["alpha_r"])) / p.alpha_peak
        if util < g.min_util:
            return
        xb = [x[IX["vx"]], x[IX["vy"]], x[IX["r"]], x[IX["delta"]], x[IX["T"]]]
        ay_meas = self._acc[1] + m.r * m.vx
        ay_pred = float(np.array(self.model.f_body(xb, mu_ref * self.mu_hat)).ravel()[1]) + m.r * m.vx
        if abs(ay_pred) < 3.0:
            return
        ratio = np.clip(ay_meas / ay_pred, 0.5, 1.5)
        dt = 0.01 if self._prev is None else max(m.t - getattr(self, "_t_mu", m.t), 0.0)
        self._t_mu = m.t
        self.mu_hat = float(np.clip(self.mu_hat * (1 + g.gain * dt * (ratio - 1)), g.lo, g.hi))

    # ---------------- main entry ----------------
    def step(self, m: Measurement) -> tuple[Command, dict]:
        c, sc = self.cfg, self.cfg.sup
        s, x = self.frenet_state(m)
        ref = self.profile.sample(np.array([s]))
        beta = np.arctan2(m.vy, max(m.vx, 0.1))
        beta_allow = abs(ref["beta"][0]) + np.radians(sc.beta_margin_deg)
        wl, wr = self.track.bounds_at(s)
        off = max(x[0] - wl, -wr - x[0])
        stale = (not m.fresh) or (self.t_last_meas is not None and m.t - self.t_last_meas > sc.stale_s)
        self.t_last_meas = m.t
        info = dict(t=m.t, s=s, n=x[0], vx=m.vx, vy=m.vy, r=m.r, beta=beta,
                    v_ref=ref["v"][0], n_ref=ref["n"][0], beta_ref=ref["beta"][0],
                    mode=ref["mode"][0], tyres_out=m.tyres_out,
                    delta_act=float(x[IX["delta"]]), T_act=float(x[IX["T"]]))

        self.update_grip(m, x, float(ref["mu"][0]))
        info["mu_hat"] = self.mu_hat
        spinning = m.vx > 5 and abs(beta) > np.radians(sc.beta_spin_deg)
        if self.paused or stale or spinning:
            self.state = "PAUSED" if self.paused else "SAFE_STOP"
            cmd = Command(0.0, -0.6 if m.vx > 2 else 0.0)
            return self._out(cmd, info)
        if self.state == "SAFE_STOP":
            # wait until the car has (nearly) stopped and is calm, then creep back on line
            if m.vx < 3.0 and abs(m.r) < 0.3:
                self.state, self.t_stable = "RECOVER", 0.0
                self.mpc.reset()
            else:
                return self._out(Command(0.0, -0.6 if m.vx > 2 else 0.0), info)
        if self.state == "RECOVER":
            if m.vx > 12.0 and abs(x[IX["xi"]]) < 0.3 and off < 0:
                self.state, self.t_stable = "MPC_SAFE", 0.0
            else:
                return self._out(self._fallback(s, x, m, creep=True), info)
        if self.state == "PAUSED":
            self.state = "MPC_SAFE"

        # (re)plan on schedule. MPC_SAFE = same MPC with a grip haircut (slower refs,
        # more conservative model) - far better at catching a slide than a geometric law.
        if self.state in ("MPC", "MPC_SAFE", "FALLBACK") and \
                m.t - getattr(self, "_t_last_solve", -1e9) >= c.dt_mpc - 1e-9:
            ds = float(np.clip(x[IX["vx"]] * c.horizon_time / self.mpc.N, c.ds_min, c.ds_max))
            u_now = self.res.command_at(m.t - self.t_plan + c.latency_s)[2] if self.res else (0, 0)
            sp, xp = self.predict(s, x.copy(), c.latency_s, u_now)
            mu_s = self.mu_hat * (1.0 if self.state == "MPC" else sc.safe_mu_scale)
            res = self.mpc.solve(xp, sp, ds, self.track, self.profile, mu_scale=mu_s)
            for _ in range(sc.solver_retries):       # continue iterating from its own iterate
                if res.ok:
                    break
                res = self.mpc.solve(xp, sp, ds, self.track, self.profile, mu_scale=mu_s,
                                     rescue=sc.rescue_with_ipopt)
            # a plan that needs track-limit slack is still the best available plan:
            # execute it, but flag the excursion (-> MPC_SAFE)
            self.pred_slack = res.max_slack if res.ok else getattr(self, "pred_slack", 0.0)
            if res.ok:
                self.res, self.fails = res, 0
                self.t_plan = m.t + c.latency_s
            else:
                # keep executing the last good plan (time-shifted) while the solver
                # re-converges from its own iterate on the next ticks
                self.fails += 1
            info.update(solve_ms=res.solve_ms, mpc_ok=res.ok, slack=res.max_slack,
                        status=res.status, defect=res.defect)
            self._t_last_solve = m.t

        # supervisor transitions
        plan_age = m.t - self.t_plan
        plan_expired = self.res is None or plan_age > sc.max_plan_age_s
        solver_down = self.fails >= sc.max_solver_fails and plan_expired
        excursion = (off > sc.offtrack_m or abs(beta) > beta_allow
                     or getattr(self, "pred_slack", 0.0) > sc.offtrack_m)
        dt_tick = m.t - getattr(self, "_t_prev", m.t)
        why = "solver" if solver_down else ("offtrack" if off > sc.offtrack_m else
                                            ("beta" if abs(beta) > beta_allow else ""))
        if solver_down:
            if self.state != "FALLBACK":
                self.mpc.reset()
            self.state, self.t_stable = "FALLBACK", 0.0
        elif self.state == "MPC" and excursion:
            self.state, self.t_stable = "MPC_SAFE", 0.0
        elif self.state in ("MPC_SAFE", "FALLBACK"):
            self.t_stable = self.t_stable + dt_tick if not excursion else 0.0
            if self.state == "FALLBACK" and self.fails == 0 and self.res is not None:
                self.state = "MPC_SAFE"
            elif self.t_stable >= sc.recover_s:
                self.state = "MPC"
        info["why"] = why

        self._t_prev = m.t
        if self.state in ("MPC", "MPC_SAFE") and self.res is not None:
            # sample the plan where the command will actually land (transport delay),
            # then invert the first-order lag by leading the plan by the lag constant:
            # x(t + tau) ~= x + tau*x_dot, but smooth (no steps from piecewise-constant u)
            tau = m.t - self.t_plan + c.latency_s
            d = self.res.command_at(tau + c.tau_delta)[0]
            T = self.res.command_at(tau + c.tau_T)[1]
            cmd = Command(d, T)
        else:
            cmd = self._fallback(s, x, m)
        info["state"] = self.state
        return self._out(cmd, info)

    def _out(self, cmd, info):
        info.setdefault("state", self.state)
        p = self.model.p
        cmd.delta = float(np.clip(cmd.delta, -p.delta_max, p.delta_max))
        cmd.T = float(np.clip(cmd.T, -1, 1))
        self.last_cmd = cmd
        info.update(delta_cmd=cmd.delta, T_cmd=cmd.T)
        self.info = info
        return cmd, info

    def _fallback(self, s, x, m, creep=False):
        """Pure pursuit on the reference line + P speed control at reduced speed."""
        sc = self.cfg.sup
        p = self.model.p
        Ld = max(8.0, m.vx * sc.lookahead_s)
        if creep and abs(x[IX["xi"]]) > 1.2:          # facing the wrong way: aim at the line ahead
            Ld = 25.0
        ref = self.profile.sample(np.array([s + Ld]))
        tx, ty = self.track.to_world(s + Ld, ref["n"][0])
        dx, dy = tx - m.X, ty - m.Y
        alpha = wrap(np.arctan2(dy, dx) - m.psi)
        delta = np.arctan2(2 * (p.lf + p.lr) * np.sin(alpha), Ld)
        v_t = sc.fallback_speed_scale * self.profile.sample(np.array([s]))["v"][0]
        if creep:
            v_t = min(v_t, 15.0)
        # gentle: never stamp on the brakes mid-slide (that is what causes the spin)
        beta = np.arctan2(m.vy, max(m.vx, 0.1))
        T_min = -0.35 if abs(beta) < np.radians(8) else 0.0
        T = np.clip(0.1 * (v_t - m.vx), T_min, 0.5)
        return Command(delta, T)
