"""
Session: wires config -> track, car model, race line, profile, MPC, controller,
environment (sim or AC), live tuning and lap-to-lap learning, and runs laps.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import fields
from pathlib import Path

import numpy as np
import yaml

from .controller import Controller, ControllerCfg, SupervisorCfg, GripEstCfg
from .learning import GripLearner, LearnCfg
from .mpc import SpatialMPC
from .planner import RaceLine, plan_min_lap_time
from .profile import ReferenceProfile
from .track import Track
from .vehicle import VehicleModel, VehicleParams


def _dc(cls, d):
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in (d or {}).items() if k in names})


def load_track(tc: dict, root: Path) -> Track:
    src = tc.get("source", "synthetic")
    if src == "synthetic":
        return Track.synthetic_club(tc.get("half_width", 6.0))
    path = Path(tc["path"])
    path = path if path.is_absolute() else root / path
    if src == "csv":
        return Track.from_csv(path)
    if src == "fast_lane_ai":
        return Track.from_fast_lane_ai(path, sign_z=tc.get("sign_z", -1.0),
                                       swap_sides=tc.get("swap_sides", False),
                                       default_half_width=tc.get("half_width", 6.0),
                                       subsample=tc.get("subsample", 4))
    raise ValueError(src)


class Session:
    def __init__(self, cfg_path, overrides: dict | None = None):
        self.cfg_path = Path(cfg_path)
        self.root = self.cfg_path.parent.parent if self.cfg_path.parent.name == "configs" else self.cfg_path.parent
        cfg = yaml.safe_load(self.cfg_path.read_text())
        for k, v in (overrides or {}).items():
            cfg.setdefault(k, {}).update(v) if isinstance(v, dict) else cfg.__setitem__(k, v)
        self.cfg = cfg
        R = lambda p: p if Path(p).is_absolute() else self.root / p

        self.params = VehicleParams.from_yaml(R(cfg["vehicle"]))
        self.model = VehicleModel(self.params)
        self.track = load_track(cfg["track"], self.root)
        print("[session]", self.track.summary())

        rc = cfg.get("raceline", {})
        self.rl_path = R(rc.get("cache", f"out/{self.track.name}_{self.params.name}.npz"))
        self.rl_path.parent.mkdir(parents=True, exist_ok=True)
        self.plan_kw = dict(ds=rc.get("ds", 4.0), margin=rc.get("margin", 0.4),
                            beta_max_deg=rc.get("beta_max_deg", 8.0))
        lc = cfg.get("learning", {})
        self.learner = GripLearner(self.track, self.model, _dc(LearnCfg, lc), mu0=lc.get("mu0", 1.0))
        self.learning = lc.get("enabled", False)
        if self.rl_path.exists() and not rc.get("replan", False):
            self.raceline = RaceLine.load(self.rl_path)
            print(f"[session] loaded race line {self.rl_path.name}: {self.raceline.lap_time:.3f} s")
        else:
            self.raceline = plan_min_lap_time(self.track, self.params, mu_map=self.learner.mu_map(),
                                              **self.plan_kw)
            self.raceline.save(self.rl_path)

        prof_path = cfg.get("profile")
        self.profile_path = R(prof_path) if prof_path else None
        pcfg = ReferenceProfile.load_yaml(self.profile_path) if self.profile_path else {}
        self.profile = ReferenceProfile(self.track, self.raceline, self.model, pcfg)

        mc = cfg.get("mpc", {})
        self.mpc = SpatialMPC(self.model, N=mc.get("N", 20), solver=mc.get("solver", "sqp"),
                              sqp_iter=mc.get("sqp_iter", 2), max_iter=mc.get("max_iter", 40))
        cc = cfg.get("controller", {})
        ccfg = _dc(ControllerCfg, cc)
        ccfg.sup = _dc(SupervisorCfg, cc.get("supervisor", {}))
        ccfg.grip = _dc(GripEstCfg, cc.get("grip", {}))
        self.controller = Controller(self.track, self.model, self.profile, self.mpc, ccfg)
        self.env = self._make_env(cfg.get("env", {"type": "sim"}))
        self._stop = threading.Event()
        self.lap_times: list[float] = []
        self.out_dir = R(cfg.get("log_dir", "out/logs"))
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._replan_thread = None

    # ------------------------------------------------------------------
    def _make_env(self, ec):
        if ec.get("type", "sim") == "sim":
            from .sim import SimEnv
            truth = self.params
            pert = ec.get("truth_scale", {})       # e.g. {mu: 0.93, ClA: 0.9}
            if pert:
                d = {k: getattr(truth, k) * v for k, v in pert.items()}
                truth = truth.with_vec([d.get(k, getattr(truth, k)) for k in VehicleParams.DYN])
            grip = None
            if ec.get("grip_patch"):
                a, b, f = ec["grip_patch"]          # [s_start, s_end, factor]
                L = self.track.length
                grip = lambda s: f if (s - a) % L <= (b - a) % L else 1.0
            return SimEnv(self.track, truth, delay_s=ec.get("delay_s", 0.02),
                          tau_delta=ec.get("tau_delta", 0.03), tau_T=ec.get("tau_T", 0.05),
                          noise=ec.get("noise", 1.0), grip_map=grip)
        from .ac.ac_env import ACEnv
        cal = ec.get("calibration", {})
        if isinstance(cal, str):
            cal = yaml.safe_load(Path(self.root / cal).read_text())
        return ACEnv(self.params, cal, ec.get("vjoy_device", 1), ec.get("auto_shift"))

    def status(self):
        i = self.controller.info
        return dict(state=self.controller.state, s=round(float(i.get("s", 0)), 1),
                    v=round(float(i.get("vx", 0)), 1), beta_deg=round(float(np.degrees(i.get("beta", 0))), 1),
                    profile_version=self.profile.version, laps=self.lap_times,
                    warnings=self.profile.warnings)

    def request_stop(self):
        self._stop.set()

    # ------------------------------------------------------------------
    def run(self, laps: int = 5, live: bool = True, dt: float = 0.01, max_time: float = 600.0,
            on_tick=None):
        if live:
            from .live import CommandServer, ProfileWatcher
            lv = self.cfg.get("live", {})
            self.server = CommandServer(self, port=lv.get("port", 47800)).start()
            if self.profile_path and lv.get("watch_profile", True):
                ProfileWatcher(self.profile, str(self.profile_path)).start()
        env, ctl = self.env, self.controller
        if not env.realtime:
            s0 = self.cfg.get("env", {}).get("start_s", 0.0)
            ref = self.profile.sample(np.array([s0]))
            m = env.reset(s0=s0, n0=float(ref["n"][0]), v0=float(ref["v"][0]))
        else:
            m = env.read()
        log, s_prev, t_lap0, lap = [], None, m.t, 0
        t_wall = time.perf_counter()
        try:
            while not self._stop.is_set() and lap < laps:
                if env.realtime and time.perf_counter() - t_wall > max_time:
                    break
                tick0 = time.perf_counter()
                cmd, info = ctl.step(m)
                env.write(cmd)
                log.append(info)
                if on_tick:
                    on_tick(self, info)
                if env.realtime:
                    env.step(dt - (time.perf_counter() - tick0))
                else:
                    env.step(dt)
                m = env.read()
                s = info["s"]
                if s_prev is not None and s_prev > 0.8 * self.track.length and s < 0.2 * self.track.length:
                    lap_time = m.t - t_lap0
                    self._end_lap(lap, lap_time, log)
                    lap, log, t_lap0 = lap + 1, [], m.t
                s_prev = s
                if not env.realtime and m.t > max_time:
                    break
        finally:
            env.neutral()
        return self.lap_times

    def _end_lap(self, lap, lap_time, log):
        out = log
        incidents = sum(1 for l in out if l.get("state") != "MPC" or l.get("tyres_out", 0) > 2)
        self.lap_times.append(round(lap_time, 3))
        np.save(self.out_dir / f"lap{lap:03d}.npy", np.array(out, dtype=object), allow_pickle=True)
        print(f"[lap {lap}] {lap_time:.3f} s  (plan {self.raceline.lap_time:.3f}, "
              f"incident ticks {incidents}, profile v{self.profile.version})")
        if self.learning and (lap > 0 or not self.env.realtime):  # in AC lap 0 is an out-lap
            mu = self.learner.update(out, lap_time)
            print(f"[learn] mu mean {mu.mean():.3f} min {mu.min():.3f} max {mu.max():.3f}"
                  + ("  CONVERGED" if self.learner.converged() else ""))
            if not self.learner.converged():
                self._replan(blocking=not self.env.realtime)

    def _replan(self, blocking):
        def job():
            rl = plan_min_lap_time(self.track, self.params, mu_map=self.learner.mu_map(),
                                   warm=self.raceline, verbose=False, **self.plan_kw)
            if rl.meta["status"] in ("optimal", "Solved_To_Acceptable_Level"):
                self.raceline = rl
                rl.save(self.rl_path)
                self.profile.set_raceline(rl)
                print(f"[learn] re-planned: {rl.lap_time:.3f} s ({rl.meta['solve_s']} s solve)")
            else:
                print(f"[learn] re-plan failed ({rl.meta['status']}); keeping current line")
        if blocking:
            job()
        elif self._replan_thread is None or not self._replan_thread.is_alive():
            self._replan_thread = threading.Thread(target=job, daemon=True)
            self._replan_thread.start()
