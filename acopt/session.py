"""
Session: wires config -> track, car model, race line, profile, MPC, controller,
environment (sim, AC, or the AC mock), live tuning, telemetry and lap-to-lap
learning, and runs laps. `run()` can be called repeatedly (the dashboard does).
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
from .telemetry import TelemetryHub, lap_metrics, lap_trace, write_lap_csv, dumps
from .track import Track
from .vehicle import VehicleModel, VehicleParams


class ConfigError(ValueError):
    """The session config is unusable; the message says what to fix."""


def _dc(cls, d):
    names = {f.name for f in fields(cls)}
    unknown = set(d or {}) - names - {"supervisor", "grip", "enabled", "mu0"}
    if unknown:
        print(f"[session] WARNING: ignoring unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**{k: v for k, v in (d or {}).items() if k in names})


def load_config(cfg_path) -> tuple[dict, Path]:
    """Read a session YAML and resolve the project root. Raises ConfigError."""
    cfg_path = Path(cfg_path)
    if not cfg_path.is_file():
        raise ConfigError(f"config file not found: {cfg_path}")
    try:
        cfg = yaml.safe_load(cfg_path.read_text())
    except yaml.YAMLError as e:
        raise ConfigError(f"{cfg_path}: not valid YAML ({e})") from None
    if not isinstance(cfg, dict):
        raise ConfigError(f"{cfg_path}: expected a mapping at the top level")
    for key in ("vehicle", "track"):
        if key not in cfg:
            raise ConfigError(f"{cfg_path}: missing required section '{key}'")
    cfg_path = cfg_path.resolve()
    root = cfg_path.parent.parent if cfg_path.parent.name == "configs" else cfg_path.parent
    return cfg, root


def load_track(tc: dict, root: Path) -> Track:
    src = tc.get("source", "synthetic")
    if src == "synthetic":
        return Track.synthetic_club(tc.get("half_width", 6.0))
    if src == "synthetic_long":
        return Track.synthetic_long(tc.get("length_km", 20.8), tc.get("half_width", 4.5))
    if src not in ("csv", "fast_lane_ai"):
        raise ConfigError(f"track.source must be synthetic, synthetic_long, csv or fast_lane_ai (got {src!r})")
    if not tc.get("path"):
        raise ConfigError(f"track.path is required for source '{src}'")
    path = Path(tc["path"])
    path = path if path.is_absolute() else Path(root) / path
    if not path.is_file():
        hint = (" Edit track.path in the config to point at your install: "
                "<assettocorsa>/content/tracks/<track>/<layout>/ai/fast_lane.ai"
                if src == "fast_lane_ai" else "")
        raise ConfigError(f"track file not found: {path}.{hint}")
    try:
        if src == "csv":
            return Track.from_csv(path)
        return Track.from_fast_lane_ai(path, sign_z=tc.get("sign_z", -1.0),
                                       swap_sides=tc.get("swap_sides", False),
                                       default_half_width=tc.get("half_width", 6.0),
                                       subsample=tc.get("subsample", 4))
    except ConfigError:
        raise
    except Exception as e:  # noqa: BLE001
        raise ConfigError(f"could not read track {path}: {type(e).__name__}: {e}") from e


class Session:
    def __init__(self, cfg_path, overrides: dict | None = None, hub: TelemetryHub | None = None):
        self.hub = hub or TelemetryHub()
        self.cfg_path = Path(cfg_path)
        cfg, self.root = load_config(cfg_path)
        for k, v in (overrides or {}).items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
        self.cfg = cfg
        R = lambda p: Path(p) if Path(p).is_absolute() else self.root / p

        self.hub.set_phase("loading", "vehicle model and track")
        veh = R(cfg["vehicle"])
        if not veh.is_file():
            raise ConfigError(f"vehicle file not found: {veh}")
        self.params = VehicleParams.from_yaml(veh)
        self.model = VehicleModel(self.params)
        self.track = load_track(cfg["track"], self.root)
        print("[session]", self.track.summary())

        rc = cfg.get("raceline", {})
        self.rl_path = R(rc.get("cache", f"out/{self.track.name}_{self.params.name}.npz"))
        self.rl_path.parent.mkdir(parents=True, exist_ok=True)
        self.plan_kw = dict(ds=rc.get("ds", 4.0), margin=rc.get("margin", 0.4),
                            beta_max_deg=rc.get("beta_max_deg", 8.0),
                            window_m=rc.get("window_m"), overlap_m=rc.get("overlap_m", 600.0))
        lc = cfg.get("learning", {})
        self.learner = GripLearner(self.track, self.model, _dc(LearnCfg, lc), mu0=lc.get("mu0", 1.0))
        self.learning = bool(lc.get("enabled", False))
        self.raceline = None
        if self.rl_path.exists() and not rc.get("replan", False):
            try:
                rl = RaceLine.load(self.rl_path)
                if abs(rl.meta.get("track_length", self.track.length) - self.track.length) > 1.0 \
                        or float(rl.s[-1]) > self.track.length:
                    print(f"[session] cached race line {self.rl_path.name} is for a different track; re-planning")
                else:
                    self.raceline = rl
                    print(f"[session] loaded race line {self.rl_path.name}: {rl.lap_time:.3f} s")
            except Exception as e:  # noqa: BLE001 - truncated / old-format cache
                print(f"[session] could not read cached race line ({e}); re-planning")
        if self.raceline is None:
            n_st = self.track.length / self.plan_kw["ds"]
            eta = "about a minute" if n_st <= 1200 else f"roughly {max(2, round(n_st / 400))} minutes"
            msg = f"computing the minimum-lap-time line ({eta}, cached afterwards)"
            self.hub.set_phase("planning", msg)
            rl = plan_min_lap_time(
                self.track, self.params, mu_map=self.learner.mu_map(),
                verbose=bool(cfg.get("verbose_planner", False)),
                progress=lambda i, n: self.hub.set_phase("planning", f"{msg}: part {i + 1} of {n}"),
                **self.plan_kw)
            if not np.all(np.isfinite(rl.X)) or rl.meta["status"] not in OK_STATUS:
                raise ConfigError(f"race line planner failed ({rl.meta['status']}). Check the vehicle "
                                  "parameters and track widths; try a larger raceline.ds.")
            rl.meta["track_length"] = self.track.length
            rl.save(self.rl_path)
            self.raceline = rl
            print(f"[session] planned race line: {rl.lap_time:.3f} s ({rl.meta['solve_s']} s solve)")

        self.hub.set_phase("loading", "reference profile and MPC")
        prof_path = cfg.get("profile")
        self.profile_path = R(prof_path) if prof_path else None
        if self.profile_path and not self.profile_path.is_file():
            raise ConfigError(f"profile file not found: {self.profile_path}")
        pcfg = ReferenceProfile.load_yaml(self.profile_path) if self.profile_path else {}
        self.profile = ReferenceProfile(self.track, self.raceline, self.model, pcfg)
        for w in self.profile.warnings:
            self.hub.event("profile", w, "warn")

        mc = cfg.get("mpc", {})
        self.mpc = SpatialMPC(self.model, N=mc.get("N", 20), solver=mc.get("solver", "sqp"),
                              sqp_iter=mc.get("sqp_iter", 2), max_iter=mc.get("max_iter", 40))
        cc = cfg.get("controller", {})
        ccfg = _dc(ControllerCfg, cc)
        ccfg.sup = _dc(SupervisorCfg, cc.get("supervisor", {}))
        ccfg.grip = _dc(GripEstCfg, cc.get("grip", {}))
        self.controller = Controller(self.track, self.model, self.profile, self.mpc, ccfg)
        self.env_type = cfg.get("env", {}).get("type", "sim")
        self.env = self._make_env(cfg.get("env", {"type": "sim"}))
        self._stop = threading.Event()
        self.running = False
        self.pace = float(cfg.get("env", {}).get("pace", 0.0))   # sim: x real time, 0 = flat out
        self.lap_times: list[float] = []
        self.lap_no = 0
        self.log_root = R(cfg.get("log_dir", "out/logs"))
        self.out_dir = self.log_root / time.strftime("%Y%m%d_%H%M%S")
        self._replan_thread = None
        self.server = self.watcher = None
        self.hub.fit_lap(self.raceline.lap_time)
        self.hub.set_phase("ready")

    # ------------------------------------------------------------------
    def _make_env(self, ec):
        kind = ec.get("type", "sim")
        if kind not in ("sim", "ac", "ac_mock"):
            raise ConfigError(f"env.type must be sim, ac or ac_mock (got {kind!r})")
        cal = ec.get("calibration", {})
        if isinstance(cal, str):
            cp = Path(self.root / cal)
            if not cp.is_file():
                raise ConfigError(f"calibration file not found: {cp} (run scripts/ac_probe.py)")
            cal = yaml.safe_load(cp.read_text()) or {}
        if kind == "ac":
            from .ac.ac_env import ACEnv
            return ACEnv(self.params, cal, ec.get("vjoy_device", 1), ec.get("auto_shift"))
        from .sim import SimEnv
        truth = self.params
        pert = ec.get("truth_scale", {})       # e.g. {mu: 0.93, ClA: 0.9}
        if pert:
            bad = set(pert) - set(VehicleParams.DYN)
            if bad:
                raise ConfigError(f"env.truth_scale: unknown vehicle parameters {sorted(bad)}")
            d = {k: getattr(truth, k) * v for k, v in pert.items()}
            truth = truth.with_vec([d.get(k, getattr(truth, k)) for k in VehicleParams.DYN])
        grip = None
        if ec.get("grip_patch"):
            a, b, f = ec["grip_patch"]          # [s_start, s_end, factor]
            L = self.track.length
            grip = lambda s: f if (s - a) % L <= (b - a) % L else 1.0
        sim = SimEnv(self.track, truth, delay_s=ec.get("delay_s", 0.02),
                     tau_delta=ec.get("tau_delta", 0.03), tau_T=ec.get("tau_T", 0.05),
                     noise=ec.get("noise", 1.0), grip_map=grip)
        if kind == "sim":
            return sim
        # ac_mock: the real ACEnv code path, fed by the simulator through AC's conventions
        from .ac.ac_env import ACEnv
        from .ac.mock import MockAC
        mock = MockAC(sim, cal, self.params)
        env = ACEnv(self.params, cal, auto_shift=ec.get("auto_shift"), sm=mock, out=mock,
                    clock=mock.clock, sleep=mock.advance, realtime=False)
        env.mock = mock
        env.kick = sim.kick

        def reset(**kw):
            sim.reset(**kw)
            mock._refresh_graphics(force=True)
            env._gpos = None
            return env.read()
        env.reset = reset
        return env

    def status(self):
        i = self.controller.info
        return dict(state=self.controller.state, s=round(float(i.get("s", 0)), 1),
                    v=round(float(i.get("vx", 0)), 1), beta_deg=round(float(np.degrees(i.get("beta", 0))), 1),
                    profile_version=self.profile.version, laps=self.lap_times,
                    warnings=self.profile.warnings, running=self.running, phase=self.hub.phase,
                    paused=self.controller.paused, mu_hat=round(self.controller.mu_hat, 3),
                    log_dir=str(self.out_dir))

    def request_stop(self):
        self._stop.set()

    # ------------------------------------------------------------------
    def _start_live(self):
        """UDP command server + profile file watcher. Started once; a busy port is a
        warning, not a crash (the dashboard can still control everything)."""
        lv = self.cfg.get("live", {})
        if self.server is None and lv.get("udp", True):
            from .live import CommandServer
            try:
                self.server = CommandServer(self, port=lv.get("port", 47800)).start()
            except OSError as e:
                self.server = False
                self.hub.event("live", f"UDP port {lv.get('port', 47800)} unavailable ({e}); "
                                       "scripts/tune.py is disabled for this session", "warn")
        if self.watcher is None and self.profile_path and lv.get("watch_profile", True):
            from .live import ProfileWatcher
            self.watcher = ProfileWatcher(
                self.profile, str(self.profile_path),
                on_event=lambda msg, level="info": self.hub.event("live", msg, level))
            self.watcher.start()

    def close(self):
        self.request_stop()
        if self.server:
            self.server.stop()
        if self.watcher:
            self.watcher.stop()
        try:
            self.env.close()
        except Exception:  # noqa: BLE001
            pass

    def run(self, laps: int = 5, live: bool = True, dt: float = 0.01, max_time: float | None = None,
            on_tick=None, pace: float | None = None):
        """Drive `laps` laps (or until stopped). Returns the lap times of the session."""
        if self.running:
            raise RuntimeError("session is already running")
        self.running = True
        self._stop.clear()
        if pace is not None:
            self.pace = pace
        env, ctl, hub = self.env, self.controller, self.hub
        log, lap_start_s = [], 0.0
        try:
            if live:
                self._start_live()
            ctl.reset()
            if not env.realtime:
                s0 = float(self.cfg.get("env", {}).get("start_s", 0.0)) % self.track.length
                ref = self.profile.sample(np.array([s0]))
                v0 = self.cfg.get("env", {}).get("start_v")
                m = env.reset(s0=s0, n0=float(ref["n"][0]), v0=float(ref["v"][0]) if v0 is None else float(v0))
                if max_time is None:
                    max_time = 60.0 + 4.0 * laps * self.raceline.lap_time
            else:
                hub.set_phase("waiting", "waiting for Assetto Corsa: load a session and get the car on track")
                m = env.wait_live(stop=self._stop)
                if m is None:
                    return self.lap_times
                d = env.describe()
                if d:
                    hub.event("ac", f"connected: car '{d.get('car')}', track '{d.get('track')}'")
            hub.set_phase("running", f"{laps} lap{'s' if laps != 1 else ''}")
            s_prev, t_lap0, t_run0, done = None, m.t, m.t, 0
            state_prev = ctl.state
            t_wall = time.perf_counter()
            min_lap = 0.3 * self.raceline.lap_time
            while not self._stop.is_set() and done < laps:
                if max_time is not None and (
                        (env.realtime and time.perf_counter() - t_wall > max_time)
                        or (not env.realtime and m.t - t_run0 > max_time)):
                    hub.event("session", f"time limit of {max_time:.0f} s reached", "warn")
                    break
                tick0 = time.perf_counter()
                cmd, info = ctl.step(m)
                env.write(cmd)
                info["lap_t"] = m.t - t_lap0
                log.append(info)
                hub.push(info, self.lap_no, info["lap_t"])
                if info["state"] != state_prev:
                    back = info["state"] == "MPC"
                    hub.event("supervisor", f"{state_prev} -> {info['state']}"
                              + (f" ({info.get('why')})" if info.get("why") else "")
                              + f" at s={info['s']:.0f} m, {3.6 * info['vx']:.0f} km/h",
                              "info" if back else "warn")
                    state_prev = info["state"]
                if on_tick:
                    on_tick(self, info)
                if env.realtime:
                    env.step(dt - (time.perf_counter() - tick0))
                else:
                    env.step(dt)
                    if self.pace > 0:           # watchable simulator: hold sim time to wall time
                        lag = dt / self.pace - (time.perf_counter() - tick0)
                        if lag > 0:
                            time.sleep(lag)
                m = env.read()
                s = info["s"]
                if s_prev is None:
                    lap_start_s = s
                elif s_prev > 0.8 * self.track.length and s < 0.2 * self.track.length \
                        and m.t - t_lap0 > min_lap:
                    lap_time = m.t - t_lap0
                    self._end_lap(lap_time, log)
                    done, log, t_lap0 = done + 1, [], m.t
                    if not self._stop.is_set() and done < laps:
                        hub.set_phase("running", f"lap {done + 1} of {laps}")
                s_prev = s
        except Exception as e:
            hub.event("session", f"run aborted: {type(e).__name__}: {e}", "error")
            raise
        finally:
            try:
                env.neutral()
            except Exception as e:  # noqa: BLE001
                hub.event("session", f"could not centre the controls: {e}", "error")
            if len(log) > 50:
                self._save_log("partial", log)      # the unfinished lap, for diagnosis
            self.running = False
            hub.set_phase("ready", "stopped" if self._stop.is_set() else "finished")
        return self.lap_times

    # ------------------------------------------------------------------
    def _save_log(self, stem, log):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.out_dir / f"{stem}.npy", np.array(log, dtype=object), allow_pickle=True)
        write_lap_csv(self.out_dir / f"{stem}.csv", log)
        (self.log_root / "latest.txt").write_text(str(self.out_dir))

    def write_summary(self) -> Path:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        summ = self.hub.session_summary()
        summ.update(config=str(self.cfg_path), car=self.params.name, track=self.track.name,
                    track_length_m=round(self.track.length, 1), env=self.env_type,
                    profile=self.profile.as_dict(), created=time.strftime("%Y-%m-%d %H:%M:%S"))
        p = self.out_dir / "session.json"
        p.write_text(json.dumps(json.loads(dumps(summ)), indent=1))
        return p

    def _end_lap(self, lap_time, log):
        lap = self.lap_no
        self.lap_no += 1
        self.lap_times.append(round(lap_time, 3))
        met = lap_metrics(log, lap_time, self.raceline.lap_time, self.track.length)
        met.update(lap=lap, profile_version=self.profile.version, csv=f"lap{lap:03d}.csv",
                   mu_map_mean=round(float(self.learner.mu.mean()), 3))
        self.hub.end_lap(met, lap_trace(log))
        try:
            self._save_log(f"lap{lap:03d}", log)
            self.write_summary()
        except OSError as e:                     # disk full / folder locked: keep driving
            self.hub.event("session", f"could not write lap log: {e}", "error")
        self.hub.event("lap", f"lap {lap}: {lap_time:.3f} s (plan {self.raceline.lap_time:.3f}, "
                              f"incident ticks {met['incident_ticks']}"
                              + ("" if met["complete"] else ", partial lap") + ")")
        # an out-lap (AC) or a lap with a stop in it teaches nothing about grip at speed
        if self.learning and met["complete"]:
            mu = self.learner.update(log, lap_time)
            self.hub.event("learn", f"grip map: mean {mu.mean():.3f} min {mu.min():.3f} max {mu.max():.3f}"
                           + ("  (converged)" if self.learner.converged() else ""))
            if not self.learner.converged():
                self._replan(blocking=not self.env.realtime and self.pace <= 0)

    def _replan(self, blocking):
        def job():
            try:
                rl = plan_min_lap_time(self.track, self.params, mu_map=self.learner.mu_map(),
                                       warm=self.raceline, verbose=False, **self.plan_kw)
                if rl.meta["status"] in OK_STATUS and np.all(np.isfinite(rl.X)):
                    rl.meta["track_length"] = self.track.length
                    self.raceline = rl
                    rl.save(self.rl_path)
                    self.profile.set_raceline(rl)
                    self.hub.event("learn", f"re-planned line: {rl.lap_time:.3f} s predicted "
                                            f"({rl.meta['solve_s']} s solve)")
                else:
                    self.hub.event("learn", f"re-plan failed ({rl.meta['status']}); keeping current line", "warn")
            except Exception as e:  # noqa: BLE001 - a failed re-plan must never stop the car
                self.hub.event("learn", f"re-plan crashed ({type(e).__name__}: {e}); keeping current line", "warn")
        if blocking:
            was = (self.hub.phase, self.hub.phase_msg)
            self.hub.set_phase("planning", "re-planning the line with the learned grip map")
            job()
            self.hub.set_phase(*was)
        elif self._replan_thread is None or not self._replan_thread.is_alive():
            self._replan_thread = threading.Thread(target=job, daemon=True)
            self._replan_thread.start()


OK_STATUS = ("optimal", "Solve_Succeeded", "Solved_To_Acceptable_Level")
