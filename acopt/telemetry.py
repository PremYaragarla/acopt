"""
Telemetry: everything that leaves the control loop as data.

* TelemetryHub   thread-safe buffer between the control loop and observers
                 (dashboard, tests): decimated tick history, lap summaries,
                 events (supervisor transitions, warnings, commands), phase.
* lap_metrics    one row of numbers per lap (time, tracking errors, solver
                 timing, supervisor shares) - what the lap table and the
                 report are built from.
* write_lap_csv / load_lap   full-rate lap logs as plain CSV (one row per
                 control tick), readable by pandas / Excel / MoTeC importers.

The hub never blocks the control loop for longer than a deque append.
"""
from __future__ import annotations

import csv
import json
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

# columns of the per-tick CSV (SI units, radians) in a stable order
CSV_FIELDS = ("t", "lap_t", "s", "X", "Y", "psi", "n", "n_ref", "vx", "vy", "r", "v_ref",
              "beta", "beta_ref", "mode", "delta_cmd", "T_cmd", "delta_act", "T_act",
              "mu_hat", "state", "why", "tyres_out", "solve_ms", "mpc_ok", "slack", "defect")

# columns streamed to the dashboard (display units: km/h, degrees)
TICK_FIELDS = ("seq", "lap", "lap_t", "s", "X", "Y", "psi", "n_err", "v", "v_ref",
               "beta", "beta_ref", "steer", "T", "mode", "mu_hat", "solve_ms", "state")

STATES = ("MPC", "MPC_SAFE", "FALLBACK", "SAFE_STOP", "RECOVER", "PAUSED")


def _f(x, default=float("nan")):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _col(log, key, default=np.nan):
    return np.array([_f(l.get(key, default), default) for l in log], dtype=float)


def lap_metrics(log: list[dict], lap_time: float, plan_time: float | None = None,
                track_length: float | None = None) -> dict:
    """Summary numbers for one lap. All angles in degrees, speeds in km/h."""
    if not log:
        return dict(lap_time=round(lap_time, 3), ticks=0, complete=False)
    s, vx = _col(log, "s"), _col(log, "vx")
    n_err = _col(log, "n") - _col(log, "n_ref")
    beta, beta_ref = np.degrees(_col(log, "beta")), np.degrees(_col(log, "beta_ref"))
    mode = _col(log, "mode", 0.0)
    drift = mode > 0.5
    states = [l.get("state", "MPC") for l in log]
    solve = _col(log, "solve_ms")
    solve = solve[np.isfinite(solve)]
    ok = [l["mpc_ok"] for l in log if "mpc_ok" in l]
    tyres = _col(log, "tyres_out", 0.0)
    rms = lambda a: float(np.sqrt(np.mean(np.square(a)))) if len(a) else 0.0
    share = {st: round(100.0 * states.count(st) / len(states), 2) for st in STATES if st in states}
    transitions = sum(1 for a, b in zip(states, states[1:]) if a != b and b != "MPC")
    covered = 1.0
    if track_length:
        # fraction of the lap actually driven (an out-lap in AC starts mid-track)
        covered = float(np.sum(np.abs((np.diff(s) + track_length / 2) % track_length
                                      - track_length / 2)) / track_length)
    m = dict(
        lap_time=round(float(lap_time), 3),
        plan_time=None if plan_time is None else round(float(plan_time), 3),
        ticks=len(log),
        complete=bool(covered > 0.97),
        v_max_kmh=round(3.6 * float(np.nanmax(vx)), 1),
        v_min_kmh=round(3.6 * float(np.nanmin(vx)), 1),
        v_avg_kmh=round(3.6 * float(np.nanmean(vx)), 1),
        n_err_rms_m=round(rms(n_err[~drift]) if (~drift).any() else rms(n_err), 3),
        n_err_max_m=round(float(np.nanmax(np.abs(n_err))), 3),
        v_err_rms_kmh=round(3.6 * rms(vx - _col(log, "v_ref")), 2),
        beta_max_deg=round(float(np.nanmax(np.abs(beta))), 2),
        drift_share_pct=round(100.0 * float(drift.mean()), 1),
        drift_beta_err_rms_deg=round(rms((beta - beta_ref)[drift]), 2) if drift.any() else None,
        drift_beta_err_max_deg=round(float(np.max(np.abs((beta - beta_ref)[drift]))), 2) if drift.any() else None,
        mu_hat_mean=round(float(np.nanmean(_col(log, "mu_hat", 1.0))), 3),
        solve_ms_median=round(float(np.median(solve)), 1) if len(solve) else None,
        solve_ms_p95=round(float(np.percentile(solve, 95)), 1) if len(solve) else None,
        solver_fail_pct=round(100.0 * (1 - float(np.mean(ok))), 2) if ok else 0.0,
        incident_ticks=int(sum(1 for st, to in zip(states, tyres) if st != "MPC" or to > 2)),
        interventions=int(transitions),
        tyres_out_max=int(np.nanmax(tyres)) if len(tyres) else 0,
        state_share_pct=share,
    )
    return m


def write_lap_csv(path, log: list[dict]) -> Path:
    path = Path(path)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(CSV_FIELDS)
        for l in log:
            row = []
            for k in CSV_FIELDS:
                v = l.get(k, "")
                if isinstance(v, (bool, np.bool_)):
                    v = int(v)
                elif isinstance(v, (float, np.floating)):
                    v = f"{float(v):.6g}"
                row.append(v)
            w.writerow(row)
    return path


def load_lap(path) -> list[dict]:
    """Read a lap log (.csv written by write_lap_csv, or a legacy .npy) as a list of dicts."""
    path = Path(path)
    if path.suffix == ".npy":
        return list(np.load(path, allow_pickle=True))
    out = []
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            d = {}
            for k, v in row.items():
                if v == "":
                    continue
                if k in ("state", "why"):
                    d[k] = v
                else:
                    d[k] = float(v)
            if "mpc_ok" in d:
                d["mpc_ok"] = bool(d["mpc_ok"])
            out.append(d)
    return out


def lap_trace(log: list[dict], n_points: int | None = None) -> dict:
    """Lap resampled on an even grid of lap distance, for plotting (display units).
    Default resolution: about one point per 3 m, between 600 and 2500 points."""
    if len(log) < 2:
        return dict(s=[])
    s = _col(log, "s")
    if n_points is None:
        n_points = int(np.clip((np.nanmax(s) - np.nanmin(s)) / 3.0, 600, 2500))
    # unwrap distance so a lap that starts mid-track is still monotonic
    L = max(float(np.nanmax(s)), 1.0)
    order = np.argsort(s, kind="stable")
    ss = s[order]
    grid = np.linspace(float(ss[0]), float(ss[-1]), min(n_points, len(ss)))
    def g(a):
        return [round(float(v), 3) for v in np.interp(grid, ss, np.asarray(a, float)[order])]
    return dict(
        s=[round(float(v), 1) for v in grid],
        v=g(3.6 * _col(log, "vx")), v_ref=g(3.6 * _col(log, "v_ref")),
        beta=g(np.degrees(_col(log, "beta"))), beta_ref=g(np.degrees(_col(log, "beta_ref"))),
        n_err=g(_col(log, "n") - _col(log, "n_ref")),
        steer=g(np.degrees(_col(log, "delta_cmd"))), T=g(100 * _col(log, "T_cmd")),
        mode=g(_col(log, "mode", 0.0)), mu_hat=g(_col(log, "mu_hat", 1.0)),
        lap_t=g(_col(log, "lap_t", 0.0)),
        length=L,
    )


class TelemetryHub:
    MAXLEN = 6000

    def __init__(self, hist_hz: float = 20.0, maxlen: int = MAXLEN):
        self._lock = threading.Lock()
        self.hist_dt = 1.0 / hist_hz
        self.ticks: deque = deque(maxlen=maxlen)      # rows in TICK_FIELDS order
        self.events: deque = deque(maxlen=400)
        self.laps: list[dict] = []
        self.traces: dict[int, dict] = {}
        self.latest: dict = {}
        self.seq = 0
        self.ev_seq = 0
        self._t_hist = -1e9
        self.phase, self.phase_msg = "starting", ""
        self.t0 = time.time()

    def fit_lap(self, lap_time_s: float):
        """Pick the history rate so that more than one whole lap always fits in the buffer
        (20 Hz for short laps, ~10 Hz for an 8-minute lap)."""
        self.hist_dt = max(0.05, 1.25 * float(lap_time_s) / (self.ticks.maxlen or self.MAXLEN))

    # ---- writers (control loop / session) ----
    def set_phase(self, phase: str, msg: str = ""):
        with self._lock:
            changed = (phase, msg) != (self.phase, self.phase_msg)
            self.phase, self.phase_msg = phase, msg
        if changed:
            self.event("phase", f"{phase}" + (f": {msg}" if msg else ""))

    def event(self, kind: str, msg: str, level: str = "info"):
        with self._lock:
            self.ev_seq += 1
            self.events.append(dict(seq=self.ev_seq, t=round(time.time() - self.t0, 2),
                                    kind=kind, level=level, msg=str(msg)))
        print(f"[{kind}] {msg}")

    def push(self, info: dict, lap: int, lap_t: float):
        """Called every control tick. Keeps the latest tick and a decimated history."""
        t = _f(info.get("t"), 0.0)
        row = None
        if abs(t - self._t_hist) >= self.hist_dt - 1e-6:
            self._t_hist = t
            r = lambda k, nd=3, d=0.0: round(_f(info.get(k), d), nd)
            row = [0, lap, round(lap_t, 3), r("s", 1), r("X", 2), r("Y", 2), r("psi"),
                   round(_f(info.get("n"), 0) - _f(info.get("n_ref"), 0), 3),
                   round(3.6 * _f(info.get("vx"), 0), 2), round(3.6 * _f(info.get("v_ref"), 0), 2),
                   round(np.degrees(_f(info.get("beta"), 0)), 2),
                   round(np.degrees(_f(info.get("beta_ref"), 0)), 2),
                   round(np.degrees(_f(info.get("delta_cmd"), 0)), 2), r("T_cmd"),
                   r("mode", 2), r("mu_hat", 3, 1.0), r("solve_ms", 1, -1.0),
                   info.get("state", "?")]
        with self._lock:
            self.latest = info
            self._lap, self._lap_t = lap, lap_t
            if row is not None:
                self.seq += 1
                row[0] = self.seq
                self.ticks.append(row)

    def end_lap(self, metrics: dict, trace: dict):
        with self._lock:
            self.laps.append(metrics)
            self.traces[metrics["lap"]] = trace
            for k in sorted(self.traces)[:-40]:       # bound memory on long sessions
                del self.traces[k]

    # ---- readers ----
    def snapshot(self, since: int = 0, ev_since: int = 0, max_ticks: int = MAXLEN) -> dict:
        with self._lock:
            ticks = [r for r in self.ticks if r[0] > since][-max_ticks:]
            events = [e for e in self.events if e["seq"] > ev_since]
            return dict(seq=self.seq, ev_seq=self.ev_seq, phase=self.phase, phase_msg=self.phase_msg,
                        ticks=ticks, events=events, laps=list(self.laps))

    def trace(self, lap: int):
        with self._lock:
            return self.traces.get(lap)

    def session_summary(self) -> dict:
        with self._lock:
            laps = list(self.laps)
        done = [l for l in laps if l.get("complete")]
        times = [l["lap_time"] for l in done]
        return dict(
            laps=laps,
            n_laps=len(laps),
            best_lap=min(times) if times else None,
            best_lap_no=done[int(np.argmin(times))]["lap"] if times else None,
            improvement_s=round(times[0] - min(times), 3) if len(times) > 1 else 0.0,
            total_incident_ticks=int(sum(l.get("incident_ticks", 0) for l in laps)),
        )


def to_jsonable(o):
    """json.dumps default= hook for numpy types."""
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o).__name__)


def dumps(obj) -> str:
    def clean(v):
        if isinstance(v, float):
            return v if np.isfinite(v) else None
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [clean(x) for x in v]
        if isinstance(v, np.generic):
            return clean(v.item())
        if isinstance(v, np.ndarray):
            return clean(v.tolist())
        return v
    return json.dumps(clean(obj), default=to_jsonable)
