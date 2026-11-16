"""End-to-end checks in the simulator: the car drives, survives abuse, and the data
that comes out is usable."""
import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

from acopt.dashboard import App, serve
from acopt.env import Measurement
from acopt.live import execute
from acopt.profile import ProfileError
from acopt.telemetry import load_lap


def _drive(session, seconds, hook=None, **kw):
    """Run for a fixed simulated time and return the tick log."""
    log = []

    def on_tick(s, info):
        log.append(info)
        if hook:
            hook(s, info)
        if info["lap_t"] >= seconds and not kw.get("laps_only"):
            s.request_stop()
    session.controller.paused = False
    session.run(laps=kw.get("laps", 99), live=False, on_tick=on_tick)
    return log


def test_full_lap_matches_plan_and_writes_data(session):
    n0 = len(session.lap_times)
    session.run(laps=1, live=False)
    assert len(session.lap_times) == n0 + 1
    lap, plan = session.lap_times[-1], session.raceline.lap_time
    assert plan < lap < 1.04 * plan, (lap, plan)          # the sim car has less grip than the model
    met = session.hub.laps[-1]
    assert met["complete"] and met["interventions"] == 0 and met["n_err_rms_m"] < 0.5
    assert met["solve_ms_median"] > 0 and met["state_share_pct"] == {"MPC": 100.0}
    rows = load_lap(session.out_dir / met["csv"])
    assert len(rows) == met["ticks"] and abs(rows[-1]["lap_t"] - lap) < 0.05
    summ = json.loads((session.out_dir / "session.json").read_text())
    assert summ["best_lap"] == min(l["lap_time"] for l in summ["laps"]) and summ["track"] == "synthetic_club"
    assert session.hub.trace(met["lap"])["s"]


def test_recovers_from_a_spin(session):
    kicked = []

    def hook(s, info):
        if info["lap_t"] > 4.0 and not kicked:
            kicked.append(info["s"])
            s.env.kick(yaw_rate=3.5)
    log = _drive(session, 30.0, hook)
    states = [l["state"] for l in log]
    assert "SAFE_STOP" in states or "MPC_SAFE" in states      # the supervisor noticed
    assert states[-1] == "MPC", states[-1]                    # ...and handed back to normal control
    assert log[-1]["vx"] > 20 and abs(log[-1]["n"] - log[-1]["n_ref"]) < 1.0
    assert all(np.isfinite([l["delta_cmd"], l["T_cmd"]]).all() for l in log)
    assert any("->" in e["msg"] for e in session.hub.snapshot()["events"] if e["kind"] == "supervisor")


def test_standing_start(session):
    session.cfg["env"]["start_v"] = 0.0
    try:
        log = _drive(session, 12.0)
    finally:
        session.cfg["env"].pop("start_v")
    assert log[-1]["vx"] > 30 and log[-1]["state"] == "MPC"
    assert max(abs(l["n"] - l["n_ref"]) for l in log) < 2.5


def test_bad_telemetry_never_reaches_the_solver(session):
    ctl = session.controller
    ctl.reset()
    good = session.env.reset(s0=0.0, n0=0.0, v0=40.0)
    cmd, info = ctl.step(good)
    bad = Measurement(t=good.t + 0.01, X=float("nan"), Y=good.Y, psi=good.psi, vx=good.vx, vy=0.0, r=0.0,
                      delta=0.0, T=0.0)
    cmd, info = ctl.step(bad)
    assert info["state"] == "SAFE_STOP" and cmd.delta == 0.0 and cmd.T < 0
    stale = Measurement(t=good.t + 0.02, X=good.X, Y=good.Y, psi=good.psi, vx=good.vx, vy=0.0, r=0.0,
                        delta=0.0, T=0.0, fresh=False)
    cmd, info = ctl.step(stale)
    assert info["state"] == "SAFE_STOP" and info["why"] == "stale"


def test_solver_crash_degrades_to_fallback(session, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("solver exploded")
    monkeypatch.setattr(session.mpc, "solve", boom)
    log = _drive(session, 3.0)
    assert log[-1]["state"] == "FALLBACK"
    assert all(np.isfinite([l["delta_cmd"], l["T_cmd"]]).all() for l in log)
    assert abs(log[-1]["n"] - log[-1]["n_ref"]) < 3.0          # still on the road


def test_live_drift_segment_is_tracked(session):
    try:
        r = execute(session, {"cmd": "segment", "name": "sweep", "mode": "drift", "s_start": 1420,
                              "s_end": 1720, "beta_deg": 15, "blend_m": 40})
        assert r["ok"], r
        session.cfg["env"]["start_s"] = 1350.0
        log = _drive(session, 9.0)
    finally:
        session.cfg["env"].pop("start_s", None)
        execute(session, {"cmd": "remove", "name": "sweep"})
    core = [l for l in log if l["mode"] > 0.95]
    assert len(core) > 100
    err = np.degrees([l["beta"] - l["beta_ref"] for l in core])
    ref = np.degrees([abs(l["beta_ref"]) for l in core])
    assert ref.max() > 10 and np.sqrt(np.mean(err ** 2)) < 2.5, (ref.max(), np.sqrt(np.mean(err ** 2)))
    assert all(l["state"] == "MPC" for l in core)


def test_bad_live_commands_are_refused_without_side_effects(session):
    v = session.profile.version
    for cmd in ({"cmd": "segment", "name": "x", "mode": "drift"},
                {"cmd": "segment", "name": "x", "s_start": 0, "s_end": 50, "beta_deg": 400, "mode": "drift"},
                {"cmd": "weights", "mode": "drift", "nonsense": 1},
                {"cmd": "speed_scale", "value": "fast"}, {"cmd": "remove", "name": "ghost"},
                {"cmd": "load_profile", "path": "does/not/exist.yaml"}, {"cmd": "warp"}, "junk"):
        r = execute(session, cmd)
        assert r["ok"] is False and r["error"], cmd
    assert session.profile.version == v
    with pytest.raises(ProfileError):
        session.profile.update({"speed_scale": 9})


def test_ac_interface_roundtrip(mock_session):
    """ACEnv (the code that talks to the game) driven through the mock with an awkward
    calibration: frame conversion, steering/brake mapping and position extrapolation."""
    s = mock_session
    log = _drive(s, 10.0)
    assert all(l["state"] == "MPC" for l in log)
    assert max(abs(l["n"] - l["n_ref"]) for l in log) < 1.0 and log[-1]["vx"] > 20
    truth = s.env.mock.sim.state
    assert abs(log[-1]["psi"] - truth[2]) < 0.05 or abs(abs(log[-1]["psi"] - truth[2]) - 2 * np.pi) < 0.05


def test_ac_freeze_triggers_safe_stop(mock_session):
    s = mock_session

    def hook(sess, info):
        if info["lap_t"] > 2.0:
            sess.env.mock.frozen = True          # game hangs: physics page stops updating
    try:
        log = _drive(s, 4.0, hook)
    finally:
        s.env.mock.frozen = False
    stopped = [l for l in log if l["state"] == "SAFE_STOP"]
    assert log[-1]["state"] == "SAFE_STOP" and stopped[0]["why"] == "stale"
    assert stopped[0]["T_cmd"] < 0 and all(l["delta_cmd"] == 0.0 for l in stopped)
    assert log[-1]["vx"] < 0.5 * stopped[0]["vx"]                  # it actually slowed the car
    assert s.env.mock._gas == 0.0 and s.env.mock._brake == 0.0     # controls released after the run


def test_dashboard_api(session):
    app = App.__new__(App)
    app.__init__(session.cfg_path)
    app.session, app.hub = session, session.hub
    srv, url = serve(app, port=8790)

    def get(path):
        with urllib.request.urlopen(url + path, timeout=10) as r:
            return r.read()

    def post(obj, headers=None):
        req = urllib.request.Request(url + "/api/command", data=json.dumps(obj).encode(),
                                     headers={"Content-Type": "application/json", **(headers or {})})
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())
    try:
        assert b"<title>acopt dashboard</title>" in get("/")
        st = json.loads(get("/api/static"))
        assert st["ready"] and len(st["left"][0]) == len(st["s"]) and st["corners"]
        ref = json.loads(get("/api/reference"))
        assert len(ref["X"]) == len(ref["s"]) == len(ref["v"])
        state = json.loads(get("/api/state"))
        assert state["ready"] and state["profile"]["version"] == session.profile.version and state["laps"]
        n = state["laps"][-1]["lap"]
        assert json.loads(get(f"/api/lap?n={n}"))["s"]
        assert get(f"/api/lap.csv?n={n}").startswith(b"t,lap_t,s,")
        assert json.loads(get("/api/session.json"))["n_laps"] >= 1
        assert post({"cmd": "speed_scale", "value": 0.9})[1]["ok"]
        assert session.profile.as_dict()["speed_scale"] == 0.9
        assert post({"cmd": "speed_scale", "value": 0.97})[1]["ok"]
        code, r = post({"cmd": "segment", "name": "q"})
        assert code == 200 and not r["ok"] and "s_start" in r["error"]
        assert post({"cmd": "stop"}, {"Origin": "http://evil.example"})[0] == 403
        # start / stop a run through the API
        assert post({"cmd": "start", "laps": 50, "pace": 0})[1]["ok"]
        assert not post({"cmd": "start", "laps": 1})[1]["ok"]            # already running
        done = threading.Event()
        for _ in range(100):
            if json.loads(get("/api/state"))["seq"] > state["seq"] + 20:
                break
            done.wait(0.1)
        live = json.loads(get(f"/api/state?since={state['seq']}"))
        assert live["running"] and live["ticks"] and live["horizon"]
        assert post({"cmd": "stop"})[1]["ok"]
        app._thread.join(timeout=20)
        assert not session.running
    finally:
        session.request_stop()
        srv.shutdown()


def test_impossible_start_ends_in_recovery_not_a_loop(session):
    """Dropped mid-corner at corner speed with no yaw rate: the car cannot make the
    corner. It must stop safely, get back on the road and resume normal control."""
    session.cfg["env"]["start_s"] = 1250.0
    try:
        log = _drive(session, 14.0)
    finally:
        session.cfg["env"].pop("start_s", None)
    assert log[-1]["state"] == "MPC" and log[-1]["vx"] > 15
    assert abs(log[-1]["n"] - log[-1]["n_ref"]) < 1.5
