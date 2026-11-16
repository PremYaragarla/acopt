"""Fast checks that need no race line."""
import math

import numpy as np
import pytest

from acopt.dashboard import find_corners
from acopt.profile import ProfileError, validate
from acopt.session import ConfigError, Session, load_config
from acopt.telemetry import TelemetryHub, lap_metrics, lap_trace, load_lap, write_lap_csv
from acopt.track import Track


def test_track_frenet_roundtrip():
    tr = Track.synthetic_club()
    rng = np.random.default_rng(0)
    for s, n in zip(rng.uniform(0, tr.length, 50), rng.uniform(-4, 4, 50)):
        X, Y = tr.to_world(s, n)
        s2, n2, _, _ = tr.to_frenet(float(X), float(Y))
        assert abs(tr.ds_signed(s2, s)) < 0.3 and abs(n2 - n) < 0.1


def test_corners_cover_both_directions():
    c = find_corners(Track.synthetic_club())
    assert len(c) >= 4 and {"left", "right"} == {x["dir"] for x in c}
    assert all(x["radius"] > 5 for x in c)


@pytest.mark.parametrize("bad, msg", [
    ({"segments": [{"name": "a", "mode": "drift"}]}, "s_start"),
    ({"segments": [{"name": "a", "mode": "slide", "s_start": 0, "s_end": 100}]}, "mode"),
    ({"segments": [{"name": "a", "s_start": 0, "s_end": 100, "beta_deg": "lots", "mode": "drift"}]}, "beta_deg"),
    ({"segments": [{"name": "a", "s_start": 0, "s_end": 100, "beta_deg": 95, "mode": "drift"}]}, "range"),
    ({"segments": [{"name": "a", "s_start": 0, "s_end": 100}, {"name": "a", "s_start": 200, "s_end": 300}]}, "duplicate"),
    ({"segments": [{"name": "a", "s_start": 100, "s_end": 101}]}, "shorter"),
    ({"speed_scale": 3.0}, "speed_scale"),
    ({"weights": {"drift": {"betta": 1}}}, "unknown weight"),
    ({"weights": {"race": {"n": -1}}}, "range"),
    ({"segments": "T1"}, "list"),
])
def test_profile_validation_rejects(bad, msg):
    with pytest.raises(ProfileError, match=msg):
        validate(bad, 1780.0)


def test_profile_validation_normalises():
    out = validate({"segments": [{"s_start": "100", "s_end": 140, "mode": "drift", "blend_m": 0}]}, 1780.0)
    sg = out["segments"][0]
    assert sg["name"] == "seg1" and sg["s_start"] == 100.0 and sg["blend_m"] >= 1.0 and sg["beta_deg"] == 15.0


def _fake_log(n=400, L=1000.0):
    out = []
    for i in range(n):
        t = 0.01 * i * 10
        out.append(dict(t=t, lap_t=t, s=L * i / n, X=1.0 * i, Y=0.0, psi=0.0, n=0.2 * math.sin(i / 20), n_ref=0.0,
                        vx=50.0 + 0.01 * i, vy=0.5, r=0.0, v_ref=50.0, beta=0.01, beta_ref=0.0,
                        mode=1.0 if 100 <= i < 200 else 0.0, delta_cmd=0.01, T_cmd=0.5, delta_act=0.01, T_act=0.5,
                        mu_hat=1.0, state="MPC" if i != 50 else "MPC_SAFE", why="", tyres_out=0,
                        **({"solve_ms": 20.0 + i % 5, "mpc_ok": True} if i % 5 == 0 else {})))
    return out


def test_lap_metrics_and_csv_roundtrip(tmp_path):
    log = _fake_log()
    m = lap_metrics(log, 40.0, 39.0, 1000.0)
    assert m["complete"] and m["ticks"] == 400 and m["interventions"] == 1 and m["incident_ticks"] == 1
    assert m["drift_share_pct"] == 25.0 and m["drift_beta_err_rms_deg"] == pytest.approx(0.57, abs=0.01)
    assert 20 <= m["solve_ms_median"] <= 25 and m["v_max_kmh"] > 180
    p = write_lap_csv(tmp_path / "lap.csv", log)
    back = load_lap(p)
    assert len(back) == len(log) and back[50]["state"] == "MPC_SAFE"
    assert back[10]["vx"] == pytest.approx(log[10]["vx"], rel=1e-5) and "solve_ms" not in back[1]
    assert lap_metrics(back, 40.0, 39.0, 1000.0)["n_err_rms_m"] == m["n_err_rms_m"]
    tr = lap_trace(back, 100)
    assert len(tr["s"]) == 100 and len(tr["v"]) == 100


def test_partial_lap_is_flagged():
    assert not lap_metrics(_fake_log()[:150], 15.0, 39.0, 1000.0)["complete"]


def test_hub_increments():
    hub = TelemetryHub(hist_hz=20)
    for i, l in enumerate(_fake_log(100)):
        l["t"] = 0.01 * i
        hub.push(l, 0, l["t"])
    snap = hub.snapshot()
    assert snap["seq"] == len(snap["ticks"]) == 20
    assert hub.snapshot(since=snap["seq"])["ticks"] == []
    hub.event("x", "hello", "warn")
    assert hub.snapshot(ev_since=0)["events"][-1]["msg"] == "hello"


def test_config_errors_are_readable(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")
    p = tmp_path / "bad.yaml"
    p.write_text("track: {source: synthetic}\n")
    with pytest.raises(ConfigError, match="vehicle"):
        load_config(p)
    p.write_text("vehicle: v.yaml\ntrack: {source: fast_lane_ai, path: C:/nowhere/fast_lane.ai}\n")
    (tmp_path / "v.yaml").write_text("name: x\n")
    with pytest.raises(ConfigError, match="track file not found"):
        Session(p)
