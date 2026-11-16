"""Shared fixtures. The first run plans a race line for the synthetic circuit
(~1 minute) and caches it in out/; later runs start in a few seconds."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from acopt.session import Session  # noqa: E402

CFG = ROOT / "configs" / "sim_test.yaml"


def make_session(tmp, **over):
    ov = {"log_dir": str(tmp), "learning": {"enabled": False}, "live": {"udp": False, "watch_profile": False},
          "raceline": {"cache": "out/test_synthetic_f2004.npz"}}
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(ov.get(k), dict):
            ov[k].update(v)
        else:
            ov[k] = v
    return Session(CFG, ov)


@pytest.fixture(scope="session")
def session(tmp_path_factory):
    s = make_session(tmp_path_factory.mktemp("logs"))
    yield s
    s.close()


@pytest.fixture(scope="session")
def mock_session(tmp_path_factory):
    """Same car and track, but driven through ACEnv + the AC mock with a deliberately
    awkward calibration (mirrored heading with an offset, inverted steering axis)."""
    cal = dict(sign_z=-1.0, heading_sign=-1.0, heading_offset=1.1, yaw_rate_sign=-1.0,
               steer_sign=-1.0, brake_gamma=2.0, stale_s=0.1)
    s = make_session(tmp_path_factory.mktemp("logs_mock"), env={"type": "ac_mock", "calibration": cal})
    yield s
    s.close()
