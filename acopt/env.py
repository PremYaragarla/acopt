"""
Environment interface shared by the simulator and Assetto Corsa.

Both produce Measurement in OUR planar frame (see vehicle.py conventions) and
accept Command(delta [rad, road wheel], T in [-1,1], optional gear request).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Measurement:
    t: float                 # seconds, monotonic
    X: float
    Y: float
    psi: float
    vx: float                # body frame
    vy: float
    r: float
    delta: float             # measured road-wheel angle (or last command)
    T: float                 # measured gas-brake (or last command)
    tyres_out: int = 0
    lap: int = 0
    fresh: bool = True
    extra: dict = field(default_factory=dict)


@dataclass
class Command:
    delta: float
    T: float
    gear_up: bool = False
    gear_down: bool = False


class Env:
    realtime: bool = False

    def reset(self, **kw) -> Measurement: ...
    def read(self) -> Measurement: ...
    def write(self, cmd: Command) -> None: ...
    def step(self, dt: float) -> None:
        """Advance time (sim only; AC runs on its own clock)."""
    def neutral(self) -> None:
        self.write(Command(0.0, 0.0))
    def close(self) -> None: ...
