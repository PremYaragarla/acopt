"""
Track geometry in Frenet form: centreline, heading, curvature, widths, and
fast projection world <-> (s, n, xi).

Loaders:
  Track.from_fast_lane_ai(path)  - AC's  content/tracks/<trk>/<layout>/ai/fast_lane.ai
  Track.from_csv(path)           - x,y,w_left,w_right (or TUMFTM x_m,y_m,w_tr_right_m,w_tr_left_m)
  Track.synthetic_club()         - built-in test circuit (NOT a model of any real track)

World mapping for AC: AC is Y-up; the ground plane is (x, z). We map to our
right-handed planar frame as X = x, Y = sign_z * z. Verify sign_z with
scripts/ac_probe.py (it reports whether curvature and yaw rate agree).
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class Track:
    def __init__(self, xy, w_left, w_right, name="track", ds=1.0, kappa_smooth_m=6.0):
        xy = np.asarray(xy, float)
        if np.linalg.norm(xy[0] - xy[-1]) < 1e-6:
            xy, w_left, w_right = xy[:-1], np.asarray(w_left)[:-1], np.asarray(w_right)[:-1]
        w_left, w_right = np.asarray(w_left, float), np.asarray(w_right, float)
        # periodic spline through raw points, parameterised by chord length
        seg = np.linalg.norm(np.diff(np.vstack([xy, xy[:1]]), axis=0), axis=1)
        u = np.concatenate([[0], np.cumsum(seg)])
        P = np.vstack([xy, xy[:1]])
        sx = CubicSpline(u, P[:, 0], bc_type="periodic")
        sy = CubicSpline(u, P[:, 1], bc_type="periodic")
        wl = np.concatenate([w_left, w_left[:1]])
        wr = np.concatenate([w_right, w_right[:1]])
        # arc-length reparameterisation
        uu = np.linspace(0, u[-1], int(u[-1] / 0.25) + 1)
        dl = np.hypot(sx(uu, 1), sy(uu, 1))
        sarc = np.concatenate([[0], np.cumsum(0.5 * (dl[1:] + dl[:-1]) * np.diff(uu))])
        self.length = float(sarc[-1])
        n = int(round(self.length / ds))
        self.s = np.linspace(0, self.length, n, endpoint=False)
        self.ds = self.length / n
        uq = np.interp(self.s, sarc, uu)
        self.x, self.y = sx(uq), sy(uq)
        dx, dy = sx(uq, 1), sy(uq, 1)
        ddx, ddy = sx(uq, 2), sy(uq, 2)
        self.psi = np.unwrap(np.arctan2(dy, dx))
        k = (dx * ddy - dy * ddx) / np.power(dx * dx + dy * dy, 1.5)
        sig = max(kappa_smooth_m / self.ds, 1e-3)
        self.kappa = gaussian_filter1d(k, sig, mode="wrap")
        self.w_left = np.interp(uq, u, wl)
        self.w_right = np.interp(uq, u, wr)
        self.name = name
        self._tx, self._ty = np.cos(self.psi), np.sin(self.psi)

    # ---------- periodic interpolation ----------
    def interp(self, arr, s):
        s = np.mod(s, self.length)
        ss = np.concatenate([self.s, [self.length]])
        aa = np.concatenate([arr, [arr[0]]])
        return np.interp(s, ss, aa)

    def kappa_at(self, s):
        return self.interp(self.kappa, s)

    def psi_at(self, s):
        # unwrap-safe: interpolate tangent vector
        return np.arctan2(self.interp(self._ty, s), self.interp(self._tx, s))

    def bounds_at(self, s):
        return self.interp(self.w_left, s), self.interp(self.w_right, s)

    # ---------- coordinate transforms ----------
    def to_world(self, s, n):
        x, y = self.interp(self.x, s), self.interp(self.y, s)
        psi = self.psi_at(s)
        return x - n * np.sin(psi), y + n * np.cos(psi)

    def to_frenet(self, X, Y, psi=None, hint=None, window=60):
        """Project a world point. Returns (s, n, xi, idx). hint = previous idx."""
        N = len(self.s)
        if hint is None:
            idx = np.arange(N)
        else:
            idx = np.arange(hint - window, hint + window) % N
        d2 = (self.x[idx] - X) ** 2 + (self.y[idx] - Y) ** 2
        i = int(idx[np.argmin(d2)])
        best = None
        for j in (i - 1, i):
            a, b = j % N, (j + 1) % N
            ax_, ay_ = self.x[a], self.y[a]
            bx, by = self.x[b], self.y[b]
            vx, vy = bx - ax_, by - ay_
            L2 = vx * vx + vy * vy
            t = np.clip(((X - ax_) * vx + (Y - ay_) * vy) / L2, 0, 1)
            px, py = ax_ + t * vx, ay_ + t * vy
            d = (X - px) ** 2 + (Y - py) ** 2
            if best is None or d < best[0]:
                L = np.sqrt(L2)
                n = (vx * (Y - ay_) - vy * (X - ax_)) / L
                best = (d, self.s[a] + t * L, n)
        s = float(np.mod(best[1], self.length))
        xi = None if psi is None else float(wrap(psi - self.psi_at(s)))
        return s, float(best[2]), xi, i

    def ds_signed(self, s_new, s_old):
        """Signed progress accounting for the start/finish wrap."""
        d = s_new - s_old
        return (d + self.length / 2) % self.length - self.length / 2

    # ---------- loaders ----------
    @staticmethod
    def from_csv(path, name=None, **kw):
        data = np.genfromtxt(path, delimiter=",", names=True)
        cols = data.dtype.names
        if "w_tr_right_m" in cols:   # TUMFTM racetrack-database format
            xy = np.c_[data["x_m"], data["y_m"]]
            return Track(xy, data["w_tr_left_m"], data["w_tr_right_m"], name or Path(path).stem, **kw)
        xy = np.c_[data["x"], data["y"]]
        return Track(xy, data["w_left"], data["w_right"], name or Path(path).stem, **kw)

    @staticmethod
    def from_fast_lane_ai(path, sign_z=-1.0, swap_sides=False, name=None,
                          default_half_width=6.0, subsample=4, **kw):
        """Parse AC's AI spline. Layout (community-documented, v7):
            header  int32 version, count, lapTime, sampleCount
            count x { float x, y, z, length; int32 id }
            int32 extraCount
            extraCount x 18 floats: speed, gas, brake, obsoleteLatG, radius,
                sideLeft, sideRight, camber, direction, normal[3], length,
                forwardVector[3], tag, grade
        The extra block is parsed defensively; widths fall back to default_half_width.
        """
        b = Path(path).read_bytes()
        version, count, lap_time, sample_count = struct.unpack_from("<4i", b, 0)
        off = 16
        pts = np.frombuffer(b, dtype=np.dtype([("p", "<f4", 3), ("len", "<f4"), ("id", "<i4")]),
                            count=count, offset=off)
        off += count * 20
        wl = wr = None
        if len(b) >= off + 4:
            extra_count = struct.unpack_from("<i", b, off)[0]
            off += 4
            remaining = len(b) - off
            if extra_count == count and remaining >= count * 4 * 7:
                stride = remaining // (count * 4)
                stride = 18 if stride >= 18 else stride
                ex = np.frombuffer(b, dtype="<f4", count=count * stride, offset=off).reshape(count, stride)
                wl, wr = ex[:, 5].astype(float), ex[:, 6].astype(float)
                if not (np.all(np.isfinite(wl)) and 0.5 < np.median(wl) < 40):
                    wl = wr = None
        X = pts["p"][:, 0].astype(float)
        Y = sign_z * pts["p"][:, 2].astype(float)
        if wl is None:
            wl = wr = np.full(count, default_half_width)
        if swap_sides:  # if your sign_z choice mirrors the track, left/right swap
            wl, wr = wr, wl
        sl = slice(None, None, subsample)
        return Track(np.c_[X, Y][sl], np.asarray(wl)[sl], np.asarray(wr)[sl],
                     name or Path(path).parent.parent.name, **kw)

    @staticmethod
    def synthetic_club(half_width=6.0):
        """~1.7 km test circuit with a long straight, a hairpin, a chicane and sweepers.
        Shape is invented; use it to test the pipeline, not to predict real lap times."""
        cp = np.array([
            [0, 0], [250, 0], [420, 0], [500, 30], [520, 110], [480, 160], [420, 150],
            [380, 110], [330, 120], [300, 170], [320, 240], [280, 300], [180, 310],
            [120, 280], [90, 220], [40, 200], [-40, 210], [-110, 180], [-140, 110],
            [-120, 40], [-70, 5],
        ], float)
        w = np.full(len(cp), half_width)
        return Track(cp, w, w, name="synthetic_club")

    def summary(self):
        return (f"{self.name}: length {self.length:.0f} m, "
                f"min radius {1/np.max(np.abs(self.kappa)):.1f} m, "
                f"width {np.median(self.w_left + self.w_right):.1f} m")
