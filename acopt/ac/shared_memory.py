"""
Assetto Corsa shared-memory reader (Windows only).

Pages: Local\\acpmf_physics (updates at the physics rate), Local\\acpmf_graphics
(updates at the render rate), Local\\acpmf_static. Layouts follow the AC 1.x
SDK headers (#pragma pack(4)). Only fields up to localVelocity are declared,
which is everything this framework needs.

Verify on your install with scripts/ac_probe.py before trusting it.
"""
from __future__ import annotations

import ctypes as C
import mmap
import sys

f = C.c_float
i = C.c_int
F3, F4 = f * 3, f * 4


class SPageFilePhysics(C.Structure):
    _pack_ = 4
    _fields_ = [
        ("packetId", i), ("gas", f), ("brake", f), ("fuel", f), ("gear", i), ("rpms", i),
        ("steerAngle", f), ("speedKmh", f), ("velocity", F3), ("accG", F3),
        ("wheelSlip", F4), ("wheelLoad", F4), ("wheelsPressure", F4), ("wheelAngularSpeed", F4),
        ("tyreWear", F4), ("tyreDirtyLevel", F4), ("tyreCoreTemperature", F4), ("camberRAD", F4),
        ("suspensionTravel", F4), ("drs", f), ("tc", f), ("heading", f), ("pitch", f), ("roll", f),
        ("cgHeight", f), ("carDamage", f * 5), ("numberOfTyresOut", i), ("pitLimiterOn", i),
        ("abs", f), ("kersCharge", f), ("kersInput", f), ("autoShifterOn", i), ("rideHeight", f * 2),
        ("turboBoost", f), ("ballast", f), ("airDensity", f), ("airTemp", f), ("roadTemp", f),
        ("localAngularVel", F3), ("finalFF", f), ("performanceMeter", f), ("engineBrake", i),
        ("ersRecoveryLevel", i), ("ersPowerLevel", i), ("ersHeatCharging", i), ("ersIsCharging", i),
        ("kersCurrentKJ", f), ("drsAvailable", i), ("drsEnabled", i), ("brakeTemp", F4), ("clutch", f),
        ("tyreTempI", F4), ("tyreTempM", F4), ("tyreTempO", F4), ("isAIControlled", i),
        ("tyreContactPoint", F3 * 4), ("tyreContactNormal", F3 * 4), ("tyreContactHeading", F3 * 4),
        ("brakeBias", f), ("localVelocity", F3),
    ]


W = C.c_uint16  # wchar_t is 2 bytes on Windows


class SPageFileGraphics(C.Structure):
    _pack_ = 4
    _fields_ = [
        ("packetId", i), ("status", i), ("session", i),
        ("currentTime", W * 15), ("lastTime", W * 15), ("bestTime", W * 15), ("split", W * 15),
        ("completedLaps", i), ("position", i), ("iCurrentTime", i), ("iLastTime", i), ("iBestTime", i),
        ("sessionTimeLeft", f), ("distanceTraveled", f), ("isInPit", i), ("currentSectorIndex", i),
        ("lastSectorTime", i), ("numberOfLaps", i), ("tyreCompound", W * 33),
        ("replayTimeMultiplier", f), ("normalizedCarPosition", f), ("carCoordinates", F3),
        ("penaltyTime", f), ("flag", i), ("idealLineOn", i), ("isInPitLane", i), ("surfaceGrip", f),
    ]


class SPageFileStatic(C.Structure):
    _pack_ = 4
    _fields_ = [
        ("smVersion", W * 15), ("acVersion", W * 15), ("numberOfSessions", i), ("numCars", i),
        ("carModel", W * 33), ("track", W * 33), ("playerName", W * 33), ("playerSurname", W * 33),
        ("playerNick", W * 33), ("sectorCount", i), ("maxTorque", f), ("maxPower", f), ("maxRpm", i),
        ("maxFuel", f), ("suspensionMaxTravel", F4), ("tyreRadius", F4),
    ]


def wstr(arr) -> str:
    return "".join(chr(c) for c in arr if c).strip()


class ACSharedMemory:
    def __init__(self):
        if sys.platform != "win32":
            raise RuntimeError("AC shared memory is only available on Windows")
        self._maps = {}
        self.physics = self._open("Local\\acpmf_physics", SPageFilePhysics)
        self.graphics = self._open("Local\\acpmf_graphics", SPageFileGraphics)
        self.static = self._open("Local\\acpmf_static", SPageFileStatic)

    def _open(self, tag, struct):
        mm = mmap.mmap(-1, C.sizeof(struct), tag, access=mmap.ACCESS_READ)
        self._maps[tag] = mm
        return lambda: struct.from_buffer_copy(mm[: C.sizeof(struct)])

    def close(self):
        for mm in self._maps.values():
            mm.close()
