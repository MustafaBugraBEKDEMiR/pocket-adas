"""Event and status types produced by the ADAS pipeline."""
from dataclasses import dataclass, field
from typing import Optional

HARSH_BRAKE = "HARSH_BRAKE"
HARSH_ACCEL = "HARSH_ACCEL"
HARSH_TURN = "HARSH_TURN"
BUMP = "BUMP"
POTHOLE = "POTHOLE"
SPEEDING = "SPEEDING"
SUDDEN_MOVE = "SUDDEN_MOVE"  # horizontal jolt before the forward axis is known

ALL_KINDS = [HARSH_BRAKE, HARSH_ACCEL, HARSH_TURN, BUMP, POTHOLE, SPEEDING, SUDDEN_MOVE]

MESSAGES = {
    HARSH_BRAKE: "HARSH BRAKING",
    HARSH_ACCEL: "HARSH ACCELERATION",
    HARSH_TURN: "SHARP TURN",
    BUMP: "BUMP TAKEN FAST",
    POTHOLE: "POTHOLE",
    SPEEDING: "SPEEDING",
    SUDDEN_MOVE: "SUDDEN MANEUVER",
}

# Status levels shown on the phone screen
CALIBRATING = "CALIBRATING"
SAFE = "SAFE"
WARNING = "WARNING"
DANGER = "DANGER"


@dataclass
class Event:
    t: float                 # seconds since start
    kind: str
    peak: float              # m/s^2 for motion events, km/h for speeding
    severity: float          # peak / threshold (>= 1.0)
    speed_kmh: float = 0.0
    lat: Optional[float] = None
    lon: Optional[float] = None
    duration: float = 0.0    # filled when the event ends
    limit_kmh: Optional[float] = None

    @property
    def message(self) -> str:
        return MESSAGES.get(self.kind, self.kind)

    def to_dict(self) -> dict:
        return {
            "t": round(self.t, 3), "kind": self.kind, "peak": round(self.peak, 3),
            "severity": round(self.severity, 3), "speed_kmh": round(self.speed_kmh, 1),
            "lat": self.lat, "lon": self.lon, "duration": round(self.duration, 2),
            "limit_kmh": self.limit_kmh,
        }


@dataclass
class Status:
    level: str = CALIBRATING
    message: str = "CALIBRATING - KEEP STILL"
    event: Optional[Event] = field(default=None, repr=False)
