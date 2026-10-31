"""Trip score: 100 minus severity-weighted penalties per km."""
from collections import Counter

from .events import HARSH_BRAKE, HARSH_ACCEL, HARSH_TURN, BUMP, POTHOLE, SPEEDING, SUDDEN_MOVE

PENALTY = {
    HARSH_BRAKE: 6.0, HARSH_ACCEL: 4.0, HARSH_TURN: 5.0, SPEEDING: 5.0,
    BUMP: 3.0, POTHOLE: 1.0, SUDDEN_MOVE: 4.0,
}


def trip_score(events, distance_m: float) -> int:
    km = max(distance_m / 1000.0, 2.0)  # short trips are judged as 2 km
    penalty = sum(PENALTY.get(e.kind, 3.0) * min(e.severity, 2.5) for e in events)
    return int(max(0, round(100 - penalty / km)))


def event_counts(events) -> dict:
    return dict(Counter(e.kind for e in events))
