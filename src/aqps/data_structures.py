"""
Vehicle state for the AQPS simulator (revision v3). All energies in kWh (never % SoC).

E_session : energy the operator would like added this visit (soft target, Eq. 3/9)
E_mission : next trip + return + reserve - arrival energy (hard target for priority EVs, Eq. 10)
"""

from dataclasses import dataclass, field, replace
from typing import Dict, Any


@dataclass
class Vehicle:
    id: str
    arrival_idx: int
    departure_idx: int
    p_max_kw: float
    battery_cap_kwh: float
    eod_soc_kwh: float
    e_session_kwh: float
    e_mission_kwh: float
    next_trip_kwh: float
    return_kwh: float
    reserve_kwh: float
    further_trips_today_kwh: float = 0.0
    evening_return_idx: int = 0
    next_day_dep_idx: int = 0
    next_day_req_kwh: float = 0.0
    trip_criticality_w: float = 1.0
    day_ahead_priority: bool = False   # initial operator/day-ahead flag (never changed)
    roster_priority: bool = False      # true mission criticality, updated by intra-day injections
    overnight: bool = False

    # dynamic state
    is_priority: bool = False
    energy_delivered: float = 0.0
    atc_score: float = 0.0
    laxity_h: float = 0.0
    safe_released: bool = False
    preempted_count: int = 0
    rotated_count: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    # ---- derived quantities -------------------------------------------------
    @property
    def session_remaining(self) -> float:
        return max(0.0, self.e_session_kwh - self.energy_delivered)

    @property
    def mission_remaining(self) -> float:
        return max(0.0, self.e_mission_kwh - self.energy_delivered)

    @property
    def current_energy_kwh(self) -> float:
        return min(self.battery_cap_kwh, self.eod_soc_kwh + self.energy_delivered)

    @property
    def mission_level_kwh(self) -> float:
        """Battery energy needed for next trip + return + reserve (safe-release condition (a))."""
        return self.next_trip_kwh + self.return_kwh + self.reserve_kwh

    @property
    def done(self) -> bool:
        return self.session_remaining <= 1e-3

    def present(self, t: int) -> bool:
        return self.arrival_idx <= t < self.departure_idx

    def clone(self) -> "Vehicle":
        return replace(self, extra=dict(self.extra))

    @staticmethod
    def from_record(ev: Dict[str, Any], reserve_frac: float = None) -> "Vehicle":
        is_p = ev["priority"] == "High"
        reserve = float(ev["reserve_kwh"])
        trip = float(ev["next_trip_kwh"])
        ret = float(ev["return_kwh"])
        eod = float(ev["eod_soc_kwh"])
        e_sess = float(ev["e_session_kwh"])
        e_miss = float(ev["e_mission_kwh"])
        if reserve_frac is not None:  # E6 reserve sweep: recompute mission with new reserve
            reserve = reserve_frac * float(ev["battery_cap_kwh"])
            e_miss = max(0.0, trip + ret + reserve - eod)
            e_sess = max(e_sess, e_miss)
        return Vehicle(
            id=ev["id"], arrival_idx=int(ev["arrival_idx"]), departure_idx=int(ev["departure_idx"]),
            p_max_kw=float(ev["p_max"]), battery_cap_kwh=float(ev["battery_cap_kwh"]),
            eod_soc_kwh=eod, e_session_kwh=e_sess, e_mission_kwh=min(e_miss, e_sess),
            next_trip_kwh=trip, return_kwh=ret, reserve_kwh=reserve,
            further_trips_today_kwh=float(ev.get("further_trips_today_kwh", 0.0)),
            evening_return_idx=int(ev["evening_return_idx"]),
            next_day_dep_idx=int(ev["next_day_dep_idx"]),
            next_day_req_kwh=float(ev["next_day_req_kwh"]),
            trip_criticality_w=float(ev["trip_criticality_w"]),
            day_ahead_priority=is_p, roster_priority=is_p, overnight=bool(ev.get("overnight", False)),
            is_priority=is_p,
        )
