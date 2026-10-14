"""
Data structures for the revised 3-Layer AQPS framework (Layer 0, Layer 1, Layer 2 + Operator Loop).
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict
from enum import Enum


@dataclass
class Vehicle:
    """
    Represents a commercial fleet EV with multi-trip roster metadata.
    Grounded in empirical commercial van parameters (Vales-Alonso & Alcaraz, 2026; Fiori & Marzano, 2018).
    """
    id: str
    arrival_time_idx: int
    departure_time_idx: int
    target_energy_kwh: float
    max_charging_power_kw: float
    is_priority: bool = False
    initial_energy_kwh: float = 0.0
    energy_delivered: float = 0.0
    energy_needed: float = field(init=False)

    # WP2 / WP3 Layer 0 & Safe-Release attributes (all in kWh, never % SoC)
    battery_cap_kwh: float = 68.0
    eod_soc_kwh: float = 15.0
    next_trip_kwh: float = 18.0
    return_kwh: float = 6.0
    reserve_kwh: float = 6.8          # 10% safety floor (SoC_min = 0.10)
    further_trips_today_kwh: float = 0.0
    next_day_dep_idx: int = 96        # Next-day first departure step index
    next_day_req_kwh: float = 25.0    # Energy required for next-day first shift
    trip_criticality_w: float = 1.0   # Roster criticality weight w_i
    archetype: str = "Medium"
    day_ahead_priority: bool = False  # Operator/day-ahead initial flag
    atc_score: float = 0.0
    laxity: float = 0.0
    safe_released: bool = False
    preempted_count: int = 0

    def __post_init__(self):
        # E_i^req is the mission-sufficient session requirement
        self.energy_needed = max(0.0, self.target_energy_kwh - self.energy_delivered)

    @property
    def current_energy_kwh(self) -> float:
        return min(self.battery_cap_kwh, self.eod_soc_kwh + self.energy_delivered)

    @property
    def mission_sufficient_kwh(self) -> float:
        """Minimum energy needed in battery at departure for next trip + return + reserve."""
        return min(self.battery_cap_kwh, self.next_trip_kwh + self.return_kwh + self.reserve_kwh)

    @property
    def session_req_remaining_kwh(self) -> float:
        return max(0.0, self.target_energy_kwh - self.energy_delivered)

    def clone(self) -> "Vehicle":
        v = Vehicle(
            id=self.id,
            arrival_time_idx=self.arrival_time_idx,
            departure_time_idx=self.departure_time_idx,
            target_energy_kwh=self.target_energy_kwh,
            max_charging_power_kw=self.max_charging_power_kw,
            is_priority=self.is_priority,
            initial_energy_kwh=self.initial_energy_kwh,
            energy_delivered=self.energy_delivered,
            battery_cap_kwh=self.battery_cap_kwh,
            eod_soc_kwh=self.eod_soc_kwh,
            next_trip_kwh=self.next_trip_kwh,
            return_kwh=self.return_kwh,
            reserve_kwh=self.reserve_kwh,
            further_trips_today_kwh=self.further_trips_today_kwh,
            next_day_dep_idx=self.next_day_dep_idx,
            next_day_req_kwh=self.next_day_req_kwh,
            trip_criticality_w=self.trip_criticality_w,
            archetype=self.archetype,
            day_ahead_priority=self.day_ahead_priority,
            atc_score=self.atc_score,
            laxity=self.laxity,
            safe_released=self.safe_released,
            preempted_count=self.preempted_count,
        )
        v.energy_needed = self.energy_needed
        return v
