# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring

import heapq
from typing import List, Dict, Optional
from .data_structures import Vehicle


class QueueManager:
    def __init__(self, num_chargers: int):
        self.num_chargers = num_chargers
        self.all_vehicles: Dict[str, Vehicle] = {}

    def process_arrivals(self, vehicles: List[Vehicle], current_time: int):
        for v in vehicles:
            # We assume FMS or Scenario generator sets 'is_priority' boolean
            self.all_vehicles[v.id] = v

    def process_departures(self, vehicle_ids: List[str]):
        for vid in vehicle_ids:
            if vid in self.all_vehicles:
                del self.all_vehicles[vid]

    def update_vehicle_energy(self, vehicle_id: str, energy_delivered: float):
        if vehicle_id in self.all_vehicles:
            self.all_vehicles[vehicle_id].energy_delivered += energy_delivered
            # Update remaining needed
            needed = (
                self.all_vehicles[vehicle_id].target_energy_kwh
                - self.all_vehicles[vehicle_id].initial_energy_kwh
                - self.all_vehicles[vehicle_id].energy_delivered
            )
            self.all_vehicles[vehicle_id].energy_needed = max(0.0, needed)

    def get_queue_size(self) -> int:
        return max(0, len(self.all_vehicles) - self.num_chargers)

    def _calculate_min_required_power(
        self, vehicle: Vehicle, current_time: int
    ) -> float:
        """Calculates P_min_i(t)"""
        time_remaining_hours = (vehicle.departure_time_idx - current_time) * 0.25
        if time_remaining_hours <= 0:
            return vehicle.max_charging_power_kw  # It's late! Max power!
        return min(
            vehicle.max_charging_power_kw, vehicle.energy_needed / time_remaining_hours
        )

    def get_active_assignment(
        self, current_time: int, current_system_capacity: float
    ) -> List[Vehicle]:
        """
        Layer 1 Logic:
        1. Sort by Priority Status (1 > 0).
        2. Tie-Breaker: Arrival Time (FIFO).
        3. Feasibility Verification: Check if Priority min power > Current System Capacity (Grid+PV+BESS).
        4. Preemptive Reallocation: Drop lowest ranked if infeasible.
        """
        if not self.all_vehicles:
            return []

        # 1. Sorting
        scored_vehicles = []
        for v in self.all_vehicles.values():
            # Primary: Priority (High = Priority). We negate it for ascending sort or use Desc.
            # Here: Priority Rank 0 = Priority, 1 = Non-Priority (so we sort Ascending to put Priority first)
            priority_rank = 0 if getattr(v, "is_priority", False) else 1

            # Secondary: Arrival Time (Earlier is better -> Ascending sort works)
            arrival_time = getattr(v, "arrival_time_idx", 0)

            scored_vehicles.append((priority_rank, arrival_time, v))

        # Sort: Priority first (0), then earlier Arrival Time
        scored_vehicles.sort(key=lambda x: (x[0], x[1]))

        # Initial candidates (Top M)
        candidates = [x[2] for x in scored_vehicles[: self.num_chargers]]

        # 2. Feasibility Verification
        # Check against the dynamic current_system_capacity (which includes Grid + PV_now + BESS_now)

        while True:
            priority_load = 0.0
            for v in candidates:
                if getattr(v, "is_priority", False):
                    priority_load += self._calculate_min_required_power(v, current_time)

            if priority_load <= current_system_capacity:
                break  # Feasible

            # 3. Preemptive Drop
            if not candidates:
                break
            # Remove the last one (lowest ranked)
            removed = candidates.pop()
            print(
                f"Warning: Dropped vehicle {removed.id} due to Feasibility Check (Overload at t={current_time})."
            )

        return candidates
