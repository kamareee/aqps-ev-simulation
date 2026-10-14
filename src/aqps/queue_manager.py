"""
WP3 — Layer 1: Queue Management with Laxity-Based Preemption and Safe Release.

Disconnection order when EVSE capacity (M) or transformer capacity is short:
1. Non-priority EVs, using laxity-based victim selection (largest laxity disconnected first).
2. Priority EVs that satisfy all 3 safe-release conditions in kWh (largest surplus energy
   and latest next departure first).
3. Otherwise disconnect nobody; the incoming priority vehicle waits, and the system
   escalates to the operator impact-preview loop.
"""

from typing import List, Dict, Tuple, Set
from .data_structures import Vehicle


class QueueManager:
    def __init__(self, num_chargers: int, dt_hours: float = 0.25):
        self.num_chargers = num_chargers
        self.dt = dt_hours
        self.all_vehicles: Dict[str, Vehicle] = {}
        self.connected_ids: Set[str] = set()

    def process_arrivals(self, vehicles: List[Vehicle], current_time: int):
        for v in vehicles:
            self.all_vehicles[v.id] = v

    def process_departures(self, vehicle_ids: List[str]):
        for vid in vehicle_ids:
            if vid in self.all_vehicles:
                del self.all_vehicles[vid]
            self.connected_ids.discard(vid)

    def update_vehicle_energy(self, vehicle_id: str, energy_delivered: float):
        if vehicle_id in self.all_vehicles:
            v = self.all_vehicles[vehicle_id]
            v.energy_delivered += energy_delivered
            needed = v.target_energy_kwh - v.energy_delivered
            v.energy_needed = max(0.0, needed)

    def get_queue_size(self) -> int:
        active = [v for v in self.all_vehicles.values() if v.energy_needed > 0.05]
        return max(0, len(active) - self.num_chargers)

    def _compute_laxity_hours(self, v: Vehicle, current_time: int) -> float:
        rem_h = (v.departure_time_idx - current_time) * self.dt
        if rem_h <= 0:
            return -999.0
        charge_h = v.energy_needed / max(v.max_charging_power_kw, 0.1)
        v.laxity = rem_h - charge_h
        return v.laxity

    def check_safe_release(
        self, v: Vehicle, current_time: int, reserve_override_kwh: float = None
    ) -> Tuple[bool, float]:
        """
        Safe-release condition — computed strictly in kWh, never % SoC.
        All three conditions must hold:
          (a) Current energy >= next_trip_kwh + return_kwh + reserve_kwh.
          (b) No further trip today needs more energy before the next charging opportunity
              (further_trips_today_kwh <= current_energy - (next_trip + return + reserve)).
          (c) The overnight window from return to next-day first departure can still
              deliver the remaining next-day requirement at available charger power.
        Returns: (is_safe_to_release, surplus_kwh)
        """
        r = reserve_override_kwh if reserve_override_kwh is not None else v.reserve_kwh
        curr_e = v.current_energy_kwh
        req_mission = v.next_trip_kwh + v.return_kwh + r
        surplus = curr_e - req_mission

        # Condition (a): Current energy covers next trip + return + reserve r
        if surplus < 0.0:
            return False, surplus

        # Condition (b): No further trip today needs more energy before next charging window
        if v.further_trips_today_kwh > surplus + 1e-3:
            return False, surplus

        # Condition (c): Overnight window from evening return (~step 64 = 22:00) to
        # next-day first departure can deliver remaining requirement at available power
        overnight_return_idx = max(v.departure_time_idx + 8, 60)
        overnight_steps = max(0, v.next_day_dep_idx - overnight_return_idx)
        overnight_deliverable_kwh = overnight_steps * self.dt * v.max_charging_power_kw
        if overnight_deliverable_kwh < v.next_day_req_kwh - surplus:
            return False, surplus

        return True, surplus

    def get_active_assignment(
        self,
        current_time: int,
        current_system_capacity: float,
        enable_safe_release: bool = True,
        use_atc_order: bool = True,
        reserve_override_kwh: float = None,
    ) -> Tuple[List[Vehicle], Dict[str, List[str]]]:
        """
        Executes Layer 1 admission, laxity-based non-priority preemption,
        safe-release of priority EVs, and operator escalation.

        Returns:
            (admitted_vehicles, events_dict)
            where events_dict has keys: 'preempted_np', 'safe_released_p', 'escalated_p'
        """
        events = {"preempted_np": [], "safe_released_p": [], "escalated_p": []}
        active_pool = [
            v for v in self.all_vehicles.values()
            if v.energy_needed > 0.05 and v.departure_time_idx > current_time
        ]
        if not active_pool:
            self.connected_ids.clear()
            return [], events

        for v in active_pool:
            self._compute_laxity_hours(v, current_time)

        priority_evs = [v for v in active_pool if v.is_priority]
        non_priority_evs = [v for v in active_pool if not v.is_priority]

        # Sort priority EVs: already connected first, then ATC score (or FIFO for v1)
        if use_atc_order:
            priority_evs.sort(
                key=lambda v: (v.id not in self.connected_ids, -v.atc_score, v.arrival_time_idx)
            )
        else:
            priority_evs.sort(key=lambda v: (v.id not in self.connected_ids, v.arrival_time_idx))

        # Sort non-priority EVs: FIFO for admission
        non_priority_evs.sort(key=lambda v: (v.id not in self.connected_ids, v.arrival_time_idx))

        # Start with currently connected vehicles that are still active
        connected: Dict[str, Vehicle] = {
            v.id: v for v in active_pool if v.id in self.connected_ids
        }

        # Step 1: Admit priority vehicles
        for p_ev in priority_evs:
            if p_ev.id in connected:
                continue

            if len(connected) < self.num_chargers:
                connected[p_ev.id] = p_ev
                continue

            # Charger capacity is full (|C_k| == M): follow 3-tier disconnection rules
            # Tier 1: Connected non-priority EVs, highest laxity victim first
            conn_np = [v for v in connected.values() if not v.is_priority]
            if conn_np:
                conn_np.sort(key=lambda v: v.laxity, reverse=True)
                victim = conn_np[0]
                del connected[victim.id]
                victim.preempted_count += 1
                events["preempted_np"].append(victim.id)
                connected[p_ev.id] = p_ev
                continue

            # Tier 2: Connected priority EVs satisfying Safe-Release (a)-(c),
            # largest surplus energy and latest next departure first
            if enable_safe_release:
                releasable_p = []
                for c_ev in connected.values():
                    if c_ev.is_priority:
                        ok, surplus = self.check_safe_release(c_ev, current_time, reserve_override_kwh)
                        if ok:
                            releasable_p.append((surplus, c_ev.departure_time_idx, c_ev))
                if releasable_p:
                    releasable_p.sort(key=lambda x: (x[0], x[1]), reverse=True)
                    _, _, rel_ev = releasable_p[0]
                    del connected[rel_ev.id]
                    rel_ev.safe_released = True
                    rel_ev.is_priority = False
                    events["safe_released_p"].append(rel_ev.id)
                    connected[p_ev.id] = p_ev
                    continue

            # Tier 3: Disconnect nobody; incoming priority EV waits and escalates to operator
            events["escalated_p"].append(p_ev.id)

        # Step 2: Fill remaining EVSE slots with non-priority vehicles in FIFO order
        for np_ev in non_priority_evs:
            if len(connected) >= self.num_chargers:
                break
            if np_ev.id not in connected:
                connected[np_ev.id] = np_ev

        admitted = list(connected.values())
        self.connected_ids = {v.id for v in admitted}
        return admitted, events
