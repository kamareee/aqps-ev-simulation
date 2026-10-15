"""
WP3 — Layer 1: Queue Management with Active Priority Charging Queue (Q_k^act),
Laxity-Based Non-Priority Preemption, Safe Release, and Guarded FIFO Priority Rotation.

Queues maintained at each step k:
- Q_k^act : Active Priority Charging Queue (FIFO-ordered list of priority EVs currently on EVSEs, |Q_k^act| <= M).
- Q_k^p   : Priority Waiting Queue (priority EVs awaiting EVSE admission).
- Q_k^n   : Non-Priority Queue (non-priority EVs).

4-Tier Disconnection Hierarchy when all M EVSEs are occupied and an urgent priority EV arrives:
1. Tier 1 (Non-Priority Preemption): Disconnect connected non-priority EV with highest laxity.
2. Tier 2 (Safe Release from Q_k^act): Release priority EV in Q_k^act satisfying all 3 kWh safe-release
   conditions (a)-(c), largest surplus energy and latest departure first.
3. Tier 3 (Guarded FIFO Rotation from Q_k^act): Inspect priority EVs in Q_k^act in FIFO entry order
   (earliest-admitted first, having charged the longest). If incoming EV has higher ATC urgency
   (pi_new > pi_head) and the candidate retains positive slack (s_j > dt) or has met its immediate
   next-trip energy, rotate it to Q_k^p and admit the incoming priority EV to the tail of Q_k^act.
4. Tier 4 (Operator Escalation): Otherwise disconnect nobody; hold incoming EV in Q_k^p and escalate.
"""

from typing import List, Dict, Tuple, Set
from .data_structures import Vehicle


class QueueManager:
    def __init__(self, num_chargers: int, dt_hours: float = 0.25):
        self.num_chargers = num_chargers
        self.dt = dt_hours
        self.all_vehicles: Dict[str, Vehicle] = {}
        self.connected_ids: Set[str] = set()
        # Q_k^act: FIFO-ordered list of vehicle IDs currently charging as Priority EVs (max length M)
        self.active_priority_queue: List[str] = []
        # Track step index when each priority EV entered Q_k^act
        self.q_act_entry_step: Dict[str, int] = {}

    def process_arrivals(self, vehicles: List[Vehicle], current_time: int):
        for v in vehicles:
            self.all_vehicles[v.id] = v

    def process_departures(self, vehicle_ids: List[str]):
        for vid in vehicle_ids:
            if vid in self.all_vehicles:
                del self.all_vehicles[vid]
            self.connected_ids.discard(vid)
            if vid in self.active_priority_queue:
                self.active_priority_queue.remove(vid)
            self.q_act_entry_step.pop(vid, None)

    def update_vehicle_energy(self, vehicle_id: str, energy_delivered: float):
        if vehicle_id in self.all_vehicles:
            v = self.all_vehicles[vehicle_id]
            v.energy_delivered += energy_delivered
            needed = v.target_energy_kwh - v.energy_delivered
            v.energy_needed = max(0.0, needed)
            if v.energy_needed <= 0.05:
                self.connected_ids.discard(vehicle_id)
                if vehicle_id in self.active_priority_queue:
                    self.active_priority_queue.remove(vehicle_id)

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
          (b) No further trip today needs more energy before the next charging opportunity.
          (c) The overnight window from return to next-day first departure can still
              deliver the remaining next-day requirement at available charger power.
        """
        r = reserve_override_kwh if reserve_override_kwh is not None else v.reserve_kwh
        curr_e = v.current_energy_kwh
        req_mission = v.next_trip_kwh + v.return_kwh + r
        surplus = curr_e - req_mission

        if surplus < 0.0:
            return False, surplus
        if v.further_trips_today_kwh > surplus + 1e-3:
            return False, surplus

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
        Executes Layer 1 admission using Q_k^act (capacity M), Q_k^p, and Q_k^n.
        """
        events = {
            "preempted_np": [],
            "safe_released_p": [],
            "rotated_p": [],
            "escalated_p": [],
        }
        active_pool = [
            v for v in self.all_vehicles.values()
            if v.energy_needed > 0.05 and v.departure_time_idx > current_time
        ]
        if not active_pool:
            self.connected_ids.clear()
            self.active_priority_queue.clear()
            return [], events

        active_map = {v.id: v for v in active_pool}
        for v in active_pool:
            self._compute_laxity_hours(v, current_time)

        # Prune Q_k^act of vehicles that finished, departed, or are no longer priority
        self.active_priority_queue = [
            vid for vid in self.active_priority_queue
            if vid in active_map and active_map[vid].is_priority
        ]

        # Maintain currently connected dict
        connected: Dict[str, Vehicle] = {
            vid: active_map[vid]
            for vid in self.connected_ids
            if vid in active_map
        }

        # Ensure any connected priority vehicle is registered in Q_k^act in FIFO order
        for vid, v in connected.items():
            if v.is_priority and vid not in self.active_priority_queue:
                self.active_priority_queue.append(vid)
                self.q_act_entry_step.setdefault(vid, current_time)

        # Waiting priority vehicles Q_k^p (not currently in connected)
        waiting_priority = [
            v for v in active_pool if v.is_priority and v.id not in connected
        ]
        if use_atc_order:
            waiting_priority.sort(key=lambda v: (-v.atc_score, v.arrival_time_idx))
        else:
            waiting_priority.sort(key=lambda v: v.arrival_time_idx)

        # Process each waiting priority vehicle in Q_k^p
        for p_ev in waiting_priority:
            if len(connected) < self.num_chargers:
                connected[p_ev.id] = p_ev
                self.active_priority_queue.append(p_ev.id)
                self.q_act_entry_step[p_ev.id] = current_time
                continue

            # All M chargers are occupied: apply 4-Tier Disconnection Hierarchy
            # Tier 1: Connected non-priority EVs (Q_k^n), highest laxity victim first
            conn_np = [v for v in connected.values() if not v.is_priority]
            if conn_np:
                conn_np.sort(key=lambda v: v.laxity, reverse=True)
                victim = conn_np[0]
                del connected[victim.id]
                victim.preempted_count += 1
                events["preempted_np"].append(victim.id)
                connected[p_ev.id] = p_ev
                self.active_priority_queue.append(p_ev.id)
                self.q_act_entry_step[p_ev.id] = current_time
                continue

            # All M chargers are occupied by Priority EVs (|Q_k^act| == M)
            if enable_safe_release:
                # Tier 2: Full Safe Release (conditions a, b, c in kWh)
                releasable_p = []
                for vid in self.active_priority_queue:
                    c_ev = connected.get(vid)
                    if c_ev is not None:
                        ok, surplus = self.check_safe_release(c_ev, current_time, reserve_override_kwh)
                        if ok:
                            releasable_p.append((surplus, c_ev.departure_time_idx, c_ev))
                if releasable_p:
                    releasable_p.sort(key=lambda x: (x[0], x[1]), reverse=True)
                    _, _, rel_ev = releasable_p[0]
                    del connected[rel_ev.id]
                    if rel_ev.id in self.active_priority_queue:
                        self.active_priority_queue.remove(rel_ev.id)
                    rel_ev.safe_released = True
                    rel_ev.is_priority = False
                    events["safe_released_p"].append(rel_ev.id)
                    connected[p_ev.id] = p_ev
                    self.active_priority_queue.append(p_ev.id)
                    self.q_act_entry_step[p_ev.id] = current_time
                    continue

                # Tier 3: Guarded FIFO Rotation from Q_k^act (head of Q_k^act first)
                rotated_victim = None
                for vid in list(self.active_priority_queue):
                    head_ev = connected.get(vid)
                    if head_ev is None:
                        continue
                    # Guard condition: incoming priority EV is more urgent (pi_new > pi_head)
                    # AND head_ev has positive slack (laxity > dt) or has already covered next_trip_kwh
                    has_slack_or_trip = (head_ev.laxity > self.dt) or (head_ev.current_energy_kwh >= head_ev.next_trip_kwh + head_ev.reserve_kwh)
                    if p_ev.atc_score > head_ev.atc_score * 1.02 and has_slack_or_trip:
                        rotated_victim = head_ev
                        break

                if rotated_victim is not None:
                    del connected[rotated_victim.id]
                    self.active_priority_queue.remove(rotated_victim.id)
                    events["rotated_p"].append(rotated_victim.id)
                    events["safe_released_p"].append(rotated_victim.id)
                    connected[p_ev.id] = p_ev
                    self.active_priority_queue.append(p_ev.id)
                    self.q_act_entry_step[p_ev.id] = current_time
                    continue

            # Tier 4: Cannot disconnect without violating mission safety; hold in Q_k^p & escalate
            events["escalated_p"].append(p_ev.id)

        # Fill any remaining open EVSE slots with Non-Priority EVs (Q_k^n) in FIFO order
        non_priority_evs = [v for v in active_pool if not v.is_priority and v.id not in connected]
        non_priority_evs.sort(key=lambda v: v.arrival_time_idx)
        for np_ev in non_priority_evs:
            if len(connected) >= self.num_chargers:
                break
            connected[np_ev.id] = np_ev

        admitted = list(connected.values())
        self.connected_ids = {v.id for v in admitted}
        return admitted, events
