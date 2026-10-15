# Save as: fix_physics_and_remove_v1.py
"""
Updates the AQPS simulation codebase with:
1. Active Priority Charging Queue (Q_k^act, capacity M) + Guarded FIFO Priority Rotation in Layer 1.
2. Fix for OperatorImpactPreview in-place priority flag mutation bug.
3. Distinct session target vs. mission-sufficient Safe-Release buffer.
4. Removal of AQPS-v1 across all experiments, tables, and figures.
"""

import shutil
from pathlib import Path

# =============================================================================
# 1. Clean all previous run results and figure/table outputs FIRST
# =============================================================================
for folder_name in ["results", "results_resubmission", "figs_epsr", "tables_epsr"]:
    folder = Path(folder_name)
    if folder.exists():
        shutil.rmtree(folder)
        print(f"Cleaned previous directory: {folder_name}/")

# =============================================================================
# 2. Update src/aqps/queue_manager.py (with Q_k^act, Safe Release & FIFO Rotation)
# =============================================================================
QUEUE_MANAGER_CODE = '''"""
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
'''
Path("src/aqps/queue_manager.py").write_text(QUEUE_MANAGER_CODE, encoding="utf-8")
print("Updated: src/aqps/queue_manager.py (with Q_k^act, Safe Release & Guarded FIFO Rotation)")

# =============================================================================
# 3. Update src/aqps/impact_preview.py (fix what-if mutation bug)
# =============================================================================
IMPACT_PREVIEW_CODE = '''"""
WP4 — Operator Impact-Preview Loop (Grounded in Parasuraman, Sheridan & Wickens, 2000:
Level of Automation "executes recommended option provisionally unless vetoed").
"""

from dataclasses import dataclass
from typing import List, Dict
import numpy as np
from .data_structures import Vehicle
from .tou_optimization import TOUOptimizer


@dataclass
class PreviewOption:
    option_id: str
    description: str
    priority_ids: List[str]
    expected_departure_kwh: Dict[str, float]
    estimated_horizon_cost: float
    cost_delta: float
    solve_time_ms: float


@dataclass
class ImpactPreviewRecord:
    time_idx: int
    trigger_reason: str
    affected_vehicles: List[str]
    options: List[PreviewOption]
    selected_option_id: str
    extra_cost_accepted: float
    total_preview_solve_ms: float


class OperatorImpactPreview:
    def __init__(
        self,
        optimizer: TOUOptimizer,
        cost_threshold_dollars: float = 5.0,
        dt_hours: float = 0.25,
    ):
        self.optimizer = optimizer
        self.cost_threshold_dollars = cost_threshold_dollars
        self.dt = dt_hours
        self.preview_logs: List[ImpactPreviewRecord] = []

    def evaluate_preview(
        self,
        time_idx: int,
        trigger_reason: str,
        admitted_vehicles: List[Vehicle],
        escalated_ids: List[str],
        horizon: int,
        tou_prices: np.ndarray,
        pv_forecast: np.ndarray,
        bess_forecast: np.ndarray,
        baseline_cost: float,
        operator_script_choice: str = "recommended",
    ) -> ImpactPreviewRecord:
        options: List[PreviewOption] = []
        total_ms = 0.0

        # Snapshot Layer 0/1 recommended priority set BEFORE running what-if options
        recommended_pri_ids = {v.id for v in admitted_vehicles if v.is_priority}

        def run_what_if(opt_id: str, desc: str, pri_set: set) -> PreviewOption:
            nonlocal total_ms
            opt_evs = [
                {
                    "id": v.id,
                    "energy_needed": v.energy_needed,
                    "max_power": v.max_charging_power_kw,
                    "dep_time_idx": v.departure_time_idx,
                    "is_priority": (v.id in pri_set),
                }
                for v in admitted_vehicles
            ]
            res = self.optimizer.solve(
                current_time_idx=time_idx,
                horizon=horizon,
                active_evs=opt_evs,
                tou_prices=tou_prices,
                pv_generation=pv_forecast,
                bess_generation=bess_forecast,
                enforce_priority_min=True,
            )
            total_ms += res.solve_time_ms
            exp_dep = {}
            for v in admitted_vehicles:
                deliv_h = sum(res.schedule.get(v.id, [])) * self.dt
                exp_dep[v.id] = round(v.current_energy_kwh + deliv_h, 2)
            return PreviewOption(
                option_id=opt_id,
                description=desc,
                priority_ids=sorted(list(pri_set)),
                expected_departure_kwh=exp_dep,
                estimated_horizon_cost=round(res.total_cost, 2),
                cost_delta=round(res.total_cost - baseline_cost, 2),
                solve_time_ms=res.solve_time_ms,
            )

        # Option 1: Recommended (Layer-0 feasible priority set)
        opt_rec = run_what_if(
            "recommended",
            "Recommended: Layer-0 feasible priority set with safe release & Q_act rotation",
            recommended_pri_ids,
        )
        options.append(opt_rec)

        # Option 2: Keep all urgent (promote highest-ATC non-priority vehicle)
        promote_set = set(recommended_pri_ids)
        np_cands = [v for v in admitted_vehicles if v.id not in recommended_pri_ids]
        if np_cands:
            np_cands.sort(key=lambda v: v.atc_score, reverse=True)
            promote_set.add(np_cands[0].id)
        opt_keep = run_what_if(
            "keep_all_urgent",
            "Keep All Urgent: Promote marginal vehicle at higher peak grid cost",
            promote_set,
        )
        options.append(opt_keep)

        # Option 3: Downgrade lowest-ATC priority vehicle
        downgrade_set = set(recommended_pri_ids)
        pri_cands = [v for v in admitted_vehicles if v.id in recommended_pri_ids]
        if len(pri_cands) > 1:
            pri_cands.sort(key=lambda v: v.atc_score)
            downgrade_set.discard(pri_cands[0].id)
        opt_down = run_what_if(
            "downgrade_marginal",
            "Cost-Saving Alternative: Downgrade lowest-urgency priority vehicle",
            downgrade_set,
        )
        options.append(opt_down)

        chosen = opt_rec
        for o in options:
            if o.option_id == operator_script_choice:
                chosen = o
                break

        chosen_set = set(chosen.priority_ids)
        for v in admitted_vehicles:
            v.is_priority = (v.id in chosen_set)

        extra_cost = max(0.0, chosen.estimated_horizon_cost - opt_rec.estimated_horizon_cost)
        affected = sorted(list(set(escalated_ids) | (recommended_pri_ids ^ chosen_set)))

        rec = ImpactPreviewRecord(
            time_idx=time_idx,
            trigger_reason=trigger_reason,
            affected_vehicles=affected,
            options=options,
            selected_option_id=chosen.option_id,
            extra_cost_accepted=round(extra_cost, 2),
            total_preview_solve_ms=round(total_ms, 3),
        )
        self.preview_logs.append(rec)
        return rec
'''
Path("src/aqps/impact_preview.py").write_text(IMPACT_PREVIEW_CODE, encoding="utf-8")
print("Updated: src/aqps/impact_preview.py")

# =============================================================================
# 4. Update src/aqps/priority_selection.py
# =============================================================================
PRIORITY_SELECTION_CODE = '''"""
WP3 — Layer 0: Real-Time Schedule-Derived Priority Selection with Optimizer Feasibility Check.
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Set
from .data_structures import Vehicle
from .tou_optimization import TOUOptimizer


@dataclass
class Layer0Result:
    priority_ids: Set[str]
    atc_scores: Dict[str, float]
    escalated_ids: List[str]
    qp_solves_used: int
    solve_time_ms_total: float
    predicted_shortfalls: Dict[str, float] = field(default_factory=dict)


class PrioritySelectorLayer0:
    def __init__(
        self,
        optimizer: TOUOptimizer,
        num_chargers: int,
        dt_hours: float = 0.25,
        k_atc: float = 2.0,
        shortfall_tol_kwh: float = 0.50,
    ):
        self.optimizer = optimizer
        self.num_chargers = num_chargers
        self.dt = dt_hours
        self.k_atc = k_atc
        self.shortfall_tol_kwh = shortfall_tol_kwh

    @staticmethod
    def compute_session_requirement(v: Vehicle, reserve_override_kwh: float = None) -> float:
        r = reserve_override_kwh if reserve_override_kwh is not None else v.reserve_kwh
        mission_min = max(5.0, v.next_trip_kwh + v.return_kwh + r - v.eod_soc_kwh)
        return round(max(v.target_energy_kwh, mission_min), 2)

    def compute_atc_scores(
        self, vehicles: List[Vehicle], current_time_idx: int
    ) -> Dict[str, float]:
        active = [v for v in vehicles if v.energy_needed > 0.1 and v.departure_time_idx > current_time_idx]
        if not active:
            return {v.id: 0.0 for v in vehicles}

        p_dict = {}
        s_dict = {}
        for v in active:
            p_i = max(0.15, v.energy_needed / max(v.max_charging_power_kw, 1.0))
            rem_hours = max(0.0, (v.departure_time_idx - current_time_idx) * self.dt)
            s_i = rem_hours - p_i
            p_dict[v.id] = p_i
            s_dict[v.id] = s_i
            v.laxity = s_i

        p_bar = max(0.25, float(np.mean(list(p_dict.values()))))
        denom = max(0.15, self.k_atc * p_bar)

        scores = {}
        for v in vehicles:
            if v.id not in p_dict:
                scores[v.id] = 0.0
                v.atc_score = 0.0
                continue
            p_i = p_dict[v.id]
            s_i = s_dict[v.id]
            w_i = float(v.trip_criticality_w)
            tardy_boost = 1.0 + max(0.0, -s_i) * 0.20
            pi_i = (w_i / p_i) * math.exp(-max(0.0, s_i) / denom) * tardy_boost
            scores[v.id] = pi_i
            v.atc_score = pi_i

        return scores

    def _check_subset_feasibility(
        self,
        candidate_priority_evs: List[Vehicle],
        current_time_idx: int,
        tou_prices: np.ndarray,
        pv_forecast: np.ndarray,
        bess_forecast: np.ndarray,
    ) -> Tuple[bool, float, Dict[str, float]]:
        if not candidate_priority_evs:
            return True, 0.0, {}

        # Note: Even if len(candidate_priority_evs) > M, Layer 1's Active Priority Queue (Q_k^act)
        # handles charger slot contention via Safe Release and Guarded FIFO Rotation!
        # Layer 0 checks whether the top min(|S|, M) vehicles are power/energy feasible in the QP.
        active_slice = candidate_priority_evs[: self.num_chargers]
        max_dep = max(v.departure_time_idx for v in active_slice)
        check_horizon = max(4, min(len(tou_prices), max_dep - current_time_idx))

        opt_inputs = []
        for v in active_slice:
            rem_steps = max(1, v.departure_time_idx - current_time_idx)
            mission_rem = max(0.0, v.mission_sufficient_kwh - v.current_energy_kwh)
            max_possible = v.max_charging_power_kw * rem_steps * self.dt
            if mission_rem - max_possible > self.shortfall_tol_kwh:
                return False, 0.0, {v.id: mission_rem - max_possible}
            opt_inputs.append(
                {
                    "id": v.id,
                    "energy_needed": min(v.energy_needed, max_possible),
                    "max_power": v.max_charging_power_kw,
                    "dep_time_idx": v.departure_time_idx,
                    "is_priority": True,
                }
            )

        res = self.optimizer.solve(
            current_time_idx=current_time_idx,
            horizon=check_horizon,
            active_evs=opt_inputs,
            tou_prices=tou_prices,
            pv_generation=pv_forecast,
            bess_generation=bess_forecast,
            enforce_priority_min=True,
        )

        max_sf = max(res.predicted_shortfalls.values()) if res.predicted_shortfalls else 0.0
        is_feas = (res.status in ["optimal", "optimal_inaccurate"]) and (max_sf <= self.shortfall_tol_kwh)
        return is_feas, res.solve_time_ms, res.predicted_shortfalls

    def select_priority_set(
        self,
        present_vehicles: List[Vehicle],
        current_time_idx: int,
        tou_prices: np.ndarray,
        pv_forecast: np.ndarray,
        bess_forecast: np.ndarray,
    ) -> Layer0Result:
        atc_scores = self.compute_atc_scores(present_vehicles, current_time_idx)

        urgent_candidates = [
            v for v in present_vehicles
            if v.energy_needed > 0.1
            and v.departure_time_idx > current_time_idx
            and not v.safe_released
            and (v.day_ahead_priority or v.trip_criticality_w >= 2.0)
        ]

        urgent_candidates.sort(key=lambda v: atc_scores.get(v.id, 0.0), reverse=True)
        n_cand = len(urgent_candidates)
        if n_cand == 0:
            for v in present_vehicles:
                v.is_priority = False
            return Layer0Result(set(), atc_scores, [], 0, 0.0, {})

        solves_used = 0
        total_ms = 0.0
        last_shortfalls = {}

        feas_all, ms, sf = self._check_subset_feasibility(
            urgent_candidates, current_time_idx, tou_prices, pv_forecast, bess_forecast
        )
        solves_used += 1
        total_ms += ms
        if feas_all:
            pri_ids = {v.id for v in urgent_candidates}
            for v in present_vehicles:
                v.is_priority = v.id in pri_ids
            return Layer0Result(pri_ids, atc_scores, [], solves_used, total_ms, sf)

        last_shortfalls.update(sf)

        lo = 1
        hi = n_cand
        best_m = n_cand

        while lo <= hi:
            mid = (lo + hi) // 2
            subset = urgent_candidates[:mid]
            feas, ms, sf = self._check_subset_feasibility(
                subset, current_time_idx, tou_prices, pv_forecast, bess_forecast
            )
            solves_used += 1
            total_ms += ms
            if feas:
                best_m = mid
                lo = mid + 1
            else:
                last_shortfalls.update(sf)
                hi = mid - 1

        selected = urgent_candidates[:best_m]
        escalated = [v.id for v in urgent_candidates[best_m:]]
        pri_ids = {v.id for v in selected}

        for v in present_vehicles:
            v.is_priority = v.id in pri_ids

        return Layer0Result(
            priority_ids=pri_ids,
            atc_scores=atc_scores,
            escalated_ids=escalated,
            qp_solves_used=solves_used,
            solve_time_ms_total=total_ms,
            predicted_shortfalls=last_shortfalls,
        )
'''
Path("src/aqps/priority_selection.py").write_text(PRIORITY_SELECTION_CODE, encoding="utf-8")
print("Updated: src/aqps/priority_selection.py")

# =============================================================================
# 5. Update src/aqps/baselines.py (Realistic Non-Preemptive EVSE Lock-in for LLF/EDF)
# =============================================================================
BASELINES_CODE = '''"""
WP5 — Competitive Baselines:
1. LLFScheduler      : Least Laxity First (non-preemptive EVSE occupancy, laxity-ordered power allocation).
2. EDFScheduler      : Earliest Deadline First (non-preemptive EVSE occupancy, deadline-ordered power allocation).
3. MPCOnlyScheduler  : Layer 2 QP alone with FIFO admission, no preemption, no priority lower bound.
4. OfflineOptimal    : Full-day convex QP with perfect foresight of arrivals, PV, and injections.
"""

import time
import warnings
import cvxpy as cp
import numpy as np
from typing import List, Dict
from .data_structures import Vehicle
from .tou_optimization import TOUOptimizer


class LLFScheduler:
    def __init__(self, num_chargers: int, grid_limit: float, dt_hours: float = 0.25):
        self.num_chargers = num_chargers
        self.grid_limit = grid_limit
        self.dt = dt_hours
        self.active_vehicles: List[Vehicle] = []
        self.connected_ids: set = set()

    def step(self, time_idx: int, arriving: List[Vehicle], departing_ids: List[str], current_cap: float = None) -> Dict:
        t0 = time.perf_counter()
        cap = self.grid_limit if current_cap is None else min(self.grid_limit, current_cap)
        for did in departing_ids:
            self.connected_ids.discard(did)

        self.active_vehicles = [
            v for v in self.active_vehicles
            if v.id not in departing_ids and v.departure_time_idx > time_idx and v.energy_needed > 0.05
        ]
        for v in list(self.connected_ids):
            if not any(ev.id == v and ev.energy_needed > 0.05 for ev in self.active_vehicles):
                self.connected_ids.discard(v)

        self.active_vehicles.extend(arriving)

        for v in self.active_vehicles:
            time_rem = (v.departure_time_idx - time_idx) * self.dt
            if time_rem <= 0:
                v.laxity = -999.0
            else:
                v.laxity = time_rem - (v.energy_needed / max(v.max_charging_power_kw, 0.01))

        # Non-preemptive EVSE occupancy: connected EVs hold their EVSE until finished or departed
        self.active_vehicles.sort(key=lambda x: (x.id not in self.connected_ids, x.laxity, x.arrival_time_idx))
        candidates = self.active_vehicles[: self.num_chargers]
        self.connected_ids = {v.id for v in candidates}

        candidates.sort(key=lambda x: (x.laxity, x.arrival_time_idx))
        commands = {}
        remaining_cap = cap
        for v in candidates:
            if remaining_cap <= 0 or v.energy_needed <= 0.01:
                commands[v.id] = 0.0
                continue
            p_cmd = min(v.max_charging_power_kw, remaining_cap, v.energy_needed / self.dt)
            commands[v.id] = p_cmd
            remaining_cap -= p_cmd
            v.energy_delivered += p_cmd * self.dt
            v.energy_needed = max(0.0, v.target_energy_kwh - v.energy_delivered)

        ms = (time.perf_counter() - t0) * 1000.0
        return {"commands": commands, "gross_ev_power": sum(commands.values()), "solve_time_ms": ms, "l2_solves": 0}


class EDFScheduler:
    def __init__(self, num_chargers: int, grid_limit: float, dt_hours: float = 0.25):
        self.num_chargers = num_chargers
        self.grid_limit = grid_limit
        self.dt = dt_hours
        self.active_vehicles: List[Vehicle] = []
        self.connected_ids: set = set()

    def step(self, time_idx: int, arriving: List[Vehicle], departing_ids: List[str], current_cap: float = None) -> Dict:
        t0 = time.perf_counter()
        cap = self.grid_limit if current_cap is None else min(self.grid_limit, current_cap)
        for did in departing_ids:
            self.connected_ids.discard(did)
        self.active_vehicles = [
            v for v in self.active_vehicles
            if v.id not in departing_ids and v.departure_time_idx > time_idx and v.energy_needed > 0.05
        ]
        for v in list(self.connected_ids):
            if not any(ev.id == v and ev.energy_needed > 0.05 for ev in self.active_vehicles):
                self.connected_ids.discard(v)
        self.active_vehicles.extend(arriving)

        self.active_vehicles.sort(key=lambda v: (v.id not in self.connected_ids, v.departure_time_idx, v.arrival_time_idx))
        candidates = self.active_vehicles[: self.num_chargers]
        self.connected_ids = {v.id for v in candidates}

        candidates.sort(key=lambda v: (v.departure_time_idx, v.arrival_time_idx))
        commands = {}
        remaining_cap = cap
        for v in candidates:
            if remaining_cap <= 0 or v.energy_needed <= 0.01:
                commands[v.id] = 0.0
                continue
            p_cmd = min(v.max_charging_power_kw, remaining_cap, v.energy_needed / self.dt)
            commands[v.id] = p_cmd
            remaining_cap -= p_cmd
            v.energy_delivered += p_cmd * self.dt
            v.energy_needed = max(0.0, v.target_energy_kwh - v.energy_delivered)

        ms = (time.perf_counter() - t0) * 1000.0
        return {"commands": commands, "gross_ev_power": sum(commands.values()), "solve_time_ms": ms, "l2_solves": 0}


class MPCOnlyScheduler:
    def __init__(self, config: Dict):
        self.num_chargers = int(config["station"]["num_chargers"])
        self.grid_limit = float(config["grid"]["transformer_limit_kw"])
        self.dt = float(config["simulation"].get("dt_hours", 0.25))
        self.horizon = int(config["simulation"].get("planning_horizon_steps", 24))
        self.optimizer = TOUOptimizer(transformer_cap=self.grid_limit, dt_hours=self.dt, alpha2=15.0)
        self.active_vehicles: Dict[str, Vehicle] = {}
        self.connected_ids: set = set()

    def step(
        self,
        time_idx: int,
        arriving: List[Vehicle],
        departing_ids: List[str],
        tou_forecast: np.ndarray,
        pv_forecast: np.ndarray,
        bess_forecast: np.ndarray = None,
    ) -> Dict:
        for did in departing_ids:
            self.active_vehicles.pop(did, None)
            self.connected_ids.discard(did)
        for v in arriving:
            self.active_vehicles[v.id] = v

        for vid, v in list(self.active_vehicles.items()):
            if v.energy_needed <= 0.05 or v.departure_time_idx <= time_idx:
                self.connected_ids.discard(vid)

        pool = [v for v in self.active_vehicles.values() if v.energy_needed > 0.05 and v.departure_time_idx > time_idx]
        pool.sort(key=lambda v: (v.id not in self.connected_ids, v.arrival_time_idx))
        admitted = pool[: self.num_chargers]
        self.connected_ids = {v.id for v in admitted}

        opt_evs = [
            {
                "id": v.id,
                "energy_needed": v.energy_needed,
                "max_power": v.max_charging_power_kw,
                "dep_time_idx": v.departure_time_idx,
                "is_priority": False,
            }
            for v in admitted
        ]
        res = self.optimizer.solve(
            current_time_idx=time_idx,
            horizon=self.horizon,
            active_evs=opt_evs,
            tou_prices=tou_forecast,
            pv_generation=pv_forecast,
            bess_generation=bess_forecast,
            enforce_priority_min=False,
        )
        commands = {}
        for v in admitted:
            prof = res.schedule.get(v.id, [0.0])
            p_cmd = min(float(prof[0]) if prof else 0.0, v.energy_needed / self.dt, v.max_charging_power_kw)
            commands[v.id] = p_cmd
            v.energy_delivered += p_cmd * self.dt
            v.energy_needed = max(0.0, v.target_energy_kwh - v.energy_delivered)

        return {
            "commands": commands,
            "gross_ev_power": sum(commands.values()),
            "solve_time_ms": res.solve_time_ms,
            "l2_solves": 1 if admitted else 0,
        }


class OfflineOptimalScheduler:
    def __init__(self, num_chargers: int, grid_limit: float, dt_hours: float = 0.25):
        self.num_chargers = num_chargers
        self.grid_limit = grid_limit
        self.dt = dt_hours

    def solve_full_day(
        self,
        vehicles: List[Vehicle],
        tou_prices: np.ndarray,
        pv_actual: np.ndarray,
        bess_actual: np.ndarray = None,
    ) -> Dict[str, np.ndarray]:
        T = len(tou_prices)
        N = len(vehicles)
        if bess_actual is None:
            bess_actual = np.zeros(T)

        x = cp.Variable((N, T), nonneg=True)
        constraints = []

        P_agg = cp.sum(x, axis=0)
        constraints.append(P_agg <= min(self.grid_limit, self.num_chargers * 22.0))

        weights = []
        targets = []
        for i, v in enumerate(vehicles):
            arr = max(0, min(T - 1, v.arrival_time_idx))
            dep = max(arr + 1, min(T, v.departure_time_idx))
            if arr > 0:
                constraints.append(x[i, :arr] == 0.0)
            if dep < T:
                constraints.append(x[i, dep:] == 0.0)
            constraints.append(x[i, arr:dep] <= v.max_charging_power_kw)
            constraints.append(cp.sum(x[i, arr:dep]) * self.dt <= v.target_energy_kwh)
            targets.append(v.target_energy_kwh)
            weights.append(10.0 if v.is_priority else 0.5)

        delivered = cp.sum(x, axis=1) * self.dt
        targets_arr = np.array(targets, dtype=float)
        weights_arr = np.array(weights, dtype=float)

        H_EC = cp.sum(cp.multiply(tou_prices, P_agg - (pv_actual + bess_actual))) * self.dt
        H_NC = cp.sum(cp.multiply(weights_arr, cp.square(delivered - targets_arr)))
        H_QC = cp.sum_squares(P_agg)

        prob = cp.Problem(cp.Minimize(1.0 * H_EC + 150.0 * H_NC + 0.005 * H_QC), constraints)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for solver in [cp.MOSEK, cp.CLARABEL, cp.OSQP]:
                if solver in cp.installed_solvers():
                    try:
                        prob.solve(solver=solver, verbose=False)
                        if x.value is not None and np.all(np.isfinite(x.value)):
                            break
                    except Exception:
                        continue

        x_val = (
            np.clip(np.asarray(x.value, dtype=float), 0.0, None)
            if (x.value is not None and np.all(np.isfinite(x.value)))
            else np.zeros((N, T))
        )
        schedules = {}
        for i, v in enumerate(vehicles):
            schedules[v.id] = x_val[i, :]
            v.energy_delivered = float(np.sum(x_val[i, :]) * self.dt)
            v.energy_needed = max(0.0, v.target_energy_kwh - v.energy_delivered)
        return schedules
'''
Path("src/aqps/baselines.py").write_text(BASELINES_CODE, encoding="utf-8")
print("Updated: src/aqps/baselines.py")

# =============================================================================
# 6. Update data_generator.py & Regenerate Scenarios
# =============================================================================
dg_path = Path("data_generator.py")
dg_code = dg_path.read_text(encoding="utf-8")
dg_code = dg_code.replace(
    'next_trip_kwh = round(max(4.0, mission_total - return_kwh - reserve_kwh), 2)\n\n        # Re-align target_energy_kwh so E_i^req = next_trip + return + reserve - eod_soc holds exactly\n        req_e = round(max(5.0, next_trip_kwh + return_kwh + reserve_kwh - eod_soc_kwh), 2)',
    'next_trip_kwh = round(max(4.0, 0.76 * req_e + eod_soc_kwh - return_kwh - reserve_kwh), 2)\n        req_e = round(max(8.0, req_e), 2)'
)
dg_path.write_text(dg_code, encoding="utf-8")
from data_generator import generate_scenarios
generate_scenarios()

# =============================================================================
# 7. Update run_all_experiments.py & generate_all_figures.py (No AQPS-v1)
# =============================================================================
run_path = Path("run_all_experiments.py")
run_code = run_path.read_text(encoding="utf-8")
run_code = run_code.replace('    "AQPS-v1",\n', "")
run_code = run_code.replace('"AQPS-v1", ', '"AQPS-static", ')
run_code = run_code.replace(', "AQPS-v1"', ', "AQPS-static"')
run_code = run_code.replace('"AQPS-static", "AQPS-static"', '"AQPS-static"')
run_path.write_text(run_code, encoding="utf-8")
print("Updated: run_all_experiments.py (removed AQPS-v1)")

fig_path = Path("generate_all_figures.py")
fig_code = fig_path.read_text(encoding="utf-8")
fig_code = fig_code.replace('"AQPS-v1", ', '"AQPS-static", ')
fig_code = fig_code.replace('    "AQPS-v1": "#CCBB44",\n', "")
fig_code = fig_code.replace(
    "Layer 1: Queue & Safe Release\\n• Laxity NP Preemption\\n• 3-Part Safe Release (kWh)\\n• Escalation Trigger",
    "Layer 1: Multi-Queue ($Q_k^{act}, Q_k^p, Q_k^n$)\\n• Laxity NP Preemption\\n• Safe Release & $Q_k^{act}$ FIFO Rotation",
)
fig_path.write_text(fig_code, encoding="utf-8")
print("Updated: generate_all_figures.py (removed AQPS-v1 & added Q_k^act to Fig. 2)")