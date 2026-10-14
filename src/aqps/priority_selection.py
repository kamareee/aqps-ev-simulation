"""
WP3 — Layer 0: Real-Time Schedule-Derived Priority Selection with Optimizer Feasibility Check.

Implements:
1. Session requirement E_i^req = next_trip + return + reserve - eod_soc (mission-sufficient kWh).
2. Apparent Tardiness Cost (ATC) urgency score (Vepsalainen & Morton, Management Science 1987;
   lookahead tuning Valente, IJPE 2007):
       pi_i(t) = (w_i / p_i) * exp(-max(0, s_i(t)) / (k_atc * p_bar))
3. Bisection feasibility check using the Layer 2 QP (~log2(n) solves per Layer-0 event).
4. Escalation list of urgent vehicles that cannot fit without shortfall.
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
        k_atc: float = 1.5,
        shortfall_tol_kwh: float = 0.25,
        urgency_min_threshold: float = 0.05,
    ):
        self.optimizer = optimizer
        self.num_chargers = num_chargers
        self.dt = dt_hours
        self.k_atc = k_atc
        self.shortfall_tol_kwh = shortfall_tol_kwh
        self.urgency_min_threshold = urgency_min_threshold

    @staticmethod
    def compute_session_requirement(v: Vehicle, reserve_override_kwh: float = None) -> float:
        """
        Day-ahead / intra-day session requirement E_i^req (kWh):
        energy needed until next charging opportunity = next_trip + return + reserve - eod_soc.
        """
        r = reserve_override_kwh if reserve_override_kwh is not None else v.reserve_kwh
        mission_total = v.next_trip_kwh + v.return_kwh + r
        req = max(0.0, min(v.battery_cap_kwh, mission_total) - v.eod_soc_kwh)
        return round(req, 2)

    def compute_atc_scores(
        self, vehicles: List[Vehicle], current_time_idx: int
    ) -> Dict[str, float]:
        """
        Computes Apparent Tardiness Cost (ATC) urgency score pi_i(t) for each present vehicle:
            pi_i(t) = (w_i / p_i) * exp(-max(0, s_i(t)) / (k * p_bar))
        """
        active = [v for v in vehicles if v.energy_needed > 0.1 and v.departure_time_idx > current_time_idx]
        if not active:
            return {v.id: 0.0 for v in vehicles}

        p_dict = {}
        s_dict = {}
        for v in active:
            p_i = max(0.10, v.energy_needed / max(v.max_charging_power_kw, 1.0))
            rem_hours = max(0.0, (v.departure_time_idx - current_time_idx) * self.dt)
            s_i = rem_hours - p_i  # Slack in hours
            p_dict[v.id] = p_i
            s_dict[v.id] = s_i
            v.laxity = s_i

        p_bar = max(0.25, float(np.mean(list(p_dict.values()))))
        denom = max(0.10, self.k_atc * p_bar)

        scores = {}
        for v in vehicles:
            if v.id not in p_dict:
                scores[v.id] = 0.0
                v.atc_score = 0.0
                continue
            p_i = p_dict[v.id]
            s_i = s_dict[v.id]
            w_i = float(v.trip_criticality_w)
            # ATC formula (Vepsalainen & Morton, 1987)
            # If s_i < 0 (already tardy), add slight linear tie-breaker so more overdue ranks higher
            tardy_boost = 1.0 + max(0.0, -s_i) * 0.15
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
        """
        Tests whether `candidate_priority_evs` can all achieve zero predicted shortfall
        under the Layer 2 QP and EVSE count M.
        Returns (is_feasible, solve_time_ms, shortfalls_dict).
        """
        if not candidate_priority_evs:
            return True, 0.0, {}

        # Necessary condition 1: Cannot exceed total EVSE count M simultaneously
        if len(candidate_priority_evs) > self.num_chargers:
            # Check energy feasibility of top M and mark remainder as shortfall
            shortfalls = {v.id: v.energy_needed for v in candidate_priority_evs[self.num_chargers:]}
            return False, 0.0, shortfalls

        # Necessary condition 2: Individual physical rate check & aggregate energy check
        max_dep = max(v.departure_time_idx for v in candidate_priority_evs)
        check_horizon = max(4, min(len(tou_prices), max_dep - current_time_idx))

        opt_inputs = []
        for v in candidate_priority_evs:
            rem_steps = max(0, v.departure_time_idx - current_time_idx)
            max_possible = v.max_charging_power_kw * rem_steps * self.dt
            if v.energy_needed - max_possible > self.shortfall_tol_kwh:
                # Physically impossible even at max EVSE power
                return False, 0.0, {v.id: v.energy_needed - max_possible}
            opt_inputs.append(
                {
                    "id": v.id,
                    "energy_needed": v.energy_needed,
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
        """
        Ranks vehicles by ATC urgency score pi_i(t) and uses bisection over the Layer 2 QP
        to find the largest feasible priority set S.
        """
        atc_scores = self.compute_atc_scores(present_vehicles, current_time_idx)

        # Filter to vehicles that have urgent mission demand (either flagged day-ahead or high ATC)
        urgent_candidates = [
            v for v in present_vehicles
            if v.energy_needed > 0.1
            and v.departure_time_idx > current_time_idx
            and (v.day_ahead_priority or v.trip_criticality_w >= 2.0 or v.laxity <= 1.5)
        ]

        # Sort in descending order of ATC score pi_i(t)
        urgent_candidates.sort(key=lambda v: atc_scores.get(v.id, 0.0), reverse=True)

        n_cand = len(urgent_candidates)
        if n_cand == 0:
            for v in present_vehicles:
                v.is_priority = False
            return Layer0Result(set(), atc_scores, [], 0, 0.0, {})

        solves_used = 0
        total_ms = 0.0
        last_shortfalls = {}

        # First check if all urgent candidates fit (common case when not overloaded)
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

        # Bisection search over prefix length m in [0, min(n_cand, M)]
        lo = 0
        hi = min(n_cand, self.num_chargers)
        best_m = 0

        while lo <= hi:
            mid = (lo + hi) // 2
            if mid == 0:
                best_m = max(best_m, 0)
                lo = mid + 1
                continue
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
