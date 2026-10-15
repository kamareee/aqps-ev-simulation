"""
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
