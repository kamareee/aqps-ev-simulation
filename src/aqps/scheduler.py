"""
Unified AQPS Scheduler supporting all 4 AQPS variants (WP3, WP4, WP5):
- AQPS-v1     : Submitted method (operator binary flags, FIFO admission, every-step MPC)
- AQPS-static : Day-ahead Layer 0 priority, never recalculated intra-day, every-step MPC
- AQPS-RT     : Real-time Layer 0 + Layer 1 safe release, every-step MPC
- AQPS-RT-ET  : Full proposed framework (Real-time Layer 0 + Layer 1 safe release +
                Event-Triggered MPC + Operator Impact-Preview Loop)
"""

from typing import List, Dict, Optional, Any
import numpy as np
from .data_structures import Vehicle
from .queue_manager import QueueManager
from .tou_optimization import TOUOptimizer, OptimizationResult
from .priority_selection import PrioritySelectorLayer0
from .impact_preview import OperatorImpactPreview


class AQPSScheduler:
    def __init__(self, config: Dict[str, Any], mode: str = "AQPS-RT-ET"):
        """
        mode in {'AQPS-v1', 'AQPS-static', 'AQPS-RT', 'AQPS-RT-ET'}
        """
        self.config = config
        self.mode = mode
        self.num_chargers = int(config["station"]["num_chargers"])
        self.grid_limit = float(config["grid"]["transformer_limit_kw"])
        self.dt = float(config["simulation"].get("dt_hours", 0.25))
        self.horizon = int(config["simulation"].get("planning_horizon_steps", 24))

        # Tunable parameters for sensitivity analysis (E5, E6)
        opt_cfg = config.get("optimizer", {})
        self.alpha1 = float(opt_cfg.get("alpha1", 1.0))
        self.alpha2 = float(opt_cfg.get("alpha2", 100.0))
        self.alpha3 = float(opt_cfg.get("alpha3", 0.005))
        self.k_atc = float(opt_cfg.get("k_atc", 1.5))
        self.reserve_override_kwh = opt_cfg.get("reserve_override_kwh", None)

        et_cfg = config.get("event_trigger", {})
        self.t_max_steps = int(et_cfg.get("t_max_steps", 4))         # Default 4 steps = 60 min (or 15 min at 5-min step)
        self.pv_dev_threshold_kw = float(et_cfg.get("pv_dev_threshold_kw", 3.5))
        self.preview_cost_threshold = float(et_cfg.get("preview_cost_threshold", 5.0))

        self.queue_manager = QueueManager(self.num_chargers, dt_hours=self.dt)
        self.optimizer = TOUOptimizer(
            transformer_cap=self.grid_limit,
            dt_hours=self.dt,
            alpha1=self.alpha1,
            alpha2=self.alpha2,
            alpha3=self.alpha3,
        )
        self.layer0 = PrioritySelectorLayer0(
            optimizer=self.optimizer,
            num_chargers=self.num_chargers,
            dt_hours=self.dt,
            k_atc=self.k_atc,
        )
        self.impact_preview = OperatorImpactPreview(
            optimizer=self.optimizer,
            cost_threshold_dollars=self.preview_cost_threshold,
            dt_hours=self.dt,
        )

        # Event-Triggered MPC plan cache
        self.cached_plan: Dict[str, List[float]] = {}
        self.cached_step_offset: int = 0
        self.last_solve_time_idx: int = -999
        self.last_pv_forecast_at_solve: Optional[np.ndarray] = None
        self.last_horizon_cost: float = 0.0
        self.static_initialized: bool = False

        # Append-only structured logs
        self.solve_log: List[Dict[str, Any]] = []
        self.event_log: List[Dict[str, Any]] = []
        self.preemption_count: int = 0
        self.safe_release_count: int = 0
        self.escalation_count: int = 0
        self.extra_operator_cost: float = 0.0

    def step(
        self,
        time_idx: int,
        arriving_vehicles: List[Vehicle],
        departing_vehicle_ids: List[str],
        tou_forecast: np.ndarray,
        pv_forecast: np.ndarray,
        bess_forecast: Optional[np.ndarray] = None,
        pv_actual_now: Optional[float] = None,
        operator_injection: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Executes one control step Δt of the AQPS framework.
        """
        if bess_forecast is None:
            bess_forecast = np.zeros_like(pv_forecast)

        current_pv = float(pv_actual_now if pv_actual_now is not None else (pv_forecast[0] if len(pv_forecast) > 0 else 0.0))
        current_bess = float(bess_forecast[0] if len(bess_forecast) > 0 else 0.0)

        # 1. Process departures and arrivals
        self.queue_manager.process_departures(departing_vehicle_ids)
        for v in arriving_vehicles:
            v.day_ahead_priority = bool(v.is_priority)
            if self.mode != "AQPS-v1":
                v.target_energy_kwh = self.layer0.compute_session_requirement(v, self.reserve_override_kwh)
                v.energy_needed = max(0.0, v.target_energy_kwh - v.energy_delivered)
        self.queue_manager.process_arrivals(arriving_vehicles, time_idx)

        present_evs = [
            v for v in self.queue_manager.all_vehicles.values()
            if v.energy_needed > 0.05 and v.departure_time_idx > time_idx
        ]

        # 2. Event Detection (Table in §2.3)
        reasons = []
        if operator_injection is not None:
            reasons.append(f"operator_injection:{operator_injection.get('type', 'update')}")
        if len(arriving_vehicles) > 0 or len(departing_vehicle_ids) > 0:
            reasons.append("arrival_or_departure")

        # Check PV/BESS deviation against cached forecast
        pv_dev = 0.0
        if self.last_pv_forecast_at_solve is not None and self.cached_step_offset < len(self.last_pv_forecast_at_solve):
            expected_pv = float(self.last_pv_forecast_at_solve[self.cached_step_offset])
            pv_dev = abs(current_pv - expected_pv)
            if pv_dev > self.pv_dev_threshold_kw:
                reasons.append(f"pv_deviation:{pv_dev:.2f}kW")

        # Check predicted priority shortfall from cached plan
        for v in present_evs:
            if v.is_priority:
                rem_steps = max(0, v.departure_time_idx - time_idx)
                plan_rem = self.cached_plan.get(v.id, [])[self.cached_step_offset : self.cached_step_offset + rem_steps]
                planned_deliv = sum(plan_rem) * self.dt
                if min(v.energy_needed, rem_steps * v.max_charging_power_kw * self.dt) - planned_deliv > 0.5:
                    reasons.append(f"predicted_shortfall:{v.id}")
                    break

        # Periodic trigger T_max
        if (time_idx - self.last_solve_time_idx) >= self.t_max_steps:
            reasons.append("periodic_Tmax")

        event_triggered = (self.mode != "AQPS-RT-ET") or (len(reasons) > 0) or (not self.cached_plan)

        # 3. Layer 0: Priority Selection
        l0_solves = 0
        l0_ms = 0.0
        escalated_l0 = []

        if self.mode == "AQPS-v1":
            # Operator binary flags fixed as given
            for v in present_evs:
                v.is_priority = v.day_ahead_priority
        elif self.mode == "AQPS-static":
            # Score arriving vehicles once using day-ahead Layer 0, never recalculate for existing
            if arriving_vehicles or not self.static_initialized:
                self.layer0.compute_atc_scores(present_evs, time_idx)
                for v in arriving_vehicles:
                    v.is_priority = bool(v.day_ahead_priority or v.atc_score >= 0.35)
                self.static_initialized = True
        elif self.mode in ["AQPS-RT", "AQPS-RT-ET"]:
            need_l0 = (self.mode == "AQPS-RT") or any(
                r.startswith(("operator_injection", "arrival_or_departure", "predicted_shortfall", "pv_deviation"))
                for r in reasons
            )
            if need_l0 and present_evs:
                l0_res = self.layer0.select_priority_set(
                    present_vehicles=present_evs,
                    current_time_idx=time_idx,
                    tou_prices=tou_forecast,
                    pv_forecast=pv_forecast,
                    bess_forecast=bess_forecast,
                )
                l0_solves = l0_res.qp_solves_used
                l0_ms = l0_res.solve_time_ms_total
                escalated_l0 = l0_res.escalated_ids

        # 4. Layer 1: Queue Management & Safe Release
        enable_sr = self.mode in ["AQPS-RT", "AQPS-RT-ET"]
        use_atc = self.mode != "AQPS-v1"

        if event_triggered or self.mode != "AQPS-RT-ET":
            active_vehicles, l1_events = self.queue_manager.get_active_assignment(
                current_time=time_idx,
                current_system_capacity=self.grid_limit,
                enable_safe_release=enable_sr,
                use_atc_order=use_atc,
                reserve_override_kwh=self.reserve_override_kwh,
            )
            self.preemption_count += len(l1_events["preempted_np"])
            self.safe_release_count += len(l1_events["safe_released_p"])
            total_escalated = sorted(list(set(escalated_l0 + l1_events["escalated_p"])))
            self.escalation_count += len(total_escalated)
        else:
            active_vehicles = [
                self.queue_manager.all_vehicles[vid]
                for vid in self.queue_manager.connected_ids
                if vid in self.queue_manager.all_vehicles
            ]
            l1_events = {"preempted_np": [], "safe_released_p": [], "escalated_p": []}
            total_escalated = []

        # 5. Operator Impact-Preview Loop (WP4, only in AQPS-RT-ET when priority affected or injection/PV dev)
        preview_ms = 0.0
        preview_solves = 0
        if self.mode == "AQPS-RT-ET" and event_triggered and active_vehicles:
            trigger_preview = (
                operator_injection is not None
                or len(total_escalated) > 0
                or len(l1_events["safe_released_p"]) > 0
                or pv_dev > self.pv_dev_threshold_kw
            )
            if trigger_preview:
                op_choice = "recommended"
                if operator_injection is not None:
                    op_choice = operator_injection.get("operator_choice", "recommended")
                prev_rec = self.impact_preview.evaluate_preview(
                    time_idx=time_idx,
                    trigger_reason=",".join(reasons) if reasons else "manual",
                    admitted_vehicles=active_vehicles,
                    escalated_ids=total_escalated,
                    horizon=self.horizon,
                    tou_prices=tou_forecast,
                    pv_forecast=pv_forecast,
                    bess_forecast=bess_forecast,
                    baseline_cost=self.last_horizon_cost,
                    operator_script_choice=op_choice,
                )
                preview_ms = prev_rec.total_preview_solve_ms
                preview_solves = len(prev_rec.options)
                self.extra_operator_cost += prev_rec.extra_cost_accepted

        # 6. Layer 2: Event-Triggered or Every-Step MPC Solve
        power_cmds: Dict[str, float] = {}
        l2_ms = 0.0
        l2_solves = 0

        if event_triggered:
            opt_input_evs = [
                {
                    "id": v.id,
                    "energy_needed": v.energy_needed,
                    "max_power": v.max_charging_power_kw,
                    "dep_time_idx": v.departure_time_idx,
                    "is_priority": v.is_priority,
                }
                for v in active_vehicles
            ]
            res: OptimizationResult = self.optimizer.solve(
                current_time_idx=time_idx,
                horizon=self.horizon,
                active_evs=opt_input_evs,
                tou_prices=tou_forecast,
                pv_generation=pv_forecast,
                bess_generation=bess_forecast,
                enforce_priority_min=True,
            )
            l2_ms = res.solve_time_ms
            l2_solves = 1 if active_vehicles else 0
            self.cached_plan = res.schedule
            self.cached_step_offset = 0
            self.last_solve_time_idx = time_idx
            self.last_pv_forecast_at_solve = pv_forecast.copy()
            self.last_horizon_cost = res.total_cost

            for v in active_vehicles:
                prof = self.cached_plan.get(v.id, [0.0])
                p_cmd = float(prof[0]) if len(prof) > 0 else 0.0
                p_cmd = min(p_cmd, v.energy_needed / self.dt, v.max_charging_power_kw)
                power_cmds[v.id] = p_cmd

            self.event_log.append(
                {
                    "time_idx": time_idx,
                    "reasons": reasons,
                    "active_evs": len(active_vehicles),
                    "preempted_np": l1_events["preempted_np"],
                    "safe_released_p": l1_events["safe_released_p"],
                    "escalated_p": total_escalated,
                    "l0_solves": l0_solves,
                    "l2_solves": l2_solves,
                    "preview_solves": preview_solves,
                    "total_solve_ms": round(l0_ms + l2_ms + preview_ms, 3),
                }
            )
        else:
            # Apply next step of stored plan without re-solving QP
            self.cached_step_offset += 1
            for v in active_vehicles:
                prof = self.cached_plan.get(v.id, [])
                if self.cached_step_offset < len(prof):
                    p_cmd = float(prof[self.cached_step_offset])
                else:
                    p_cmd = 0.0
                p_cmd = min(p_cmd, v.energy_needed / self.dt, v.max_charging_power_kw)
                power_cmds[v.id] = p_cmd

        # Enforce gross site limit safety clamp
        tot_p = sum(power_cmds.values())
        if tot_p > self.grid_limit + 1e-3:
            scale = self.grid_limit / tot_p
            for vid in power_cmds:
                power_cmds[vid] *= scale

        # Update vehicle energy states
        for vid, p_cmd in power_cmds.items():
            self.queue_manager.update_vehicle_energy(vid, p_cmd * self.dt)

        if l2_solves > 0 or l0_solves > 0:
            self.solve_log.append(
                {
                    "time_idx": time_idx,
                    "l0_solves": l0_solves,
                    "l2_solves": l2_solves,
                    "preview_solves": preview_solves,
                    "l2_solve_ms": l2_ms,
                    "total_step_ms": l0_ms + l2_ms + preview_ms,
                }
            )

        gross_ev_kw = sum(power_cmds.values())
        net_grid_kw = max(0.0, gross_ev_kw - current_pv - current_bess)

        return {
            "commands": power_cmds,
            "gross_ev_power": gross_ev_kw,
            "grid_power": net_grid_kw,
            "active_count": len(active_vehicles),
            "queue_size": self.queue_manager.get_queue_size(),
            "event_triggered": event_triggered,
            "l2_solves": l2_solves,
            "total_solves": l0_solves + l2_solves + preview_solves,
            "solve_time_ms": l2_ms,
            "total_step_ms": l0_ms + l2_ms + preview_ms,
        }
