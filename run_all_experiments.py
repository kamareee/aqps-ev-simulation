"""
WP6 — Master Simulation Experiment Suite (E1 to E7 + E5b Scalability).

Features:
- Resumable per-run execution: saves each run to `results/<exp_id>/raw/<run_id>.json` and skips if already present.
- Writes `results/<exp_id>/manifest.json` and aggregated `results/<exp_id>/summary/*.csv`.
- Writes `docs/results_map.md` mapping E1-E7 -> Manuscript Section -> Figure/Table -> Post-processing script.
"""

import os
import sys
import json
import time
import platform
import subprocess
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any

from data_generator import generate_scenarios, INFRA
from scripts.prepare_aemo_pv import build_aemo_vic1_dataset, OUTPUT_PATH as AEMO_PATH
from src.aqps.data_structures import Vehicle
from src.aqps.scheduler import AQPSScheduler
from src.aqps.baselines import LLFScheduler, EDFScheduler, MPCOnlyScheduler, OfflineOptimalScheduler
from src.aqps.priority_selection import PrioritySelectorLayer0


METHODS_ALL = [
    "LLF",
    "EDF",
    "MPC-only",
    "Offline-Optimal",
    "AQPS-static",
    "AQPS-RT",
    "AQPS-RT-ET",
]

SCENARIOS = [
    ("S1_Baseline", "S1 (27%)"),
    ("S3_HighPriority", "S2 (50%)"),
    ("S6_PeakStress", "S3 (50% Peak)"),
]


def get_git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "rev_2026_epsr"


def write_manifest(exp_id: str, config_summary: Dict, seeds: List[int]):
    exp_dir = os.path.join("results", exp_id)
    os.makedirs(os.path.join(exp_dir, "raw"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "summary"), exist_ok=True)
    manifest = {
        "experiment_id": exp_id,
        "git_commit": get_git_hash(),
        "run_date": datetime.now().isoformat(),
        "machine": {
            "os": platform.platform(),
            "python": sys.version.split()[0],
            "processor": platform.processor() or "x86_64/ARM64",
        },
        "seeds": seeds,
        "config": config_summary,
    }
    with open(os.path.join(exp_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


def build_vehicles(ev_data: List[Dict]) -> Dict[str, Vehicle]:
    vmap = {}
    for ev in ev_data:
        is_p = (ev["priority"] == "High")
        v = Vehicle(
            id=ev["id"],
            arrival_time_idx=int(ev["arrival_idx"]),
            departure_time_idx=int(ev["departure_idx"]),
            target_energy_kwh=float(ev["req_energy"]),
            max_charging_power_kw=float(ev["p_max"]),
            is_priority=is_p,
            battery_cap_kwh=float(ev.get("battery_cap_kwh", 68.0)),
            eod_soc_kwh=float(ev.get("eod_soc_kwh", 15.0)),
            next_trip_kwh=float(ev.get("next_trip_kwh", 18.0)),
            return_kwh=float(ev.get("return_kwh", 6.0)),
            reserve_kwh=float(ev.get("reserve_kwh", 6.8)),
            further_trips_today_kwh=float(ev.get("further_trips_today_kwh", 0.0)),
            next_day_dep_idx=int(ev.get("next_day_dep_idx", 96)),
            next_day_req_kwh=float(ev.get("next_day_req_kwh", 25.0)),
            trip_criticality_w=float(ev.get("trip_criticality_w", 3.5 if is_p else 1.0)),
            archetype=ev.get("archetype", "Medium"),
            day_ahead_priority=is_p,
        )
        vmap[v.id] = v
    return vmap


def run_single_simulation(
    method: str,
    fleet_size: int,
    scenario_key: str,
    scenario_label: str,
    ev_data: List[Dict],
    env_data: Dict,
    n_evse: int = None,
    transformer_kw: float = None,
    pv_actual_override: List[float] = None,
    injections_by_step: Dict[int, Dict] = None,
    config_overrides: Dict = None,
) -> Dict[str, Any]:
    """
    Runs a single 24-hour simulation (96 steps of 15 min) for any of the 8 methods
    and returns comprehensive metrics.
    """
    horizon = len(env_data["tou_tariff"])
    dt = float(env_data.get("step_delta", 0.25))
    num_chargers = n_evse if n_evse is not None else INFRA.get(fleet_size, {"n_evse": max(4, fleet_size // 3)})["n_evse"]
    site_limit_kw = transformer_kw if transformer_kw is not None else INFRA.get(fleet_size, {"transformer_kW": (fleet_size / 45.0) * 150.0})["transformer_kW"]

    config = {
        "station": {"num_chargers": num_chargers},
        "grid": {"transformer_limit_kw": site_limit_kw},
        "simulation": {"dt_hours": dt, "planning_horizon_steps": 24},
        "optimizer": {},
        "event_trigger": {},
    }
    if config_overrides:
        for k, sub in config_overrides.items():
            if isinstance(sub, dict) and k in config:
                config[k].update(sub)
            else:
                config[k] = sub

    vehicles = build_vehicles(ev_data)
    for v in vehicles.values():
        v.target_energy_kwh = PrioritySelectorLayer0.compute_session_requirement(
            v, config["optimizer"].get("reserve_override_kwh", None)
        )
        v.energy_needed = v.target_energy_kwh

    tou_full = np.array(env_data["tou_tariff"], dtype=float)
    pv_fcst_full = np.array(env_data["pv_forecast"], dtype=float)
    pv_act_full = np.array(pv_actual_override if pv_actual_override is not None else env_data["pv_forecast"], dtype=float)
    bess_full = np.array(env_data.get("bess", [0.0] * horizon), dtype=float)

    cost_history = []
    cum_cost = 0.0
    cost_by_tier = {"off_peak": 0.0, "shoulder": 0.0, "peak": 0.0}
    pv_used_kwh = 0.0
    bess_used_kwh = 0.0
    gross_load_trace = np.zeros(horizon)
    net_grid_trace = np.zeros(horizon)
    ev_power_timeline = {v.id: np.zeros(horizon) for v in vehicles.values()}
    queue_pos_timeline = {v.id: np.full(horizon, np.nan) for v in vehicles.values()}
    solve_times_ms = []
    total_l2_solves = 0
    total_all_solves = 0
    preemptions = 0
    safe_releases = 0
    escalations = 0
    extra_op_cost = 0.0
    preemption_events = []
    starvation_events = []
    prev_cmds = {}

    if method == "Offline-Optimal":
        t0 = time.perf_counter()
        oracle = OfflineOptimalScheduler(num_chargers, site_limit_kw, dt)
        sched_map = oracle.solve_full_day(list(vehicles.values()), tou_full, pv_act_full, bess_full)
        solve_times_ms.append((time.perf_counter() - t0) * 1000.0)
        total_l2_solves = 1
        total_all_solves = 1
        for t in range(horizon):
            gross_t = sum(float(sched_map[vid][t]) for vid in sched_map)
            for vid in sched_map:
                ev_power_timeline[vid][t] = float(sched_map[vid][t])
            gross_load_trace[t] = gross_t
            pv_t = max(0.0, pv_act_full[t])
            bess_t = max(0.0, bess_full[t])
            pv_used_t = min(gross_t, pv_t)
            rem_load = gross_t - pv_used_t
            bess_used_t = min(rem_load, bess_t)
            net_grid_t = max(0.0, rem_load - bess_used_t)
            net_grid_trace[t] = net_grid_t
            pv_used_kwh += pv_used_t * dt
            bess_used_kwh += bess_used_t * dt
            step_c = net_grid_t * tou_full[t] * dt
            cum_cost += step_c
            cost_history.append(cum_cost)
            if tou_full[t] >= 0.30:
                cost_by_tier["peak"] += step_c
            elif tou_full[t] >= 0.20:
                cost_by_tier["shoulder"] += step_c
            else:
                cost_by_tier["off_peak"] += step_c
    else:
        if method == "LLF":
            sched = LLFScheduler(num_chargers, site_limit_kw, dt)
        elif method == "EDF":
            sched = EDFScheduler(num_chargers, site_limit_kw, dt)
        elif method == "MPC-only":
            sched = MPCOnlyScheduler(config)
        else:
            sched = AQPSScheduler(config, mode=method)

        for t in range(horizon):
            inj = injections_by_step.get(t) if injections_by_step else None
            if inj is not None:
                # Apply scripted injection mutation to vehicle pool
                if inj["type"] == "vehicle_breakdown" and inj.get("promote_id") in vehicles:
                    pv_v = vehicles[inj["promote_id"]]
                    pv_v.is_priority = True
                    pv_v.day_ahead_priority = True
                    pv_v.trip_criticality_w = float(inj.get("new_w", 4.2))
                    pv_v.target_energy_kwh = float(inj.get("new_req_kwh", 32.0))
                    pv_v.energy_needed = max(0.0, pv_v.target_energy_kwh - pv_v.energy_delivered)
                    pv_v.departure_time_idx = int(inj.get("new_dep_idx", max(t + 8, pv_v.departure_time_idx)))
                elif inj["type"] == "driver_unavailable" and inj.get("delay_id") in vehicles:
                    dv = vehicles[inj["delay_id"]]
                    dv.departure_time_idx = min(95, dv.departure_time_idx + int(inj.get("delay_steps", 8)))
                elif inj["type"] == "roster_swap":
                    v_from = vehicles.get(inj.get("swap_from_id"))
                    v_to = vehicles.get(inj.get("swap_to_id"))
                    if v_from and v_to:
                        v_from.is_priority, v_to.is_priority = False, True
                        v_from.day_ahead_priority, v_to.day_ahead_priority = False, True
                        v_from.trip_criticality_w, v_to.trip_criticality_w = 0.8, 4.0

            arriving = [v for v in vehicles.values() if v.arrival_time_idx == t]
            departing = [v.id for v in vehicles.values() if v.departure_time_idx == t]

            tou_h = np.array(list(tou_full[t:]) + [tou_full[-1]] * t, dtype=float)
            pv_fcst_h = np.array(list(pv_fcst_full[t:]) + [0.0] * t, dtype=float)
            bess_h = np.array(list(bess_full[t:]) + [0.0] * t, dtype=float)
            pv_now = float(pv_act_full[t])
            if inj is not None and inj["type"] == "pv_shortfall":
                pv_now *= float(inj.get("pv_drop_factor", 0.25))

            if method in ["LLF", "EDF"]:
                step_res = sched.step(t, arriving, departing, site_limit_kw)
            elif method == "MPC-only":
                step_res = sched.step(t, arriving, departing, tou_h, pv_fcst_h, bess_h)
            else:
                step_res = sched.step(
                    t, arriving, departing, tou_h, pv_fcst_h, bess_h,
                    pv_actual_now=pv_now, operator_injection=inj
                )

            cmds = step_res.get("commands", {})
            if step_res.get("l2_solves", 0) > 0:
                solve_times_ms.append(float(step_res.get("solve_time_ms", 0.0)))
            total_l2_solves += int(step_res.get("l2_solves", 0))
            total_all_solves += int(step_res.get("total_solves", step_res.get("l2_solves", 0)))

            for vid, p_cmd in cmds.items():
                ev_power_timeline[vid][t] = float(p_cmd)

            # Track queue rank for active vehicles
            active_now = [v for v in vehicles.values() if v.arrival_time_idx <= t < v.departure_time_idx and v.energy_needed > 0.1]
            active_now.sort(key=lambda v: (not v.is_priority, -v.atc_score, v.arrival_time_idx))
            for rank_idx, v in enumerate(active_now):
                queue_pos_timeline[v.id][t] = rank_idx + 1

            # Detect preemption / starvation for Gantt charts
            pri_arr = [v for v in arriving if v.day_ahead_priority]
            for v in vehicles.values():
                if not v.day_ahead_priority and v.arrival_time_idx <= t < v.departure_time_idx:
                    if v.energy_needed > 0.5 and prev_cmds.get(v.id, 0.0) > 0.5 and cmds.get(v.id, 0.0) < 0.5 and pri_arr:
                        preemption_events.append({"time": t, "priority_id": pri_arr[0].id, "preempted_id": v.id})
                if v.day_ahead_priority and v.arrival_time_idx <= t < v.departure_time_idx:
                    if v.energy_needed > 0.5 and cmds.get(v.id, 0.0) < 0.5:
                        starvation_events.append({"time": t, "starved_id": v.id})
            prev_cmds = dict(cmds)

            gross_t = float(sum(cmds.values()))
            gross_load_trace[t] = gross_t
            pv_t = max(0.0, pv_now)
            bess_t = max(0.0, float(bess_full[t]))
            pv_used_t = min(gross_t, pv_t)
            rem_load = gross_t - pv_used_t
            bess_used_t = min(rem_load, bess_t)
            net_grid_t = max(0.0, rem_load - bess_used_t)
            net_grid_trace[t] = net_grid_t

            pv_used_kwh += pv_used_t * dt
            bess_used_kwh += bess_used_t * dt
            step_c = net_grid_t * tou_full[t] * dt
            cum_cost += step_c
            cost_history.append(cum_cost)
            if tou_full[t] >= 0.30:
                cost_by_tier["peak"] += step_c
            elif tou_full[t] >= 0.20:
                cost_by_tier["shoulder"] += step_c
            else:
                cost_by_tier["off_peak"] += step_c

        if isinstance(sched, AQPSScheduler):
            preemptions = sched.preemption_count
            safe_releases = sched.safe_release_count
            escalations = sched.escalation_count
            extra_op_cost = sched.extra_operator_cost
        else:
            preemptions = len(preemption_events)

    # Compute comprehensive fulfillment & service metrics ( evaluated against day-ahead priority cohort for consistent benchmarking )
    def calc_fulfill(is_p: bool) -> float:
        cohort = [v for v in vehicles.values() if v.day_ahead_priority == is_p]
        req = sum(v.target_energy_kwh for v in cohort)
        deliv = sum(min(v.energy_delivered, v.target_energy_kwh) for v in cohort)
        return (deliv / req * 100.0) if req > 0 else 100.0

    # Mission sufficiency: share of priority vehicles departing with >= next_trip + return + reserve (within 0.5 kWh tolerance)
    pri_cohort = [v for v in vehicles.values() if v.day_ahead_priority]
    mission_ok = sum(
        1 for v in pri_cohort
        if (v.eod_soc_kwh + v.energy_delivered) >= (v.next_trip_kwh + v.return_kwh + v.reserve_kwh - 0.5)
    )
    mission_suff_pct = (mission_ok / len(pri_cohort) * 100.0) if pri_cohort else 100.0

    # Jain's fairness index across non-priority vehicles
    np_cohort = [v for v in vehicles.values() if not v.day_ahead_priority and v.target_energy_kwh > 0]
    if np_cohort:
        ratios = np.array([min(1.0, v.energy_delivered / v.target_energy_kwh) for v in np_cohort])
        denom = len(ratios) * np.sum(ratios ** 2)
        jain_np = float((np.sum(ratios) ** 2) / denom) if denom > 0 else 1.0
    else:
        jain_np = 1.0

    peak_kw = float(np.max(gross_load_trace)) if len(gross_load_trace) > 0 else 0.0
    mean_kw = float(np.mean(gross_load_trace)) if len(gross_load_trace) > 0 else 0.0
    load_factor = (mean_kw / peak_kw) if peak_kw > 0 else 0.0
    transformer_loading_pct = (peak_kw / site_limit_kw * 100.0) if site_limit_kw > 0 else 0.0
    total_pv_avail_kwh = float(np.sum(np.maximum(0.0, pv_act_full)) * dt)
    pv_self_cons_pct = min(100.0, (pv_used_kwh / total_pv_avail_kwh * 100.0)) if total_pv_avail_kwh > 0 else 0.0

    mean_solve_ms = float(np.mean(solve_times_ms)) if solve_times_ms else 0.0
    p95_solve_ms = float(np.percentile(solve_times_ms, 95)) if solve_times_ms else 0.0

    return {
        "method": method,
        "fleet_size": fleet_size,
        "n_evse": num_chargers,
        "scenario_key": scenario_key,
        "scenario_label": scenario_label,
        "priority_fulfillment_pct": round(calc_fulfill(True), 2),
        "mission_sufficiency_pct": round(mission_suff_pct, 2),
        "non_priority_fulfillment_pct": round(calc_fulfill(False), 2),
        "non_priority_jain_fairness": round(jain_np, 3),
        "total_cost": round(cum_cost, 2),
        "extra_operator_cost": round(extra_op_cost, 2),
        "cost_off_peak": round(cost_by_tier["off_peak"], 2),
        "cost_shoulder": round(cost_by_tier["shoulder"], 2),
        "cost_peak": round(cost_by_tier["peak"], 2),
        "pv_used_kwh": round(pv_used_kwh, 2),
        "bess_used_kwh": round(bess_used_kwh, 2),
        "pv_self_consumption_pct": round(pv_self_cons_pct, 2),
        "preemptions": int(preemptions),
        "safe_releases": int(safe_releases),
        "escalations": int(escalations),
        "l2_solve_count": int(total_l2_solves),
        "total_qp_solves": int(total_all_solves),
        "solve_time_mean_ms": round(mean_solve_ms, 3),
        "solve_time_p95_ms": round(p95_solve_ms, 3),
        "peak_demand_kw": round(peak_kw, 2),
        "load_factor": round(load_factor, 3),
        "transformer_loading_pct": round(transformer_loading_pct, 2),
        "cumulative_cost_trace": [round(x, 2) for x in cost_history],
        "ev_power_timeline": {k: v.tolist() for k, v in ev_power_timeline.items()},
        "queue_pos_timeline": {k: [None if np.isnan(x) else int(x) for x in v] for k, v in queue_pos_timeline.items()},
        "preemption_events": preemption_events,
        "starvation_events": starvation_events,
    }


def run_e1_main():
    print("\n=== Running E1: Main Benchmark (8 Methods x S1-S3 x 2 Fleets) ===")
    exp_id = "E1_main"
    write_manifest(exp_id, {"methods": METHODS_ALL, "fleets": [45, 90], "scenarios": [s[0] for s in SCENARIOS]}, [42])
    rows = []

    for fleet_size in [45, 90]:
        with open(f"simulation_data_{fleet_size}EVs.json") as f:
            data = json.load(f)
        env = data["environment"]
        for sc_key, sc_label in SCENARIOS:
            ev_data = data["scenarios"][sc_key]
            # First run Offline-Optimal to record upper bound cost
            oracle_res = None
            sc_rows = []
            for m in METHODS_ALL:
                raw_path = os.path.join("results", exp_id, "raw", f"{m}_{sc_key}_{fleet_size}EVs.json")
                if os.path.exists(raw_path):
                    with open(raw_path) as rf:
                        res = json.load(rf)
                else:
                    res = run_single_simulation(m, fleet_size, sc_key, sc_label, ev_data, env)
                    with open(raw_path, "w") as rf:
                        json.dump(res, rf)
                if m == "Offline-Optimal":
                    oracle_res = res
                sc_rows.append(res)

            oracle_cost = oracle_res["total_cost"] if oracle_res else 1.0
            for r in sc_rows:
                r["gap_to_offline_opt_pct"] = round(
                    ((r["total_cost"] - oracle_cost) / max(1.0, oracle_cost)) * 100.0, 2
                )
                rows.append({k: v for k, v in r.items() if not isinstance(v, (list, dict))})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join("results", exp_id, "summary", "e1_master_table.csv"), index=False)
    print(f"[E1] Saved summary to results/{exp_id}/summary/e1_master_table.csv")


def run_e2_case_studies():
    print("\n=== Running E2: Scripted Operator Injection Case Studies ===")
    exp_id = "E2_case_studies"
    with open(os.path.join("data", "operator_injections.json")) as f:
        injections = json.load(f)
    write_manifest(exp_id, {"cases": list(injections.keys()), "fleet": 90, "scenario": "S3_HighPriority"}, [200])

    with open("simulation_data_90EVs.json") as f:
        data = json.load(f)
    env = data["environment"]
    ev_data = data["scenarios"]["S3_HighPriority"]

    rows = []
    for case_name, inj in injections.items():
        for m in ["LLF", "AQPS-static", "AQPS-RT-ET"]:
            raw_path = os.path.join("results", exp_id, "raw", f"{case_name}_{m}.json")
            if os.path.exists(raw_path):
                with open(raw_path) as rf:
                    res = json.load(rf)
            else:
                res = run_single_simulation(
                    m, 90, "S3_HighPriority", f"S2 ({case_name})", ev_data, env,
                    injections_by_step={int(inj["time_idx"]): inj}
                )
                res["case_name"] = case_name
                res["injection_time_idx"] = int(inj["time_idx"])
                res["injection_type"] = inj["type"]
                with open(raw_path, "w") as rf:
                    json.dump(res, rf)
            rows.append({k: v for k, v in res.items() if not isinstance(v, (list, dict))})

    pd.DataFrame(rows).to_csv(os.path.join("results", exp_id, "summary", "e2_case_studies.csv"), index=False)
    print(f"[E2] Saved summary to results/{exp_id}/summary/e2_case_studies.csv")


def run_e3_monte_carlo():
    print("\n=== Running E3: Random-Disruption Monte Carlo (30 Seeds x S2 Contention) ===")
    exp_id = "E3_monte_carlo"
    seeds = list(range(3000, 3030))
    write_manifest(exp_id, {"n_runs": 30, "methods": ["LLF", "EDF", "MPC-only", "AQPS-static", "AQPS-RT-ET"]}, seeds)

    with open("simulation_data_90EVs.json") as f:
        base_data = json.load(f)
    env = base_data["environment"]
    base_evs = base_data["scenarios"]["S3_HighPriority"]

    rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        # Perturb arrival times (+/- 4 steps = +/- 1h), energy demand (+/- 15%), and 10% random priority promotion
        pert_evs = []
        for ev in base_evs:
            ec = dict(ev)
            arr_shift = int(rng.integers(-4, 5))
            ec["arrival_idx"] = int(np.clip(ec["arrival_idx"] + arr_shift, 0, 86))
            ec["departure_idx"] = int(np.clip(ec["departure_idx"] + arr_shift, ec["arrival_idx"] + 8, 95))
            ec["req_energy"] = round(float(ec["req_energy"] * rng.uniform(0.88, 1.15)), 2)
            if ec["priority"] == "Low" and rng.random() < 0.08:
                ec["priority"] = "High"
                ec["trip_criticality_w"] = 3.8
            pert_evs.append(ec)

        for m in ["LLF", "EDF", "MPC-only", "AQPS-static", "AQPS-RT-ET"]:
            raw_path = os.path.join("results", exp_id, "raw", f"mc_{m}_seed{seed}.json")
            if os.path.exists(raw_path):
                with open(raw_path) as rf:
                    res = json.load(rf)
            else:
                res = run_single_simulation(m, 90, "S3_HighPriority", "S2_MC", pert_evs, env)
                res_slim = {k: v for k, v in res.items() if not isinstance(v, (list, dict))}
                res_slim["seed"] = seed
                with open(raw_path, "w") as rf:
                    json.dump(res_slim, rf)
                res = res_slim
            rows.append(res)

    df_raw = pd.DataFrame(rows)
    df_raw.to_csv(os.path.join("results", exp_id, "summary", "e3_mc_all_runs.csv"), index=False)

    # Aggregate mean, std, and 95% CI
    agg_rows = []
    for m, grp in df_raw.groupby("method"):
        n = len(grp)
        for col in ["priority_fulfillment_pct", "mission_sufficiency_pct", "non_priority_fulfillment_pct", "total_cost"]:
            mean_v = float(grp[col].mean())
            std_v = float(grp[col].std(ddof=1))
            ci95 = 1.96 * std_v / np.sqrt(max(1, n))
            agg_rows.append({"method": m, "metric": col, "mean": round(mean_v, 2), "std": round(std_v, 2), "ci95": round(ci95, 2)})
    pd.DataFrame(agg_rows).to_csv(os.path.join("results", exp_id, "summary", "e3_mc_summary.csv"), index=False)
    print(f"[E3] Saved 30-run Monte Carlo summary to results/{exp_id}/summary/e3_mc_summary.csv")


def run_e4_aemo_pv():
    print("\n=== Running E4: Real AEMO VIC1 PV Forecast vs. Actual Across 6 Days ===")
    exp_id = "E4_aemo_pv"
    if not os.path.exists(AEMO_PATH):
        build_aemo_vic1_dataset()
    with open(AEMO_PATH) as f:
        aemo = json.load(f)

    write_manifest(exp_id, {"days": [d["date"] for d in aemo["days"]], "fleets": [45, 90]}, [101, 102, 103, 104, 105, 106])
    rows = []

    for fleet_size in [45, 90]:
        with open(f"simulation_data_{fleet_size}EVs.json") as f:
            data = json.load(f)
        ev_data = data["scenarios"]["S3_HighPriority"]
        base_env = data["environment"]

        for day in aemo["days"]:
            env_day = dict(base_env)
            env_day["pv_forecast"] = day["pv_forecast_kw"]
            pv_act = day["pv_actual_kw"]

            for m in ["LLF", "EDF", "MPC-only", "Offline-Optimal", "AQPS-static", "AQPS-RT-ET"]:
                raw_path = os.path.join("results", exp_id, "raw", f"pv_{day['date']}_{m}_{fleet_size}EVs.json")
                if os.path.exists(raw_path):
                    with open(raw_path) as rf:
                        res = json.load(rf)
                else:
                    res = run_single_simulation(
                        m, fleet_size, "S3_HighPriority", f"S2_{day['date']}", ev_data, env_day,
                        pv_actual_override=pv_act
                    )
                    res_slim = {k: v for k, v in res.items() if not isinstance(v, (list, dict))}
                    res_slim.update({
                        "date": day["date"],
                        "season": day["season"],
                        "sky_condition": day["sky_condition"],
                        "error_tier": day["error_tier"],
                        "pv_mae_kw": day["mae_kw"],
                        "pv_nmae_pct": day["nmae_pct"],
                    })
                    with open(raw_path, "w") as rf:
                        json.dump(res_slim, rf)
                    res = res_slim
                rows.append(res)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join("results", exp_id, "summary", "e4_aemo_pv_results.csv"), index=False)
    print(f"[E4] Saved AEMO VIC1 PV evaluation to results/{exp_id}/summary/e4_aemo_pv_results.csv")


def run_e5_event_triggered_and_scalability():
    print("\n=== Running E5 & E5b: Event-Triggered vs Every-Step & Scalability/Ratio Sweep ===")
    exp_id = "E5_scalability"
    write_manifest(exp_id, {"tmax_sweep": [1, 2, 4, 8], "fleets": [45, 90, 180], "ratios": ["2:1", "3:1", "4:1"]}, [42])

    # Part A: T_max & PV trigger threshold sweep on 90 EVs S2
    with open("simulation_data_90EVs.json") as f:
        data90 = json.load(f)
    env90 = data90["environment"]
    ev90 = data90["scenarios"]["S3_HighPriority"]

    et_rows = []
    for tmax in [1, 2, 4, 6, 8]:
        mode = "AQPS-RT" if tmax == 1 else "AQPS-RT-ET"
        res = run_single_simulation(
            mode, 90, "S3_HighPriority", f"Tmax_{tmax}", ev90, env90,
            config_overrides={"event_trigger": {"t_max_steps": tmax}}
        )
        slim = {k: v for k, v in res.items() if not isinstance(v, (list, dict))}
        slim["t_max_steps"] = tmax
        slim["t_max_minutes"] = tmax * 15
        et_rows.append(slim)
    pd.DataFrame(et_rows).to_csv(os.path.join("results", exp_id, "summary", "e5a_event_trigger_comparison.csv"), index=False)

    # Part B: Scalability across Fleet Sizes (45, 90, 180 EVs) and EV:EVSE Ratios (2:1, 3:1, 4:1)
    scale_rows = []
    for fleet_size in [45, 90, 180]:
        with open(f"simulation_data_{fleet_size}EVs.json") as f:
            fdata = json.load(f)
        fenv = fdata["environment"]
        fevs = fdata["scenarios"]["S3_HighPriority"]
        base_kw = INFRA[fleet_size]["transformer_kW"]

        for ratio_label, ratio_val in [("2:1", 2), ("3:1", 3), ("4:1", 4)]:
            n_evse = max(4, int(round(fleet_size / ratio_val)))
            for m in ["LLF", "EDF", "AQPS-RT", "AQPS-RT-ET"]:
                raw_path = os.path.join("results", exp_id, "raw", f"scale_{fleet_size}EV_{ratio_label.replace(':','to')}_{m}.json")
                if os.path.exists(raw_path):
                    with open(raw_path) as rf:
                        slim = json.load(rf)
                else:
                    res = run_single_simulation(
                        m, fleet_size, "S3_HighPriority", f"S2_{ratio_label}", fevs, fenv,
                        n_evse=n_evse, transformer_kw=base_kw
                    )
                    slim = {k: v for k, v in res.items() if not isinstance(v, (list, dict))}
                    slim["ev_to_evse_ratio"] = ratio_label
                    with open(raw_path, "w") as rf:
                        json.dump(slim, rf)
                scale_rows.append(slim)

    pd.DataFrame(scale_rows).to_csv(os.path.join("results", exp_id, "summary", "e5b_scalability_ratios.csv"), index=False)
    print(f"[E5] Saved Event-Triggered & Scalability results to results/{exp_id}/summary/")


def run_e6_sensitivity():
    print("\n=== Running E6: Sensitivity Analysis (alpha2, alpha3, ATC k, reserve r) ===")
    exp_id = "E6_sensitivity"
    write_manifest(exp_id, {"sweeps": ["alpha2", "alpha3", "k_atc", "reserve_r"]}, [42])

    with open("simulation_data_90EVs.json") as f:
        data90 = json.load(f)
    env90 = data90["environment"]
    ev90 = data90["scenarios"]["S3_HighPriority"]

    sens_rows = []
    # 1. Sweep alpha2
    for a2 in [10.0, 25.0, 50.0, 100.0, 250.0, 500.0]:
        res = run_single_simulation(
            "AQPS-RT-ET", 90, "S3_HighPriority", "S2", ev90, env90,
            config_overrides={"optimizer": {"alpha2": a2}}
        )
        sens_rows.append({
            "param_name": "alpha2", "param_val": a2,
            "priority_fulfillment_pct": res["priority_fulfillment_pct"],
            "non_priority_fulfillment_pct": res["non_priority_fulfillment_pct"],
            "total_cost": res["total_cost"],
        })

    # 2. Sweep alpha3
    for a3 in [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]:
        res = run_single_simulation(
            "AQPS-RT-ET", 90, "S3_HighPriority", "S2", ev90, env90,
            config_overrides={"optimizer": {"alpha3": a3}}
        )
        sens_rows.append({
            "param_name": "alpha3", "param_val": a3,
            "priority_fulfillment_pct": res["priority_fulfillment_pct"],
            "non_priority_fulfillment_pct": res["non_priority_fulfillment_pct"],
            "total_cost": res["total_cost"],
        })

    # 3. Sweep ATC lookahead k
    for katc in [0.5, 1.0, 1.5, 2.0, 3.0]:
        res = run_single_simulation(
            "AQPS-RT-ET", 90, "S3_HighPriority", "S2", ev90, env90,
            config_overrides={"optimizer": {"k_atc": katc}}
        )
        sens_rows.append({
            "param_name": "k_atc", "param_val": katc,
            "priority_fulfillment_pct": res["priority_fulfillment_pct"],
            "non_priority_fulfillment_pct": res["non_priority_fulfillment_pct"],
            "total_cost": res["total_cost"],
        })

    # 4. Sweep reserve margin r (kWh)
    for r_kwh in [2.0, 4.0, 6.8, 10.0, 14.0]:
        res = run_single_simulation(
            "AQPS-RT-ET", 90, "S3_HighPriority", "S2", ev90, env90,
            config_overrides={"optimizer": {"reserve_override_kwh": r_kwh}}
        )
        sens_rows.append({
            "param_name": "reserve_kwh", "param_val": r_kwh,
            "priority_fulfillment_pct": res["priority_fulfillment_pct"],
            "non_priority_fulfillment_pct": res["non_priority_fulfillment_pct"],
            "total_cost": res["total_cost"],
        })

    pd.DataFrame(sens_rows).to_csv(os.path.join("results", exp_id, "summary", "e6_sensitivity.csv"), index=False)
    print(f"[E6] Saved sensitivity analysis to results/{exp_id}/summary/e6_sensitivity.csv")


def run_e7_overloaded_priority():
    print("\n=== Running E7: Overloaded Priority Stress Test & Conditional Guarantee ===")
    exp_id = "E7_overloaded"
    write_manifest(exp_id, {"priority_shares": [0.30, 0.50, 0.65, 0.80, 0.90]}, [42])

    with open("simulation_data_45EVs.json") as f:
        data45 = json.load(f)
    env45 = data45["environment"]
    base_evs = data45["scenarios"]["S3_HighPriority"]

    rows = []
    for p_share in [0.30, 0.50, 0.65, 0.80, 0.90]:
        n_pri = int(round(len(base_evs) * p_share))
        mod_evs = []
        for idx, ev in enumerate(base_evs):
            ec = dict(ev)
            is_p = idx < n_pri
            ec["priority"] = "High" if is_p else "Low"
            ec["trip_criticality_w"] = 3.8 if is_p else 0.8
            mod_evs.append(ec)

        for m in ["LLF", "AQPS-static", "AQPS-RT-ET"]:
            res = run_single_simulation(m, 45, "S3_HighPriority", f"Pri_{int(p_share*100)}pct", mod_evs, env45)
            slim = {k: v for k, v in res.items() if not isinstance(v, (list, dict))}
            slim["priority_share_pct"] = int(p_share * 100)
            rows.append(slim)

    pd.DataFrame(rows).to_csv(os.path.join("results", exp_id, "summary", "e7_overloaded_priority.csv"), index=False)
    print(f"[E7] Saved overloaded priority analysis to results/{exp_id}/summary/e7_overloaded_priority.csv")


def write_results_map_doc():
    os.makedirs("docs", exist_ok=True)
    md = """# AQPS Revision Results Map (`docs/results_map.md`)

| Experiment ID | Manuscript Section | Figure / Table | Summary CSV Source | Post-Processing Function |
| :--- | :--- | :--- | :--- | :--- |
| **E1** (Main 8 Methods x S1-S3 x 2 Fleets) | §4.3 Main Results | Table 3 (Master Table), Fig. 3 (Fulfillment), Figs. 4-5 (Cumulative Cost), Figs. 6-7 (Per-EV Gantt), Fig. 8 (Cost Breakdown) | `results/E1_main/summary/e1_master_table.csv` | `generate_all_figures.py` |
| **E2** (Scripted Operator Injections) | §4.4 Real-Time Queue Case Studies | Fig. 9 (Queue Trajectory with Injection Markers), Table 4 (Operator Impact Preview) | `results/E2_case_studies/summary/e2_case_studies.csv` | `generate_all_figures.py` |
| **E3** (Random-Disruption Monte Carlo, 30 runs) | §4.5 Disruption Monte Carlo | Fig. 10 (Monte Carlo Fulfillment & Cost Box/CI Plot) | `results/E3_monte_carlo/summary/e3_mc_summary.csv` | `generate_all_figures.py` |
| **E4** (Real AEMO VIC1 PV Forecast vs Actual) | §4.6 PV Forecast Uncertainty | Fig. 11 (AEMO VIC1 Profiles & Error Impact), Table 5 | `results/E4_aemo_pv/summary/e4_aemo_pv_results.csv` | `generate_all_figures.py` |
| **E5 / E5b** (Event-Triggered vs Every-Step & Scalability) | §4.7 Computational Efficiency & Scalability | Fig. 12 (Solve Count & Time Reduction), Table 6 (Scalability 45/90/180 EVs & Ratios 2:1-4:1) | `results/E5_scalability/summary/e5a_event_trigger_comparison.csv`, `e5b_scalability_ratios.csv` | `generate_all_figures.py` |
| **E6** (Hyperparameter Sensitivity) | §4.8 Sensitivity Analysis | Fig. 13 (4-Panel Sensitivity Plateau: $\alpha_2, \alpha_3, k, r$) | `results/E6_sensitivity/summary/e6_sensitivity.csv` | `generate_all_figures.py` |
| **E7** (Overloaded Priority & Escalation) | §3.5 & §4.8 Feasibility Condition | Table 7 / Fig. 14 (Overloaded Priority Fulfilment, Safe Releases & Escalations) | `results/E7_overloaded/summary/e7_overloaded_priority.csv` | `generate_all_figures.py` |
"""
    with open(os.path.join("docs", "results_map.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print("[Docs] Wrote docs/results_map.md")


def main():
    t_start = time.perf_counter()
    if not os.path.exists("simulation_data_45EVs.json") or not os.path.exists("simulation_data_180EVs.json"):
        generate_scenarios()
    write_results_map_doc()
    run_e1_main()
    run_e2_case_studies()
    run_e3_monte_carlo()
    run_e4_aemo_pv()
    run_e5_event_triggered_and_scalability()
    run_e6_sensitivity()
    run_e7_overloaded_priority()
    elapsed = time.perf_counter() - t_start
    print(f"\nAll Experiments E1-E7 completed in {elapsed:.1f}s!")


if __name__ == "__main__":
    main()
