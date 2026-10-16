"""
Experiment runner (revision v3): E1-E7.

Layout (plan §4a)
  results/<EXP>/manifest.json            git hash, config hash+snapshot, seeds, machine, versions
  results/<EXP>/raw/<run_id>.json.gz     one file per run (summary + optional per-step trace)
  results/<EXP>/summary/runs.csv         one row per run (no aggregation; post-processing does that)
Runs are resumable: an existing raw file is loaded instead of re-run.

Usage
  python run_all_experiments.py --exp all
  python run_all_experiments.py --exp E1 E4
  python run_all_experiments.py --exp E1 --fleet 45 --scenario S2_Contention --methods LLF AQPS-RT-ET
  python run_all_experiments.py --smoke --solver CLARABEL     # quick code check, writes results_smoke/
"""

import argparse
import copy
import gzip
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from src.aqps import (AQPSScheduler, AQPSv1Scheduler, EDFScheduler, LLFScheduler, MPCOnlyScheduler,
                      OfflineOptimal, Vehicle)

ROOT = os.path.dirname(os.path.abspath(__file__))
CFG_PATH = os.path.join(ROOT, "config", "experiments.yaml")
PV_DAYS_PATH = os.path.join(ROOT, "data", "aemo_vic1_selected_days.json")
INJ_PATH = os.path.join(ROOT, "data", "operator_injections.json")
ONLINE_AQPS = ("AQPS-static", "AQPS-RT", "AQPS-RT-ET")
TOL_MISSION = 0.1  # kWh
SITE_TOL_REL = 1e-3  # solver tolerance band on the site limit; larger excess is a hard error


# ============================================================================ helpers
def load_cfg(path=CFG_PATH) -> Dict:
    with open(path) as f:
        return yaml.safe_load(f)


def deep_update(d: Dict, u: Dict) -> Dict:
    d = copy.deepcopy(d)
    for k, v in (u or {}).items():
        d[k] = deep_update(d.get(k, {}), v) if isinstance(v, dict) else v
    return d


def load_fleet(size: int) -> Dict:
    fn = os.path.join(ROOT, f"simulation_data_{size}EVs.json")
    if not os.path.exists(fn):
        sys.exit(f"{fn} missing — run data_generator.py first")
    with open(fn) as f:
        return json.load(f)


def tier_of(price: float, cfg: Dict) -> str:
    for tier in ("peak", "shoulder", "off_peak"):
        if abs(price - cfg["tariff"][tier]["price"]) < 1e-9:
            return tier
    return "other"


def pv_forecast_window(fc: np.ndarray, act_now: float, t: int, ramp: int = 8) -> np.ndarray:
    """Day-ahead forecast from step t, with a persistence correction of the ratio actual/forecast
    observed at t that decays linearly back to the day-ahead forecast over `ramp` steps.
    Applied identically to every optimisation-based method."""
    base = np.asarray(fc[t:], dtype=float).copy()
    if base.size and fc[t] > 0.5:
        ratio = float(np.clip(act_now / fc[t], 0.0, 1.5))
        w = np.clip(np.arange(base.size) / ramp, 0.0, 1.0)
        base = base * (ratio + (1.0 - ratio) * w)
    return base


def apply_injection(inj: Dict, vehicles: Dict[str, Vehicle], t: int, T: int, pv_act: np.ndarray):
    typ = inj["type"]
    if typ == "vehicle_breakdown":
        v = vehicles[inj["promote_id"]]
        v.roster_priority = True
        v.trip_criticality_w = float(inj["new_w"])
        v.e_mission_kwh = v.energy_delivered + float(inj["new_mission_kwh"])
        v.e_session_kwh = max(v.e_session_kwh, v.e_mission_kwh)
        v.next_trip_kwh = max(v.next_trip_kwh, v.current_energy_kwh + float(inj["new_mission_kwh"])
                              - v.return_kwh - v.reserve_kwh)
        v.departure_idx = int(max(t + 1, inj["new_dep_idx"]))
    elif typ == "driver_unavailable":
        v = vehicles[inj["delay_id"]]
        v.departure_idx = int(min(T - 1, v.departure_idx + int(inj["delay_steps"])))
        v.evening_return_idx = int(min(T - 1, v.evening_return_idx + int(inj["delay_steps"])))
    elif typ == "roster_swap":
        a, b = vehicles[inj["swap_from_id"]], vehicles[inj["swap_to_id"]]
        a.roster_priority, b.roster_priority = False, True
        a.trip_criticality_w, b.trip_criticality_w = 0.8, max(b.trip_criticality_w, 4.0)
        b.next_trip_kwh, a.next_trip_kwh = a.next_trip_kwh, b.next_trip_kwh
        for v in (a, b):
            v.e_mission_kwh = max(0.0, v.mission_level_kwh - v.eod_soc_kwh)
            v.e_session_kwh = max(v.e_session_kwh, v.e_mission_kwh)
    elif typ == "promotion":
        v = vehicles[inj["id"]]
        v.roster_priority = True
        v.trip_criticality_w = max(v.trip_criticality_w, float(inj.get("new_w", 3.8)))
    elif typ == "pv_shortfall":
        d = int(inj.get("duration_steps", 1))
        pv_act[t:t + d] *= float(inj["pv_drop_factor"])
    else:
        raise ValueError(typ)


# ============================================================================ one run
def simulate(method: str, records: List[Dict], env: Dict, fleet_size: int, n_evse: int, site_kw: float,
             cfg: Dict, injections: Optional[Dict[int, List[Dict]]] = None,
             pv_fc: Optional[List[float]] = None, pv_act: Optional[List[float]] = None,
             reserve_frac: Optional[float] = None, trace: bool = False) -> Dict:
    dt = cfg["time"]["step_minutes"] / 60.0
    T = int(env["n_steps"])
    tou = np.asarray(env["tou_tariff"], dtype=float)
    fc = np.asarray(pv_fc if pv_fc is not None else env["pv_forecast"], dtype=float)
    act = np.asarray(pv_act if pv_act is not None else env["pv_actual"], dtype=float).copy()
    bess = np.asarray(env["bess"], dtype=float)
    vehicles = {r["id"]: Vehicle.from_record(r, reserve_frac) for r in records}
    injections = injections or {}

    agg = {"l0": 0, "l2": 0, "prev": 0, "l2_ms": [], "all_ms": 0.0, "soft": 0, "infeas": 0,
           "commit_inf": 0, "nonopt": 0, "pre": 0, "sr": 0, "rot": 0, "esc": 0, "extra_cost": 0.0,
           "previews": [], "events": 0, "site_trims": 0, "site_trim_kw_max": 0.0, "unservable": 0}
    tr = {"gross": [], "net": [], "power": {}, "connected": [], "priority": [], "reasons": []} if trace else None
    offline_info = None

    if method == "Offline-Optimal":
        # perfect information: injections applied up-front
        for t0 in sorted(injections):
            for inj in injections[t0]:
                apply_injection(inj, vehicles, t0, T, act)
        solver = OfflineOptimal(cfg, n_evse, site_kw)
        out = solver.solve(list(vehicles.values()), tou, act, bess, integer=cfg["optimizer"]["solver"] == "MOSEK")
        offline_info = out["info"]
        sched = out["schedule"]
        agg["l2"] = 1
        agg["l2_ms"] = [offline_info["solve_ms"]]
        agg["all_ms"] = offline_info["solve_ms"]
    elif method in ONLINE_AQPS:
        sched_obj = AQPSScheduler(cfg, n_evse, site_kw, method)
    elif method == "AQPS-v1":
        sched_obj = AQPSv1Scheduler(n_evse, site_kw, dt, cfg["optimizer"]["solver"])
    elif method == "MPC-only":
        sched_obj = MPCOnlyScheduler(cfg, n_evse, site_kw)
    elif method == "LLF":
        sched_obj = LLFScheduler(n_evse, site_kw, dt)
    elif method == "EDF":
        sched_obj = EDFScheduler(n_evse, site_kw, dt)
    else:
        raise ValueError(method)

    cost = {"off_peak": 0.0, "shoulder": 0.0, "peak": 0.0, "other": 0.0}
    gross_tr, pv_used, grid_kwh = np.zeros(T), 0.0, 0.0
    snap22 = None
    idx22 = int((22 - cfg["time"]["start_hour"]) / dt)

    for t in range(T):
        if method != "Offline-Optimal":
            for inj in injections.get(t, []):
                apply_injection(inj, vehicles, t, T, act)
                if inj["type"] == "promotion":  # mission physically out of reach at promotion time
                    v = vehicles[inj["id"]]
                    agg["unservable"] += int(v.mission_remaining > v.p_max_kw * max(0, v.departure_idx - t) * dt + TOL_MISSION)
        present = [v for v in vehicles.values() if v.arrival_idx <= t < v.departure_idx]
        arrivals = [v.id for v in present if v.arrival_idx == t]
        departures = [v.id for v in vehicles.values() if v.departure_idx == t]
        inj_now = injections.get(t, [None])[-1] if method in ONLINE_AQPS else None

        if method == "Offline-Optimal":
            cmds = {v.id: float(sched[v.id][t]) for v in present if sched[v.id][t] > 1e-6}
            out = {}
        elif method in ("LLF", "EDF"):
            out = sched_obj.step(t, present)
            cmds = out["commands"]
            agg["all_ms"] += out["all_ms"]
        else:
            pv_h = pv_forecast_window(fc, act[t], t)
            tou_h, bess_h = tou[t:], bess[t:]
            if method == "MPC-only":
                out = sched_obj.step(t, present, tou_h, pv_h, bess_h)
                agg["l2"] += out["l2_solves"]
                agg["l2_ms"] += out["l2_ms"]
                agg["all_ms"] += out["all_ms"]
                agg["nonopt"] += sum(s not in ("optimal", "optimal_inaccurate") for s in out.get("status", []))
            elif method == "AQPS-v1":
                out = sched_obj.step(t, present, tou_h, pv_h, bess_h)
                agg["l2"] += out["solves"]
                if out["solves"]:
                    agg["l2_ms"].append(out["solve_ms"])
                    agg["all_ms"] += out["solve_ms"]
                    agg["nonopt"] += int(out["status"] not in ("optimal", "optimal_inaccurate"))
            else:
                out = sched_obj.step(t, present, arrivals, departures, tou_h, pv_h, bess_h, float(act[t]), inj_now)
                agg["l0"] += out["l0_solves"]
                agg["l2"] += out["l2_solves"]
                agg["prev"] += out["preview_solves"]
                agg["l2_ms"] += out["l2_ms"]
                agg["all_ms"] += out["all_ms"]
                agg["soft"] += out["soft_solves"]
                agg["infeas"] += out["infeasible_first"]
                agg["commit_inf"] += out["committed_infeasible"]
                agg["nonopt"] += sum(s not in ("optimal", "optimal_inaccurate") for s in out["status"])
                agg["pre"] += len(out["preempted"])
                agg["sr"] += len(out["safe_released"])
                agg["rot"] += len(out["rotated"])
                agg["esc"] += len(out["escalation_new"])
                agg["events"] += int(out["event_reasons"] != [])
                if out["preview"] is not None:
                    rec = out["preview"]
                    agg["extra_cost"] += rec.extra_cost
                    agg["previews"].append({"t": rec.t, "trigger": rec.trigger, "chosen": rec.chosen,
                                            "extra_cost": rec.extra_cost, "options": rec.options})
            cmds = out["commands"]

        # apply commands (single place where energy changes, for every method)
        for vid in list(cmds):
            v = vehicles[vid]
            cmds[vid] = max(0.0, min(cmds[vid], v.p_max_kw, v.session_remaining / dt))
        gross = sum(cmds.values())
        if gross > site_kw * (1.0 + SITE_TOL_REL):
            raise RuntimeError(f"{method}: site limit violated at t={t}: {gross:.4f} kW > {site_kw} kW")
        if gross > site_kw:  # solver feasibility tolerance (<= 0.1 %): scale back to the limit
            agg["site_trims"] += 1
            agg["site_trim_kw_max"] = max(agg["site_trim_kw_max"], gross - site_kw)
            for vid in cmds:
                cmds[vid] *= site_kw / gross
            gross = site_kw
        for vid, p in cmds.items():
            vehicles[vid].energy_delivered += p * dt
        net = max(0.0, gross - act[t] - bess[t])
        cost[tier_of(tou[t], cfg)] += net * tou[t] * dt
        gross_tr[t] = gross
        grid_kwh += net * dt
        pv_used += min(act[t], gross + max(0.0, -bess[t])) * dt
        if t == idx22:
            snap22 = {vid: v.energy_delivered for vid, v in vehicles.items()}
        if trace:
            tr["gross"].append(round(gross, 4))
            tr["net"].append(round(net, 4))
            for vid, p in cmds.items():
                tr["power"].setdefault(vid, [0.0] * T)[t] = round(p, 4)
            tr["connected"].append(out.get("connected", sorted(cmds)) if out else sorted(cmds))
            tr["priority"].append(sorted(v.id for v in present if v.is_priority))
            tr["reasons"].append(out.get("event_reasons", []) if out else [])

    return summarise(method, vehicles, cfg, dt, T, cost, gross_tr, grid_kwh, pv_used, act, site_kw,
                     agg, snap22, offline_info, tr, fleet_size, n_evse)


def summarise(method, vehicles, cfg, dt, T, cost, gross_tr, grid_kwh, pv_used, act, site_kw, agg,
              snap22, offline_info, tr, fleet_size, n_evse) -> Dict:
    vs = list(vehicles.values())
    P = [v for v in vs if v.roster_priority]
    N = [v for v in vs if not v.roster_priority]

    def frac(group, key=lambda v: v.energy_delivered):
        den = sum(v.e_session_kwh for v in group)
        return 100.0 * sum(min(key(v), v.e_session_kwh) for v in group) / den if den > 0 else 100.0

    e_p = sum(v.energy_delivered for v in P)
    e_n = sum(v.energy_delivered for v in N)
    e_tot = e_p + e_n
    tot_cost = sum(cost.values())
    ratios = np.array([min(1.0, v.energy_delivered / v.e_session_kwh) for v in N if v.e_session_kwh > 0])
    jain = float(ratios.sum() ** 2 / (len(ratios) * (ratios ** 2).sum())) if ratios.size and (ratios ** 2).sum() > 0 else 1.0
    peak = float(gross_tr.max()) if gross_tr.size else 0.0
    pv_tot = float(np.sum(np.maximum(act, 0.0)) * dt)
    l2_ms = np.asarray(agg["l2_ms"]) if agg["l2_ms"] else np.zeros(0)
    row = {
        "method": method, "fleet_size": fleet_size, "n_evse": n_evse, "site_kw": site_kw,
        "n_priority": len(P), "n_nonpriority": len(N),
        "mission_sufficiency_pct": round(100.0 * sum(v.energy_delivered >= v.e_mission_kwh - TOL_MISSION for v in P) / len(P), 3) if P else 100.0,
        "mission_shortfall_kwh": round(sum(max(0.0, v.e_mission_kwh - v.energy_delivered) for v in P), 3),
        "priority_session_fulfil_pct": round(frac(P), 3),
        "nonpriority_fulfil_pct": round(frac(N), 3),
        "nonpriority_fulfil_22h_pct": round(frac(N, key=lambda v: snap22.get(v.id, 0.0)), 3) if snap22 else None,
        "nonpriority_jain": round(jain, 4),
        "energy_priority_kwh": round(e_p, 3), "energy_nonpriority_kwh": round(e_n, 3),
        "energy_total_kwh": round(e_tot, 3), "grid_import_kwh": round(grid_kwh, 3),
        "cost_total": round(tot_cost, 4), "unit_cost_per_kwh": round(tot_cost / e_tot, 5) if e_tot > 0 else None,
        "cost_off_peak": round(cost["off_peak"], 4), "cost_shoulder": round(cost["shoulder"], 4),
        "cost_peak": round(cost["peak"], 4),
        "pv_used_kwh": round(pv_used, 3), "pv_self_consumption_pct": round(100.0 * pv_used / pv_tot, 3) if pv_tot > 0 else None,
        "peak_kw": round(peak, 3), "load_factor": round(float(gross_tr.mean() / peak), 4) if peak > 0 else None,
        "transformer_loading_pct": round(100.0 * peak / site_kw, 3),
        "preemptions": agg["pre"], "safe_releases": agg["sr"], "rotations": agg["rot"],
        "escalation_episodes": agg["esc"], "previews": len(agg["previews"]),
        "extra_operator_cost": round(agg["extra_cost"], 4),
        "l0_solves": agg["l0"], "l2_solves": agg["l2"], "preview_solves": agg["prev"],
        "total_solves": agg["l0"] + agg["l2"] + agg["prev"],
        "l2_solve_ms_mean": round(float(l2_ms.mean()), 3) if l2_ms.size else 0.0,
        "l2_solve_ms_p95": round(float(np.percentile(l2_ms, 95)), 3) if l2_ms.size else 0.0,
        "compute_ms_total": round(agg["all_ms"], 2), "event_steps": agg["events"],
        "soft_solves": agg["soft"], "eq10_infeasible_first": agg["infeas"],
        "committed_infeasible": agg["commit_inf"], "nonoptimal_status": agg["nonopt"],
        "site_trims": agg["site_trims"], "site_trim_kw_max": round(agg["site_trim_kw_max"], 6),
        "promoted_unservable": agg["unservable"],
        "solver": cfg["optimizer"]["solver"],
    }
    if offline_info is not None:
        row.update({"offline_status": offline_info["status"], "offline_integer": offline_info["integer"],
                    "offline_soft": offline_info["soft"], "offline_mip_gap_bound": offline_info.get("mip_gap"),
                    "offline_relax_objective": offline_info.get("relax_objective"),
                    "offline_objective": offline_info["objective"]})
    per_ev = [{"id": v.id, "roster_priority": v.roster_priority, "day_ahead_priority": v.day_ahead_priority,
               "arrival_idx": v.arrival_idx, "departure_idx": v.departure_idx, "e_session": v.e_session_kwh,
               "e_mission": v.e_mission_kwh, "delivered": round(v.energy_delivered, 4),
               "preempted": v.preempted_count, "rotated": v.rotated_count, "safe_released": v.safe_released}
              for v in vs]
    return {"summary": row, "per_ev": per_ev, "previews": agg["previews"], "trace": tr}


# ============================================================================ bookkeeping
def git_info():
    try:
        h = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                    stderr=subprocess.DEVNULL).decode().strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--", "src", "run_all_experiments.py",
                                              "config", "data_generator.py"], cwd=ROOT,
                                             stderr=subprocess.DEVNULL).decode().strip())
        return h, dirty
    except Exception:
        return "unknown", True


def versions():
    out = {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__}
    try:
        import cvxpy
        out["cvxpy"] = cvxpy.__version__
    except Exception:
        pass
    try:
        import mosek
        out["mosek"] = ".".join(map(str, mosek.Env.getversion()))
    except Exception:
        out["mosek"] = None
    return out


def file_sha(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def write_manifest(exp_dir: str, exp: str, cfg: Dict, extra: Dict):
    h, dirty = git_info()
    m = {
        "experiment": exp, "git_commit": h, "git_dirty": dirty,
        "config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
        "config": cfg, "run_started": datetime.now().isoformat(timespec="seconds"),
        "machine": {"os": platform.platform(), "processor": platform.processor(), "machine": platform.machine(),
                    "cpu_count": os.cpu_count()},
        "versions": versions(),
        "inputs": {"pv_days_sha256": file_sha(PV_DAYS_PATH), "injections_sha256": file_sha(INJ_PATH),
                   "aemo_checksums_sha256": file_sha(os.path.join(ROOT, "data", "aemo_raw", "checksums.csv")),
                   **{f"fleet_{s}_sha256": file_sha(os.path.join(ROOT, f"simulation_data_{s}EVs.json")) for s in (45, 90, 180)}},
        **extra,
    }
    os.makedirs(exp_dir, exist_ok=True)
    path = os.path.join(exp_dir, "manifest.json")
    # keep earlier invocations (resumed runs may come from an earlier commit; each run row also
    # carries its own git_commit in runs.csv)
    history = []
    if os.path.exists(path):
        with open(path) as f:
            old = json.load(f)
        history = old.pop("history", [])
        history.append({k: old.get(k) for k in ("git_commit", "git_dirty", "config_sha256", "run_started", "versions")})
    m["history"] = history
    with open(path, "w") as f:
        json.dump(m, f, indent=2, default=str)


class Runner:
    def __init__(self, out_root: str, cfg: Dict, fresh: bool = False):
        self.out_root, self.cfg, self.fresh = out_root, cfg, fresh  # fresh: ignore cached raw runs

    def run(self, exp: str, run_id: str, fn, meta: Dict, trace: bool = False):
        raw_dir = os.path.join(self.out_root, exp, "raw")
        os.makedirs(raw_dir, exist_ok=True)
        path = os.path.join(raw_dir, f"{run_id}.json.gz")
        if os.path.exists(path) and not self.fresh:
            with gzip.open(path, "rt") as f:
                return json.load(f)
        t0 = time.perf_counter()
        res = fn()
        res["summary"].update(meta)
        res["summary"]["run_id"] = run_id
        res["summary"]["wall_s"] = round(time.perf_counter() - t0, 2)
        res["summary"]["git_commit"], res["summary"]["git_dirty"] = git_info()
        if not trace:
            res["trace"] = None
        with gzip.open(path + ".part", "wt") as f:
            json.dump(res, f, default=float)
        os.replace(path + ".part", path)
        print(f"  [{exp}] {run_id}: {res['summary']['wall_s']} s")
        return res

    def write_summary(self, exp: str, results: List[Dict]):
        d = os.path.join(self.out_root, exp, "summary")
        os.makedirs(d, exist_ok=True)
        pd.DataFrame([r["summary"] for r in results]).to_csv(os.path.join(d, "runs.csv"), index=False)


# ============================================================================ experiments
def fleet_ctx(size, cfg, n_evse=None):
    data = load_fleet(size)
    infra = cfg["fleets"][size] if size in cfg["fleets"] else cfg["fleets"][str(size)]
    return data, (n_evse or infra["n_evse"]), infra["site_kw"]


def e1(R: Runner, cfg, sel):
    c = cfg["experiments"]["E1"]
    out = []
    for size in sel.get("fleets") or c["fleets"]:
        data, M, site = fleet_ctx(size, cfg)
        for sc in sel.get("scenarios") or c["scenarios"]:
            for m in sel.get("methods") or c["methods"]:
                rid = f"{m}_{sc}_{size}EV"
                out.append(R.run("E1", rid, lambda m=m, sc=sc: simulate(
                    m, data["scenarios"][sc], data["environment"], size, M, site, cfg, trace=True),
                    {"scenario": sc}, trace=True))
    R.write_summary("E1", out)


def load_injections():
    if not os.path.exists(INJ_PATH):
        sys.exit(f"{INJ_PATH} missing — run data_generator.py first")
    with open(INJ_PATH) as f:
        return json.load(f)


def e2(R: Runner, cfg, sel):
    c = cfg["experiments"]["E2"]
    data, M, site = fleet_ctx(c["fleet"], cfg)
    out = []
    for case, inj in load_injections().items():
        for m in sel.get("methods") or c["methods"]:
            rid = f"{m}_{case}"
            out.append(R.run("E2", rid, lambda m=m, inj=inj: simulate(
                m, data["scenarios"][c["scenario"]], data["environment"], c["fleet"], M, site, cfg,
                injections={inj["time_idx"]: [inj]}, trace=True),
                {"scenario": c["scenario"], "case": case, "injection_type": inj["type"],
                 "injection_t": inj["time_idx"]}, trace=True))
    R.write_summary("E2", out)
    prev = []
    for r in out:
        for p in r.get("previews") or []:
            for o in p["options"]:
                prev.append({"run_id": r["summary"]["run_id"], "case": r["summary"]["case"], "t": p["t"],
                             "trigger": p["trigger"], "chosen": p["chosen"], "option": o["option"],
                             "horizon_cost": o["horizon_cost"], "cost_delta": o["cost_delta"],
                             "soft": o["soft"], "n_priority": len(o["priority_ids"])})
    pd.DataFrame(prev).to_csv(os.path.join(R.out_root, "E2", "summary", "previews.csv"), index=False)


def disrupt(records: List[Dict], seed: int, dcfg: Dict, T: int):
    rng = np.random.default_rng(seed)
    recs, inj = [], {}
    for r in records:
        e = dict(r)
        sh = int(rng.integers(-dcfg["arrival_shift_steps"], dcfg["arrival_shift_steps"] + 1))
        e["arrival_idx"] = int(np.clip(e["arrival_idx"] + sh, 0, T - 9))
        e["departure_idx"] = int(np.clip(e["departure_idx"] + sh, e["arrival_idx"] + 8, T - 1))
        e["evening_return_idx"] = int(np.clip(e["evening_return_idx"] + sh, e["departure_idx"], T - 1))
        # energy deviation s scales the next-trip energy and the session target; E_mission is
        # recomputed from the physical mission level so Eq. 10 and safe release (a) stay consistent
        s = float(rng.uniform(*dcfg["energy_scale"]))
        e["next_trip_kwh"] = round(e["next_trip_kwh"] * s, 3)
        e["e_session_kwh"] = round(min(e["e_session_kwh"] * s, e["battery_cap_kwh"] - e["eod_soc_kwh"]), 3)
        need = e["next_trip_kwh"] + e["return_kwh"] + e["reserve_kwh"] - e["eod_soc_kwh"]
        e["e_mission_kwh"] = round(min(e["e_session_kwh"], max(0.0, need)), 3)
        if e["priority"] == "Low" and not e.get("overnight") and rng.random() < dcfg["promotion_prob"]:
            hi = max(e["arrival_idx"] + 1, e["departure_idx"] - 4)
            tp = int(rng.integers(e["arrival_idx"], hi))
            inj.setdefault(tp, []).append({"type": "promotion", "id": e["id"], "new_w": 3.8})
        recs.append(e)
    return recs, inj


def e3(R: Runner, cfg, sel):
    c = cfg["experiments"]["E3"]
    data, M, site = fleet_ctx(c["fleet"], cfg)
    s0, s1 = c["seeds"]
    seeds = list(range(s0, s1 + 1))[: sel.get("n_seeds") or None]
    out = []
    for seed in seeds:
        recs, inj = disrupt(data["scenarios"][c["scenario"]], seed, c["disruption"], data["environment"]["n_steps"])
        for m in sel.get("methods") or c["methods"]:
            out.append(R.run("E3", f"{m}_{c['scenario']}_seed{seed}", lambda m=m, recs=recs, inj=inj: simulate(
                m, recs, data["environment"], c["fleet"], M, site, cfg, injections=inj),
                {"scenario": c["scenario"], "seed": seed, "n_promotions": sum(len(v) for v in inj.values())}))
    R.write_summary("E3", out)


def e4(R: Runner, cfg, sel):
    c = cfg["experiments"]["E4"]
    with open(PV_DAYS_PATH) as f:
        pvd = json.load(f)
    out = []
    for size in sel.get("fleets") or c["fleets"]:
        data, M, site = fleet_ctx(size, cfg)
        for day in pvd["days"]:
            for m in sel.get("methods") or c["methods"]:
                rid = f"{m}_{size}EV_{day['date']}"
                out.append(R.run("E4", rid, lambda m=m, day=day: simulate(
                    m, data["scenarios"][c["scenario"]], data["environment"], size, M, site, cfg,
                    pv_fc=day["pv_forecast_kw"], pv_act=day["pv_actual_kw"]),
                    {"scenario": c["scenario"], "pv_date": day["date"], "pv_percentile": day["percentile"],
                     "pv_nmae_pct": day["nmae_pct"]}))
    R.write_summary("E4", out)


def e5(R: Runner, cfg, sel):
    c = cfg["experiments"]["E5"]
    out = []
    # (a) event-triggered vs every-step: all scenarios x fleets
    for size in sel.get("fleets") or c["fleets"]:
        data, M, site = fleet_ctx(size, cfg)
        for sc in sel.get("scenarios") or c["scenarios"]:
            for m in c["methods"]:
                out.append(R.run("E5", f"a_{m}_{sc}_{size}EV", lambda m=m, sc=sc: simulate(
                    m, data["scenarios"][sc], data["environment"], size, M, site, cfg),
                    {"part": "a_et_vs_every_step", "scenario": sc}))
    # (b) T_max sweep, 90-EV S2
    data, M, site = fleet_ctx(90, cfg)
    for tm in c["t_max_sweep"]:
        cfg2 = deep_update(cfg, {"event_trigger": {"t_max_steps": tm}})
        out.append(R.run("E5", f"b_tmax{tm}", lambda cfg2=cfg2: simulate(
            "AQPS-RT-ET", data["scenarios"]["S2_Contention"], data["environment"], 90, M, site, cfg2),
            {"part": "b_tmax_sweep", "scenario": "S2_Contention", "t_max_steps": tm}))
    # (c) scalability / EV:EVSE ratio sweep, S2
    sc_cfg = c["scalability"]
    for size in sc_cfg["fleets"]:
        data, _, site = fleet_ctx(size, cfg)
        for ratio in sc_cfg["ratios"]:
            M = int(round(size / ratio))
            for m in sc_cfg["methods"]:
                out.append(R.run("E5", f"c_{m}_{size}EV_r{ratio}", lambda m=m, M=M, data=data, size=size, site=site: simulate(
                    m, data["scenarios"]["S2_Contention"], data["environment"], size, M, site, cfg),
                    {"part": "c_scalability", "scenario": "S2_Contention", "ratio": ratio}))
    R.write_summary("E5", out)


def e6(R: Runner, cfg, sel):
    c = cfg["experiments"]["E6"]
    data, M, site = fleet_ctx(c["fleet"], cfg)
    recs, env = data["scenarios"][c["scenario"]], data["environment"]
    out = []
    sweeps = [("alpha3", "optimizer", "alpha3"), ("alpha2", "optimizer", "alpha2"),
              ("pv_dev_threshold_kw", "event_trigger", "pv_dev_threshold_kw"),
              ("preempt_slack_h", "layer1", "preempt_slack_h")]
    for key, sect, field in sweeps:
        for val in c[key]:
            cfg2 = deep_update(cfg, {sect: {field: val}})
            out.append(R.run("E6", f"{key}_{val}", lambda cfg2=cfg2: simulate(
                c["method"], recs, env, c["fleet"], M, site, cfg2), {"param": key, "value": val}))
    M_k = int(round(c["fleet"] / c["k_atc_ratio"]))
    for val in c["k_atc"]:
        cfg2 = deep_update(cfg, {"layer0": {"k_atc": val}})
        out.append(R.run("E6", f"k_atc_{val}", lambda cfg2=cfg2: simulate(
            c["method"], recs, env, c["fleet"], M_k, site, cfg2),
            {"param": "k_atc", "value": val, "n_evse_override": M_k}))
    for val in c["reserve_frac"]:
        out.append(R.run("E6", f"reserve_{val}", lambda val=val: simulate(
            c["method"], recs, env, c["fleet"], M, site, cfg, reserve_frac=val),
            {"param": "reserve_frac", "value": val}))
    R.write_summary("E6", out)


def e7(R: Runner, cfg, sel):
    c = cfg["experiments"]["E7"]
    data, M, site = fleet_ctx(c["fleet"], cfg)
    base = data["scenarios"][c["scenario"]]
    # contended cohort: centred on the median arrival of the scenario's own day-ahead priority EVs
    centre = float(np.median([r["arrival_idx"] for r in base if r["priority"] == "High"]))
    daytime = [r for r in base if not r.get("overnight")]
    order = sorted(daytime, key=lambda r: (abs(r["arrival_idx"] - centre), r["id"]))
    out = []
    for share in c["shares"]:
        n_pri = int(round(len(base) * share))
        pri_ids = {r["id"] for r in order[:n_pri]}
        recs = []
        for r in base:
            e = dict(r)
            is_p = e["id"] in pri_ids
            e["priority"] = "High" if is_p else "Low"
            e["trip_criticality_w"] = 3.8 if is_p else 0.8
            recs.append(e)
        for m in sel.get("methods") or c["methods"]:
            out.append(R.run("E7", f"{m}_{c['scenario']}_share{share}", lambda m=m, recs=recs: simulate(
                m, recs, data["environment"], c["fleet"], M, site, cfg),
                {"scenario": c["scenario"], "priority_share": share, "n_priority_assigned": n_pri,
                 "cohort_centre_idx": centre}))
    R.write_summary("E7", out)


EXPS = {"E1": e1, "E2": e2, "E3": e3, "E4": e4, "E5": e5, "E6": e6, "E7": e7}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="+", default=["all"])
    ap.add_argument("--fleet", nargs="+", type=int)
    ap.add_argument("--scenario", nargs="+")
    ap.add_argument("--methods", nargs="+")
    ap.add_argument("--n-seeds", type=int)
    ap.add_argument("--solver")
    ap.add_argument("--out", default="results")
    ap.add_argument("--smoke", action="store_true", help="E1, 45 EVs, S2 only, all methods -> results_smoke/")
    ap.add_argument("--fresh", action="store_true", help="recompute every run (ignore cached results/<EXP>/raw)")
    args = ap.parse_args()

    cfg = load_cfg()
    cfg["fleets"] = {int(k): v for k, v in cfg["fleets"].items()}
    if args.solver:
        cfg["optimizer"]["solver"] = args.solver.upper()
    exps = list(EXPS) if args.exp == ["all"] else args.exp
    out_root = args.out
    sel = {"fleets": args.fleet, "scenarios": args.scenario, "methods": args.methods, "n_seeds": args.n_seeds}
    if args.smoke:
        out_root = "results_smoke"
        exps = exps if args.exp != ["all"] else ["E1"]
        sel = {"fleets": args.fleet or [45], "scenarios": args.scenario or ["S2_Contention"],
               "methods": args.methods, "n_seeds": args.n_seeds or 1}
    R = Runner(out_root, cfg, fresh=args.fresh)
    for exp in exps:
        print(f"=== {exp} ===")
        write_manifest(os.path.join(out_root, exp), exp, cfg, {"selection": sel, "fresh": args.fresh})
        EXPS[exp](R, cfg, sel)


if __name__ == "__main__":
    main()
