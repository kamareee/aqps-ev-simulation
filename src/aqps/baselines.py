"""
WP5 — Competitive Baselines:
1. LLFScheduler      : Least Laxity First (deadline + remaining charge time slack, greedy max rate, no preemption of priority).
2. EDFScheduler      : Earliest Deadline First (sorted by departure_time_idx, max feasible rate, no priority awareness).
3. MPCOnlyScheduler  : Layer 2 QP alone with FIFO admission, no preemption, no priority lower bound.
4. OfflineOptimal    : Full-day convex QP with perfect foresight of arrivals, PV, and injections (theoretical upper bound).
"""

import time
import cvxpy as cp
import numpy as np
from typing import List, Dict
from .data_structures import Vehicle
from .tou_optimization import TOUOptimizer


class LLFScheduler:
    """Least Laxity First (LLF) baseline under identical EVSE (M) and gross site limit (P_site_max)."""
    def __init__(self, num_chargers: int, grid_limit: float, dt_hours: float = 0.25):
        self.num_chargers = num_chargers
        self.grid_limit = grid_limit
        self.dt = dt_hours
        self.active_vehicles: List[Vehicle] = []

    def step(self, time_idx: int, arriving: List[Vehicle], departing_ids: List[str], current_cap: float = None) -> Dict:
        t0 = time.perf_counter()
        cap = self.grid_limit if current_cap is None else min(self.grid_limit, current_cap)
        self.active_vehicles = [
            v for v in self.active_vehicles if v.id not in departing_ids and v.departure_time_idx > time_idx and v.energy_needed > 0.05
        ]
        self.active_vehicles.extend(arriving)

        for v in self.active_vehicles:
            time_rem = (v.departure_time_idx - time_idx) * self.dt
            if time_rem <= 0:
                v.laxity = -999.0
            else:
                v.laxity = time_rem - (v.energy_needed / max(v.max_charging_power_kw, 0.01))

        self.active_vehicles.sort(key=lambda x: (x.laxity, x.arrival_time_idx))
        commands = {}
        remaining_cap = cap

        for v in self.active_vehicles[: self.num_chargers]:
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
    """Earliest Deadline First (EDF): sorts strictly by departure_time_idx, non-preemptive FIFO tie-break."""
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
            v for v in self.active_vehicles if v.id not in departing_ids and v.departure_time_idx > time_idx and v.energy_needed > 0.05
        ]
        for v in self.active_vehicles:
            if v.energy_needed <= 0.05:
                self.connected_ids.discard(v.id)
        self.active_vehicles.extend(arriving)

        # Non-preemptive: already connected vehicles keep their EVSE until finished or departed;
        # free EVSEs are assigned by earliest deadline (departure_time_idx)
        self.active_vehicles.sort(key=lambda v: (v.id not in self.connected_ids, v.departure_time_idx, v.arrival_time_idx))
        candidates = self.active_vehicles[: self.num_chargers]
        self.connected_ids = {v.id for v in candidates}

        # Allocate power in deadline order
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
    """MPC-only baseline: Layer 2 receding-horizon QP with FIFO non-preemptive admission and no priority lower bound."""
    def __init__(self, config: Dict):
        self.num_chargers = int(config["station"]["num_chargers"])
        self.grid_limit = float(config["grid"]["transformer_limit_kw"])
        self.dt = float(config["simulation"].get("dt_hours", 0.25))
        self.horizon = int(config["simulation"].get("planning_horizon_steps", 24))
        self.optimizer = TOUOptimizer(transformer_cap=self.grid_limit, dt_hours=self.dt)
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

        # Remove completed from connected set
        for vid, v in list(self.active_vehicles.items()):
            if v.energy_needed <= 0.05 or v.departure_time_idx <= time_idx:
                self.connected_ids.discard(vid)

        # FIFO non-preemptive admission
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
                "is_priority": False,  # No priority awareness in MPC-only
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
    """
    Offline Optimal (Oracle Upper Bound):
    Solves a single full-horizon convex QP (T = 96 steps) with perfect foresight of all arrivals,
    departures, mission requirements, and realized PV/BESS profiles.
    """
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

        # Aggregate power limit at each step t: both site limit P_site_max and M * max_evse_kw
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
            weights.append(8.0 if v.is_priority else 0.4)

        delivered = cp.sum(x, axis=1) * self.dt
        targets_arr = np.array(targets, dtype=float)
        weights_arr = np.array(weights, dtype=float)

        H_EC = cp.sum(cp.multiply(tou_prices, P_agg - (pv_actual + bess_actual))) * self.dt
        H_NC = cp.sum(cp.multiply(weights_arr, cp.square(delivered - targets_arr)))
        H_QC = cp.sum_squares(P_agg)

        prob = cp.Problem(cp.Minimize(1.0 * H_EC + 120.0 * H_NC + 0.005 * H_QC), constraints)
        for solver in [cp.MOSEK, cp.CLARABEL, cp.OSQP]:
            if solver in cp.installed_solvers():
                try:
                    prob.solve(solver=solver, verbose=False)
                    if x.value is not None and np.all(np.isfinite(x.value)):
                        break
                except Exception:
                    continue

        x_val = np.clip(np.asarray(x.value, dtype=float), 0.0, None) if (x.value is not None and np.all(np.isfinite(x.value))) else np.zeros((N, T))
        schedules = {}
        for i, v in enumerate(vehicles):
            schedules[v.id] = x_val[i, :]
            v.energy_delivered = float(np.sum(x_val[i, :]) * self.dt)
            v.energy_needed = max(0.0, v.target_energy_kwh - v.energy_delivered)
        return schedules
