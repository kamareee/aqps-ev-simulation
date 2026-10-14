"""
Layer 2: Convex Quadratic Programming (QP) Optimizer (Eqs. 1-10).

Code Invariants Enforced:
- The only decision variable is x_{i,t} (shape: num_evs x steps). P_grid is notation only.
- H^QC is a Tikhonov conditioning term on aggregate EV charging load.
- No slack variables. H^NC is a direct squared-norm non-completion penalty.
- Eq. 6 site limit applies to gross aggregate EV charging power: sum_i x_{i,t} <= P_site_max.
"""

import time
import warnings
import cvxpy as cp
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class OptimizationResult:
    status: str
    schedule: Dict[str, List[float]]       # EV_ID -> [x_{i,k}, x_{i,k+1}, ...]
    grid_power: List[float]                # Net grid import [max(0, sum_i x_{i,t} - PV_t - BESS_t)]
    gross_ev_power: List[float]            # Gross EV power [sum_i x_{i,t}]
    predicted_shortfalls: Dict[str, float] # EV_ID -> max(0, E_req_horizon - delivered_horizon)
    total_cost: float
    solve_time_ms: float = 0.0


class TOUOptimizer:
    def __init__(
        self,
        transformer_cap: float,
        dt_hours: float = 0.25,
        alpha1: float = 1.0,
        alpha2: float = 100.0,
        alpha3: float = 0.005,
        min_priority_kw: float = 3.84,
    ):
        self.P_site_max = float(transformer_cap)
        self.dt = float(dt_hours)
        self.alpha1 = float(alpha1)
        self.alpha2 = float(alpha2)
        self.alpha3 = float(alpha3)
        self.min_priority_kw = float(min_priority_kw)

    def solve(
        self,
        current_time_idx: int,
        horizon: int,
        active_evs: List[Dict],
        tou_prices: np.ndarray,
        pv_generation: np.ndarray,
        bess_generation: Optional[np.ndarray] = None,
        enforce_priority_min: bool = True,
    ) -> OptimizationResult:
        """
        Solves Eqs. (1)-(10) over horizon H using MOSEK (primary), CLARABEL, or OSQP.
        Vectorized bounds ensure strict interior feasibility (Slater's condition) and zero NaN reductions.
        """
        t0 = time.perf_counter()
        num_evs = len(active_evs)
        steps = min(int(horizon), len(tou_prices))

        if steps <= 0 or num_evs == 0:
            return OptimizationResult(
                status="optimal_empty",
                schedule={},
                grid_power=[0.0] * max(0, steps),
                gross_ev_power=[0.0] * max(0, steps),
                predicted_shortfalls={},
                total_cost=0.0,
                solve_time_ms=(time.perf_counter() - t0) * 1000.0,
            )

        prices = np.asarray(tou_prices[:steps], dtype=float)
        pv = np.asarray(pv_generation[:steps], dtype=float)
        if bess_generation is None or len(bess_generation) == 0:
            bess = np.zeros(steps, dtype=float)
        else:
            bess = np.asarray(bess_generation[:steps], dtype=float)
            if len(bess) < steps:
                bess = np.pad(bess, (0, steps - len(bess)), "constant")

        # Build vectorized lower and upper bound matrices X_min, X_max of shape (num_evs, steps)
        X_min = np.zeros((num_evs, steps), dtype=float)
        X_max = np.zeros((num_evs, steps), dtype=float)
        horizon_targets = np.zeros(num_evs, dtype=float)
        priority_weights = np.ones(num_evs, dtype=float)
        e_rem_vec = np.zeros(num_evs, dtype=float)

        for i, ev in enumerate(active_evs):
            p_max = float(ev["max_power"])
            e_rem = max(0.0, float(ev["energy_needed"]))
            e_rem_vec[i] = e_rem
            dep_rel = int(ev["dep_time_idx"] - current_time_idx)
            is_pri = bool(ev.get("is_priority", False))

            if dep_rel <= 0 or e_rem <= 1e-3:
                horizon_targets[i] = 0.0
                priority_weights[i] = 1.0
                continue

            w = min(steps, dep_rel)
            X_max[i, :w] = p_max
            max_deliv_h = p_max * w * self.dt

            if dep_rel <= steps:
                e_target_h = min(e_rem, max_deliv_h)
            else:
                max_after_h = p_max * max(0, dep_rel - steps) * self.dt
                min_needed_in_h = max(0.0, e_rem - max_after_h)
                if is_pri:
                    e_target_h = min(e_rem, max_deliv_h)
                else:
                    e_target_h = min(e_rem, max(min_needed_in_h, e_rem * (w / dep_rel)))

            horizon_targets[i] = e_target_h
            priority_weights[i] = 5.0 if is_pri else 0.35

            if enforce_priority_min and is_pri and e_rem > 0.1:
                avg_rate_needed = e_rem / (dep_rel * self.dt)
                x_min_val = min(self.min_priority_kw, p_max, avg_rate_needed)
                if x_min_val > 0.05:
                    X_min[i, :w] = x_min_val

        # Ensure column sums of X_min never exceed 85% of P_site_max (guarantees strict interior for solver)
        col_sums = np.sum(X_min, axis=0)
        for t_step in range(steps):
            if col_sums[t_step] > 0.85 * self.P_site_max:
                X_min[:, t_step] *= (0.85 * self.P_site_max) / col_sums[t_step]

        # Ensure row energy of X_min never exceeds e_rem_vec
        row_e_min = np.sum(X_min, axis=1) * self.dt
        for i in range(num_evs):
            if row_e_min[i] > 0.95 * e_rem_vec[i] and row_e_min[i] > 0:
                X_min[i, :] *= (0.95 * e_rem_vec[i]) / row_e_min[i]

        # Decision variable ONLY: x_{i,t} (kW)
        x = cp.Variable((num_evs, steps))
        P_agg = cp.sum(x, axis=0)
        delivered_per_ev = cp.sum(x, axis=1) * self.dt

        # Vectorized Constraints (Eqs. 6-9)
        constraints = [
            x >= X_min,
            x <= X_max,
            P_agg <= self.P_site_max,
            delivered_per_ev <= e_rem_vec,
        ]

        # Objective Function (Eqs. 1-4)
        net_grid_expr = P_agg - (pv + bess)
        H_EC = cp.sum(cp.multiply(prices, net_grid_expr)) * self.dt
        H_NC = cp.sum(cp.multiply(priority_weights, cp.square(delivered_per_ev - horizon_targets)))
        H_QC = cp.sum_squares(P_agg)

        objective = cp.Minimize(
            self.alpha1 * H_EC + self.alpha2 * H_NC + self.alpha3 * H_QC
        )

        prob = cp.Problem(objective, constraints)
        solved = False
        installed = cp.installed_solvers()

        # Priority order: MOSEK -> CLARABEL -> OSQP
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for solver in [cp.MOSEK, cp.CLARABEL, cp.OSQP]:
                if solver in installed:
                    try:
                        if solver == cp.MOSEK:
                            prob.solve(solver=cp.MOSEK, verbose=False)
                        elif solver == cp.CLARABEL:
                            prob.solve(solver=cp.CLARABEL, verbose=False)
                        else:
                            prob.solve(
                                solver=cp.OSQP,
                                verbose=False,
                                eps_abs=1e-4,
                                eps_rel=1e-4,
                                max_iter=15000,
                                polishing=True,
                            )
                        if (
                            prob.status in ["optimal", "optimal_inaccurate"]
                            and x.value is not None
                            and np.all(np.isfinite(x.value))
                        ):
                            solved = True
                            break
                    except Exception:
                        continue

        solve_ms = (time.perf_counter() - t0) * 1000.0

        if not solved or x.value is None or not np.all(np.isfinite(x.value)):
            return self._fallback_result(current_time_idx, steps, active_evs, pv, bess, prices, solve_ms)

        x_val = np.clip(np.asarray(x.value, dtype=float), 0.0, X_max)
        schedule = {}
        shortfalls = {}
        for i, ev in enumerate(active_evs):
            vid = ev["id"]
            schedule[vid] = x_val[i, :].tolist()
            deliv_h = float(np.sum(x_val[i, :]) * self.dt)
            dep_rel = max(0, int(ev["dep_time_idx"] - current_time_idx))
            req_check = float(ev["energy_needed"]) if dep_rel <= steps else float(horizon_targets[i])
            shortfalls[vid] = max(0.0, req_check - deliv_h)

        gross_p = np.sum(x_val, axis=0)
        net_grid = np.maximum(0.0, gross_p - (pv + bess))
        realized_h_cost = float(np.sum(net_grid * prices * self.dt))

        return OptimizationResult(
            status=str(prob.status),
            schedule=schedule,
            grid_power=net_grid.tolist(),
            gross_ev_power=gross_p.tolist(),
            predicted_shortfalls=shortfalls,
            total_cost=realized_h_cost,
            solve_time_ms=solve_ms,
        )

    def _fallback_result(self, current_time_idx, steps, active_evs, pv, bess, prices, solve_ms):
        schedule = {}
        shortfalls = {}
        rem_cap = np.full(steps, self.P_site_max, dtype=float)
        sorted_evs = sorted(active_evs, key=lambda e: (not e.get("is_priority", False), e["dep_time_idx"]))
        for ev in sorted_evs:
            vid = ev["id"]
            dep_rel = max(0, int(ev["dep_time_idx"] - current_time_idx))
            w = min(steps, dep_rel)
            prof = np.zeros(steps, dtype=float)
            e_need = float(ev["energy_needed"])
            for t in range(w):
                if e_need <= 1e-3 or rem_cap[t] <= 1e-3:
                    break
                p = min(float(ev["max_power"]), rem_cap[t], e_need / self.dt)
                prof[t] = p
                rem_cap[t] -= p
                e_need -= p * self.dt
            schedule[vid] = prof.tolist()
            shortfalls[vid] = max(0.0, e_need)

        gross_p = self.P_site_max - rem_cap
        net_grid = np.maximum(0.0, gross_p - (pv + bess))
        cost = float(np.sum(net_grid * prices * self.dt))
        return OptimizationResult(
            status="fallback_heuristic",
            schedule=schedule,
            grid_power=net_grid.tolist(),
            gross_ev_power=gross_p.tolist(),
            predicted_shortfalls=shortfalls,
            total_cost=cost,
            solve_time_ms=solve_ms,
        )
