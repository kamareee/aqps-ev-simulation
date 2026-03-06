import cvxpy as cp
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple


@dataclass
class OptimizationResult:
    status: str
    schedule: Dict[str, List[float]]  # EV_ID -> [p_t, p_t+1, ...]
    grid_power: List[float]
    total_cost: float


class TOUOptimizer:
    def __init__(self, transformer_cap: float, dt_hours: float = 0.25):
        """
        Args:
            transformer_cap: Max grid import power (kW)
            dt_hours: Length of time step in hours (e.g., 15 min = 0.25)
        """
        self.P_grid_max = transformer_cap
        self.dt = dt_hours

        # Optimization weights
        self.w_cost = 1.0  # Term 1: Net Energy Cost
        self.w_peak = 0.1  # Term 2: Grid Smoothing Regularizer (alpha)
        self.w_slack = 1000.0  # Term 3: Service Penalty Regularizer (gamma)

    def solve(
        self,
        current_time_idx: int,
        horizon: int,
        active_evs: List[Dict],
        tou_prices: np.array,
        pv_generation: np.array,
        bess_generation: np.array,
    ) -> OptimizationResult:
        """
        Solves the Convex QP for the current horizon.

        Args:
            current_time_idx: Current simulation step
            horizon: Number of steps to look ahead
            active_evs: List of dicts {id, energy_needed, max_power, dep_time_idx}
            tou_prices: Array of prices for the horizon
            pv_generation: Array of PV power input for the horizon
            bess_generation: Array of BESS power input for the horizon
        """

        # 1. Setup Dimensions
        num_evs = len(active_evs)
        steps = min(horizon, len(tou_prices))

        if steps == 0:
            return self._empty_result()

        # Truncate external data to horizon
        prices = tou_prices[:steps]
        pv = pv_generation[:steps]

        # Handle BESS input - ensure it matches dimension or pad with 0
        if bess_generation is None or len(bess_generation) == 0:
            bess = np.zeros(steps)
        else:
            bess = bess_generation[:steps]
            if len(bess) < steps:
                bess = np.pad(bess, (0, steps - len(bess)), "constant")

        # ==========================================
        # GUARD CLAUSE: No EVs present
        # ==========================================
        if num_evs == 0:
            # If no EVs are charging, the grid simply balances PV and BESS.
            # Grid + PV + BESS = Sum(EVs) -> Grid = - (PV + BESS)
            grid_power = -(pv + bess)

            # Optionally calculate economic cost/profit of exporting this energy
            economic_cost = np.sum(prices * grid_power) * self.w_cost

            return OptimizationResult(
                status="optimal_empty",
                schedule={},
                grid_power=grid_power.tolist(),
                total_cost=float(economic_cost),
            )
        # ==========================================

        # 2. Define Variables
        # Power delivered to each EV [num_evs, steps]
        P_ev = cp.Variable((num_evs, steps), nonneg=True)

        # Grid power (Import positive)
        P_grid = cp.Variable(steps)

        # Slack variables for unmet energy (to ensure feasibility)
        Unmet_Energy = cp.Variable(num_evs, nonneg=True)

        # 3. Constraints
        constraints = []

        # 3.1 System Power Balance: Grid + PV + BESS = Sum(EVs)
        constraints += [P_grid + pv + bess == cp.sum(P_ev, axis=0)]

        # 3.2 Grid Limits
        constraints += [P_grid <= self.P_grid_max]

        # 3.3 EV Constraints
        for i, ev in enumerate(active_evs):
            # Max charging power limit (Hardware limit)
            constraints += [P_ev[i, :] <= ev["max_power"]]

            # Departure Constraint
            dep_relative = ev["dep_time_idx"] - current_time_idx

            if dep_relative <= 0:
                constraints += [P_ev[i, :] == 0]
            else:
                charge_window = min(steps, dep_relative)

                # Total Energy Delivered >= Required - Slack
                constraints += [
                    cp.sum(P_ev[i, :charge_window]) * self.dt
                    >= ev["energy_needed"] - Unmet_Energy[i]
                ]

                # Force 0 power after departure
                if charge_window < steps:
                    constraints += [P_ev[i, charge_window:] == 0]

        # 4. Objective Function
        # J = Cost + Alpha * P_grid^2 + Gamma * Slack

        # Term 1: Economic Cost (Net Energy Cost)
        economic_cost = self.w_cost * cp.sum(cp.multiply(prices, P_grid))

        # Term 2: Peak Shaving Regularizer
        peak_cost = self.w_peak * cp.sum_squares(P_grid)

        # Term 3: Service Penalty Regularizer
        slack_cost = self.w_slack * cp.sum(Unmet_Energy)

        objective = cp.Minimize(economic_cost + peak_cost + slack_cost)

        # 5. Solve
        prob = cp.Problem(objective, constraints)

        try:
            if "MOSEK" in cp.installed_solvers():
                prob.solve(solver=cp.MOSEK, verbose=False)
            else:
                # Fallback to OSQP if MOSEK is not installed
                prob.solve(solver=cp.OSQP, verbose=False)
        except cp.SolverError:
            prob.solve(solver=cp.ECOS, verbose=False)

        if prob.status not in ["optimal", "optimal_inaccurate"]:
            print(f"Warning: Optimization status: {prob.status}")
            return self._fallback_result(steps, num_evs)

        # 6. Format Results
        schedule = {}
        for i, ev in enumerate(active_evs):
            schedule[ev["id"]] = P_ev[i, :].value.tolist()

        return OptimizationResult(
            status=prob.status,
            schedule=schedule,
            grid_power=P_grid.value.tolist(),
            total_cost=prob.value,
        )

    def _empty_result(self):
        return OptimizationResult("empty", {}, [], 0.0)

    def _fallback_result(self, steps, num_evs):
        return OptimizationResult(
            "failure",
            {f"dummy_{i}": [0.0] * steps for i in range(num_evs)},
            [0.0] * steps,
            0.0,
        )
