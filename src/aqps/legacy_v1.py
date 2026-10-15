"""
AQPS-v1: the method submitted to J. Energy Storage (EST-D-26-06112), ported to the v3 simulator.

Faithful to commit 0c33183 (src/aqps/queue_manager.py, tou_optimization.py, scheduler.py) except:
  * the energy double-count bug is removed (the runner applies commands once, for every method);
  * the gross site limit sum_i x_it <= P_site is added so that every method faces the same network
    constraint (plan hard constraint 4). The original net-import limit P_grid <= P_site is kept.
Layer 1: sort by (operator flag, arrival), keep top M every step (EVs stay plugged until departure),
         drop lowest-ranked while sum of priority minimum rates > site + PV + BESS.
Layer 2: soft QP, 12-step horizon: min sum(price*P_grid) + 0.1*sum(P_grid^2) + 1000*sum(slack).
"""

import time
import warnings
from typing import Dict, List

import cvxpy as cp

from .data_structures import Vehicle


class AQPSv1Scheduler:
    def __init__(self, n_evse: int, site_kw: float, dt_h: float, solver: str, horizon: int = 12):
        self.M, self.site, self.dt, self.H = int(n_evse), float(site_kw), float(dt_h), int(horizon)
        self.solver = solver.upper()

    def _min_rate(self, v: Vehicle, t: int) -> float:
        rem_h = (v.departure_idx - t) * self.dt
        if rem_h <= 0:
            return v.p_max_kw
        return min(v.p_max_kw, v.session_remaining / rem_h)

    def step(self, t: int, present: List[Vehicle], tou, pv, bess) -> Dict:
        for v in present:
            v.is_priority = v.day_ahead_priority
        ranked = sorted(present, key=lambda v: (0 if v.is_priority else 1, v.arrival_idx, v.id))
        cand = ranked[: self.M]
        cap = self.site + float(pv[0]) + float(bess[0])
        while cand and sum(self._min_rate(v, t) for v in cand if v.is_priority) > cap:
            cand.pop()
        active = [v for v in cand if v.session_remaining > 1e-3]
        if not active:
            return {"commands": {}, "connected": [v.id for v in cand], "solves": 0, "solve_ms": 0.0,
                    "status": "empty"}

        H = min(self.H, len(tou))
        n = len(active)
        P_ev = cp.Variable((n, H), nonneg=True)
        P_grid = cp.Variable(H)
        slack = cp.Variable(n, nonneg=True)
        cons = [P_grid + pv[:H] + bess[:H] == cp.sum(P_ev, axis=0), P_grid <= self.site,
                cp.sum(P_ev, axis=0) <= self.site]
        for i, v in enumerate(active):
            cons.append(P_ev[i, :] <= v.p_max_kw)
            w = min(H, v.departure_idx - t)
            if w <= 0:
                cons.append(P_ev[i, :] == 0)
                continue
            cons.append(cp.sum(P_ev[i, :w]) * self.dt >= v.session_remaining - slack[i])
            if w < H:
                cons.append(P_ev[i, w:] == 0)
        obj = cp.sum(cp.multiply(tou[:H], P_grid)) + 0.1 * cp.sum_squares(P_grid) + 1000 * cp.sum(slack)
        prob = cp.Problem(cp.Minimize(obj), cons)
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                prob.solve(solver=self.solver, verbose=False)
                status = prob.status
            except cp.SolverError as exc:
                status = f"solver_error:{exc.__class__.__name__}"
        ms = (time.perf_counter() - t0) * 1000.0
        cmds = {}
        if status in ("optimal", "optimal_inaccurate") and P_ev.value is not None:
            for i, v in enumerate(active):
                cmds[v.id] = float(max(0.0, P_ev.value[i, 0]))
        return {"commands": cmds, "connected": [v.id for v in cand], "solves": 1, "solve_ms": ms,
                "status": status}
