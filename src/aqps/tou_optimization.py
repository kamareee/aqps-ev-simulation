"""
Layer 2: receding-horizon convex QP, Eqs. (1)-(10) of the manuscript.

  min  a1 * sum_t C_t * g_t * dt                                   (Eq. 2, net grid import)
     + a2 * ( sum_i (E_sess_i - sum_t x_it dt) )^2                 (Eq. 3, aggregate non-completion)
     + a3 * sum_t ( sum_i x_it )^2                                  (Eq. 4, Tikhonov regulariser)
  s.t. g_t >= 0, g_t >= sum_i x_it - PV_t - BESS_t                  (epigraph of max(0, .) in Eq. 2)
       sum_i x_it <= P_site                                         (Eq. 6; Eq. 5 with L = 1)
       0 <= x_it <= xbar_i inside the dwell window, 0 outside       (Eq. 7)
       x_it >= x_min profile for priority EVs until E_mission covered (Eq. 8)
       sum_t x_it dt <= E_sess_i            for all connected i     (Eq. 9, also battery headroom)
       sum_t x_it dt >= E_mission_i         for priority i (HARD)   (Eq. 10)

Horizon H_k = max_i (d_i - k) over the connected set (reaches every departure).
Decision variables: x (EV powers) and the auxiliary epigraph variable g (net import).
Solver comes from config; there is no silent fallback. If Eq. 10 is infeasible the result
has status 'infeasible' and the caller decides (re-run Layer 0, or the logged soft mode).
"""

import time
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cvxpy as cp
import numpy as np

OK = ("optimal", "optimal_inaccurate")


@dataclass
class OptResult:
    status: str
    ids: List[str]
    schedule: np.ndarray            # (n, H) kW
    horizon: int
    objective: float = float("nan")
    solve_ms: float = 0.0           # wall time incl. CVXPY canonicalisation
    solver_ms: float = 0.0          # solver-reported time
    solver: str = ""
    n_vars: int = 0
    soft: bool = False              # True if Eq. 10 was relaxed into the penalty (logged fallback)
    mission_shortfall: Dict[str, float] = field(default_factory=dict)

    def plan(self) -> Dict[str, np.ndarray]:
        return {vid: self.schedule[i] for i, vid in enumerate(self.ids)}


class TOUOptimizer:
    def __init__(self, site_kw: float, dt_h: float, alpha1: float, alpha2: float, alpha3: float,
                 x_min_pri_kw: float, solver: str = "MOSEK", mission_soft_weight: float = 10.0):
        self.site_kw = float(site_kw)
        self.dt = float(dt_h)
        self.a1, self.a2, self.a3 = float(alpha1), float(alpha2), float(alpha3)
        self.x_min = float(x_min_pri_kw)
        self.solver = solver.upper()
        self.mission_soft_weight = float(mission_soft_weight)
        if self.solver not in cp.installed_solvers():
            raise RuntimeError(f"Solver {self.solver} not installed (installed: {cp.installed_solvers()}). "
                               f"Install/licence it or set optimizer.solver explicitly in config.")

    # ------------------------------------------------------------------
    def solve(self, k: int, evs: List[Dict], tou: np.ndarray, pv: np.ndarray, bess: np.ndarray,
              soft: bool = False, horizon_cap: Optional[int] = None) -> OptResult:
        """evs: dicts with id, session_rem, mission_rem, p_max, dep_idx, is_priority.
        tou/pv/bess are arrays starting at step k."""
        t0 = time.perf_counter()
        ids = [e["id"] for e in evs]
        n = len(evs)
        if n == 0:
            return OptResult("empty", [], np.zeros((0, 0)), 0)
        H = max(1, max(int(e["dep_idx"]) - k for e in evs))
        H = min(H, len(tou)) if horizon_cap is None else min(H, len(tou), horizon_cap)
        dt = self.dt

        X_max = np.zeros((n, H))
        X_min = np.zeros((n, H))
        sess = np.zeros(n)
        miss = np.zeros(n)
        pri = np.zeros(n, dtype=bool)
        for i, e in enumerate(evs):
            w = max(0, min(H, int(e["dep_idx"]) - k))
            X_max[i, :w] = float(e["p_max"])
            sess[i] = max(0.0, float(e["session_rem"]))
            miss[i] = min(sess[i], max(0.0, float(e["mission_rem"])))
            pri[i] = bool(e["is_priority"])
            if pri[i] and miss[i] > 1e-6 and not soft:
                xm = min(self.x_min, float(e["p_max"]))
                n_full = int(np.floor(miss[i] / (xm * dt) + 1e-9))
                n_full_c = min(n_full, w)
                X_min[i, :n_full_c] = xm
                if n_full < w:
                    X_min[i, n_full] = max(0.0, miss[i] - n_full * xm * dt) / dt
        X_min = np.minimum(X_min, X_max)

        tou_h = np.asarray(tou[:H], dtype=float)
        net_offset = np.asarray(pv[:H], dtype=float) + np.asarray(bess[:H], dtype=float)

        x = cp.Variable((n, H))
        g = cp.Variable(H, nonneg=True)
        P = cp.sum(x, axis=0)
        deliv = cp.sum(x, axis=1) * dt
        cons = [x >= X_min, x <= X_max, P <= self.site_kw, deliv <= sess, g >= P - net_offset]
        obj = (self.a1 * cp.sum(cp.multiply(tou_h, g)) * dt
               + self.a2 * cp.square(cp.sum(sess - deliv))
               + self.a3 * cp.sum_squares(P))
        if pri.any():
            if soft:
                short = cp.pos(miss[pri] - deliv[pri])
                obj = obj + self.mission_soft_weight * self.a2 * cp.square(cp.sum(short))
            else:
                cons.append(deliv[pri] >= miss[pri])

        prob = cp.Problem(cp.Minimize(obj), cons)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                prob.solve(solver=self.solver, verbose=False)
                status = prob.status
            except cp.SolverError as exc:
                status = f"solver_error:{exc.__class__.__name__}"
        wall = (time.perf_counter() - t0) * 1000.0
        stats = prob.solver_stats
        res = OptResult(status=status, ids=ids, schedule=np.zeros((n, H)), horizon=H,
                        solve_ms=wall, solver=self.solver, n_vars=n * H + H, soft=soft,
                        solver_ms=(stats.solve_time or 0.0) * 1000.0 if stats else 0.0)
        if status in OK and x.value is not None:
            xv = np.clip(np.asarray(x.value), 0.0, X_max)
            res.schedule = xv
            res.objective = float(prob.value)
            d = xv.sum(axis=1) * dt
            res.mission_shortfall = {ids[i]: float(max(0.0, miss[i] - d[i])) for i in range(n) if pri[i]}
        return res
