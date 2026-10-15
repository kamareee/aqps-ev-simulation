"""
Baselines (revision v3). All use the same EVSE count M and gross site limit P_site.

LLF      : non-preemptive EVSE occupancy (an EV keeps its EVSE until finished or departed);
           free EVSEs filled by least laxity; power allocated greedily in laxity order.
EDF      : as LLF but ordered by departure deadline.
MPC-only : the SAME Layer-2 optimizer and weights as AQPS, no priority set (no Eqs. 8, 10),
           FIFO non-preemptive admission, solved every step.
Offline  : same objective/constraints, perfect information over the whole window, exact
           EVSE-count limit via binary z_it (sum_i z_it <= M, x_it <= xbar_i z_it), Eq. 10 hard
           for roster-priority EVs (soft fallback logged). Eq. 8 is omitted (it only restricts
           the online controller). Solved as MIQP (MOSEK) with a time limit; the continuous
           relaxation is also solved and reported as a bound.
"""

import time
import warnings
from typing import Dict, List

import cvxpy as cp
import numpy as np

from .data_structures import Vehicle
from .tou_optimization import TOUOptimizer, OK


class _Greedy:
    key = None

    def __init__(self, n_evse: int, site_kw: float, dt_h: float):
        self.M, self.site, self.dt = int(n_evse), float(site_kw), float(dt_h)
        self.connected: set = set()

    def _lax(self, v: Vehicle, t: int) -> float:
        return (v.departure_idx - t) * self.dt - v.session_remaining / max(v.p_max_kw, 1e-6)

    def step(self, t: int, present: List[Vehicle]) -> Dict:
        t0 = time.perf_counter()
        active = {v.id: v for v in present if not v.done}
        self.connected &= set(active)
        waiting = sorted((v for vid, v in active.items() if vid not in self.connected),
                         key=lambda v: self.key(v, t))
        for v in waiting:
            if len(self.connected) >= self.M:
                break
            self.connected.add(v.id)
        cap = self.site
        cmds = {}
        for v in sorted((active[i] for i in self.connected), key=lambda v: self.key(v, t)):
            p = max(0.0, min(v.p_max_kw, cap, v.session_remaining / self.dt))
            cmds[v.id] = p
            cap -= p
        return {"commands": cmds, "connected": sorted(self.connected),
                "l2_solves": 0, "l2_ms": [], "all_ms": (time.perf_counter() - t0) * 1000.0}


class LLFScheduler(_Greedy):
    def key(self, v, t):
        return (self._lax(v, t), v.arrival_idx, v.id)


class EDFScheduler(_Greedy):
    def key(self, v, t):
        return (v.departure_idx, v.arrival_idx, v.id)


class MPCOnlyScheduler:
    def __init__(self, cfg: Dict, n_evse: int, site_kw: float):
        o = cfg["optimizer"]
        self.dt = cfg["time"]["step_minutes"] / 60.0
        self.M = int(n_evse)
        self.opt = TOUOptimizer(site_kw, self.dt, o["alpha1"], o["alpha2"], o["alpha3"],
                                o["x_min_priority_kw"], o["solver"], o["mission_soft_weight"])
        self.connected: set = set()

    def step(self, t, present: List[Vehicle], tou, pv, bess) -> Dict:
        active = {v.id: v for v in present if not v.done}
        self.connected &= set(active)
        for v in sorted((v for i, v in active.items() if i not in self.connected),
                        key=lambda v: (v.arrival_idx, v.id)):
            if len(self.connected) >= self.M:
                break
            self.connected.add(v.id)
        conn = [active[i] for i in sorted(self.connected)]
        evs = [{"id": v.id, "session_rem": v.session_remaining, "mission_rem": 0.0,
                "p_max": v.p_max_kw, "dep_idx": v.departure_idx, "is_priority": False} for v in conn]
        res = self.opt.solve(t, evs, tou, pv, bess)
        cmds = {}
        plan = res.plan() if res.status in OK else {}
        for v in conn:
            p = float(plan[v.id][0]) if v.id in plan else 0.0
            cmds[v.id] = max(0.0, min(p, v.session_remaining / self.dt, v.p_max_kw))
        n = 0 if res.status == "empty" else 1
        return {"commands": cmds, "connected": sorted(self.connected), "l2_solves": n,
                "l2_ms": [res.solve_ms] if n else [], "all_ms": res.solve_ms if n else 0.0,
                "status": [res.status] if n else []}


class OfflineOptimal:
    def __init__(self, cfg: Dict, n_evse: int, site_kw: float):
        o, off = cfg["optimizer"], cfg["offline"]
        self.cfg = cfg
        self.dt = cfg["time"]["step_minutes"] / 60.0
        self.M, self.site = int(n_evse), float(site_kw)
        self.a1, self.a2, self.a3 = o["alpha1"], o["alpha2"], o["alpha3"]
        self.solver = o["solver"].upper()
        self.soft_w = o["mission_soft_weight"]
        self.time_limit = float(off["milp_time_limit_s"])
        self.gap = float(off["mip_rel_gap"])

    def _build(self, vehicles: List[Vehicle], tou, pv, bess, integer: bool, soft: bool):
        T, N, dt = len(tou), len(vehicles), self.dt
        mask = np.zeros((N, T))
        for i, v in enumerate(vehicles):
            mask[i, v.arrival_idx:v.departure_idx] = 1.0
        pmax = np.array([v.p_max_kw for v in vehicles])
        sess = np.array([v.e_session_kwh for v in vehicles])
        miss = np.array([min(v.e_mission_kwh, v.e_session_kwh) for v in vehicles])
        pri = np.array([v.roster_priority for v in vehicles])
        x = cp.Variable((N, T), nonneg=True)
        z = cp.Variable((N, T), boolean=True) if integer else cp.Variable((N, T))
        g = cp.Variable(T, nonneg=True)
        P = cp.sum(x, axis=0)
        deliv = cp.sum(x, axis=1) * dt
        cons = [x <= cp.multiply(pmax[:, None], z), z <= mask, P <= self.site,
                cp.sum(z, axis=0) <= self.M, deliv <= sess, g >= P - np.asarray(pv) - np.asarray(bess)]
        if not integer:
            cons.append(z >= 0)
        obj = (self.a1 * cp.sum(cp.multiply(np.asarray(tou), g)) * dt
               + self.a2 * cp.square(cp.sum(sess - deliv)) + self.a3 * cp.sum_squares(P))
        if pri.any():
            if soft:
                obj = obj + self.soft_w * self.a2 * cp.square(cp.sum(cp.pos(miss[pri] - deliv[pri])))
            else:
                cons.append(deliv[pri] >= miss[pri])
        return cp.Problem(cp.Minimize(obj), cons), x

    def _solve(self, prob, integer: bool):
        kw = {}
        if integer and self.solver == "MOSEK":
            kw["mosek_params"] = {"MSK_DPAR_MIO_MAX_TIME": self.time_limit,
                                  "MSK_DPAR_MIO_TOL_REL_GAP": self.gap}
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                prob.solve(solver=self.solver, verbose=False, **kw)
                status = prob.status
            except cp.SolverError as exc:
                status = f"solver_error:{exc.__class__.__name__}"
        return status, (time.perf_counter() - t0) * 1000.0

    def solve(self, vehicles: List[Vehicle], tou, pv, bess, integer: bool = True) -> Dict:
        info = {"integer": integer, "soft": False}
        prob, x = self._build(vehicles, tou, pv, bess, integer, soft=False)
        status, ms = self._solve(prob, integer)
        if status not in OK:
            prob, x = self._build(vehicles, tou, pv, bess, integer, soft=True)
            status, ms2 = self._solve(prob, integer)
            ms += ms2
            info["soft"] = True
        info.update(status=status, solve_ms=ms, objective=float(prob.value) if status in OK else float("nan"))
        if integer and status in OK:
            st = prob.solver_stats
            extra = getattr(st, "extra_stats", None)
            info["mip_gap"] = None
            try:  # MOSEK: relative gap from the task information items
                import mosek
                info["mip_gap"] = float(extra.getdouinf(mosek.dinfitem.mio_obj_rel_gap))
            except Exception:
                pass
        sched = np.clip(np.asarray(x.value), 0.0, None) if status in OK else np.zeros((len(vehicles), len(tou)))
        return {"schedule": {v.id: sched[i] for i, v in enumerate(vehicles)}, "info": info}
