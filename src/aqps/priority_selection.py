"""
Layer 0: real-time schedule-derived priority selection with an optimizer feasibility check.

ATC urgency (Eq. 12; Vepsalainen & Morton 1987):
    pi_i(t) = (w_i / p_i) * exp( -max(0, s_i) / (k * pbar) )
    p_i = E_mission_rem / xbar_i  (h, floored at one step dt for numerical safety)
    s_i = (d_i - t) * dt - p_i    (h)
Candidates : present, not finished, not safely released, and roster_priority or w_i >= w_min.
Committed  : connected priority EVs that still need mission energy. They are never demoted by
             Layer 0 (admitted EVs keep their guarantee); they are always part of the test set.
Test       : hard-Eq.-10 Layer-2 QP over committed + the first (M - |committed|) new candidates
             of the ATC-ranked prefix (exact for EVs that can be connected now), plus the
             individual dwell-rate check  E_mission_rem <= xbar * (d - t) * dt  for every new
             candidate in the prefix (necessary condition for queued EVs).
Search     : full candidate set first (1 solve); bisection over the prefix length only if it fails.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Set

import numpy as np

from .data_structures import Vehicle
from .tou_optimization import TOUOptimizer, OK


@dataclass
class L0Result:
    priority_ids: Set[str]
    escalated_ids: List[str]
    solves: int = 0
    solve_ms: float = 0.0
    committed_infeasible: bool = False
    scores: Dict[str, float] = field(default_factory=dict)


class PrioritySelector:
    def __init__(self, optimizer: TOUOptimizer, n_evse: int, dt_h: float, k_atc: float,
                 w_min: float, tol_kwh: float = 0.01):
        self.opt = optimizer
        self.M = int(n_evse)
        self.dt = float(dt_h)
        self.k = float(k_atc)
        self.w_min = float(w_min)
        self.tol = float(tol_kwh)

    # ---------------------------------------------------------------- ATC
    def score(self, vehicles: List[Vehicle], t: int) -> Dict[str, float]:
        ps, ss = {}, {}
        for v in vehicles:
            p = max(self.dt, v.mission_remaining / max(v.p_max_kw, 1e-6))
            s = (v.departure_idx - t) * self.dt - p
            ps[v.id], ss[v.id] = p, s
        if not ps:
            return {}
        pbar = float(np.mean(list(ps.values())))
        out = {}
        for v in vehicles:
            pi = (v.trip_criticality_w / ps[v.id]) * math.exp(-max(0.0, ss[v.id]) / (self.k * pbar))
            v.atc_score = pi
            out[v.id] = pi
        return out

    def is_candidate(self, v: Vehicle) -> bool:
        return (not v.safe_released) and (v.roster_priority or v.trip_criticality_w >= self.w_min)

    # ---------------------------------------------------------- feasibility
    def _dwell_ok(self, v: Vehicle, t: int) -> bool:
        return v.mission_remaining <= v.p_max_kw * max(0, v.departure_idx - t) * self.dt + self.tol

    def _qp_feasible(self, test: List[Vehicle], t, tou, pv, bess):
        evs = [{"id": v.id, "session_rem": v.session_remaining, "mission_rem": v.mission_remaining,
                "p_max": v.p_max_kw, "dep_idx": v.departure_idx, "is_priority": True} for v in test]
        if not evs:
            return True, 0.0, False
        r = self.opt.solve(t, evs, tou, pv, bess)
        ok = r.status in OK and all(sf <= self.tol for sf in r.mission_shortfall.values())
        return ok, r.solve_ms, True

    # ---------------------------------------------------------------- select
    def select(self, present: List[Vehicle], connected_ids: Set[str], t: int,
               tou, pv, bess) -> L0Result:
        cands = [v for v in present if self.is_candidate(v) and not v.done]
        scores = self.score(present, t)
        committed = [v for v in cands if v.id in connected_ids and v.is_priority
                     and v.mission_remaining > self.tol]
        met = [v for v in cands if v.is_priority and v.mission_remaining <= self.tol]
        new = [v for v in cands if v not in committed and v not in met and v.mission_remaining > self.tol]
        new.sort(key=lambda v: (-scores.get(v.id, 0.0), v.arrival_idx, v.id))
        slots = max(0, self.M - len(committed))

        res = L0Result(priority_ids=set(), escalated_ids=[], scores=scores)

        def feasible(q: int) -> bool:
            prefix = new[:q]
            if not all(self._dwell_ok(v, t) for v in prefix):
                return False
            ok, ms, solved = self._qp_feasible(committed + prefix[:slots], t, tou, pv, bess)
            res.solves += int(solved)
            res.solve_ms += ms
            return ok

        best = 0
        if new and feasible(len(new)):
            best = len(new)
        elif new or committed:
            if not feasible(0):
                res.committed_infeasible = True   # committed EVs stay; all new escalate
            else:
                lo, hi = 1, len(new) - 1
                while lo <= hi:
                    mid = (lo + hi) // 2
                    if feasible(mid):
                        best, lo = mid, mid + 1
                    else:
                        hi = mid - 1

        chosen = {v.id for v in committed} | {v.id for v in met} | {v.id for v in new[:best]}
        res.priority_ids = chosen
        res.escalated_ids = [v.id for v in new[best:]]
        for v in present:
            v.is_priority = v.id in chosen
        return res
