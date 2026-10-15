"""
Operator impact preview (Parasuraman, Sheridan & Wickens 2000, "executes unless vetoed").

Three what-if Layer-2 solves over the connected set:
  recommended      : the Layer-0/1 priority set
  keep_all_urgent  : additionally promote the most urgent connected non-priority EV (business continuity)
  downgrade        : demote the least urgent priority EV whose mission is already met (or the least urgent)
The recommended option is applied provisionally; the scripted operator choice then replaces it.
Reported per option: horizon energy cost ($) and expected energy at departure for affected EVs.
"""

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from .data_structures import Vehicle
from .tou_optimization import TOUOptimizer, OK


@dataclass
class PreviewRecord:
    t: int
    trigger: str
    options: List[Dict] = field(default_factory=list)
    chosen: str = "recommended"
    extra_cost: float = 0.0
    solves: int = 0
    solve_ms: float = 0.0


class ImpactPreview:
    def __init__(self, optimizer: TOUOptimizer, dt_h: float):
        self.opt = optimizer
        self.dt = float(dt_h)
        self.log: List[PreviewRecord] = []

    def _cost(self, sched: np.ndarray, tou, pv, bess) -> float:
        H = sched.shape[1] if sched.size else 0
        if H == 0:
            return 0.0
        P = sched.sum(axis=0)
        g = np.maximum(0.0, P - np.asarray(pv[:H]) - np.asarray(bess[:H]))
        return float(np.sum(np.asarray(tou[:H]) * g) * self.dt)

    def _what_if(self, name, t, conn: List[Vehicle], pri: set, tou, pv, bess, rec: PreviewRecord):
        evs = [{"id": v.id, "session_rem": v.session_remaining, "mission_rem": v.mission_remaining,
                "p_max": v.p_max_kw, "dep_idx": v.departure_idx, "is_priority": v.id in pri} for v in conn]
        r = self.opt.solve(t, evs, tou, pv, bess)
        soft = False
        if r.status not in OK:
            r = self.opt.solve(t, evs, tou, pv, bess, soft=True)
            soft = True
            rec.solves += 1
            rec.solve_ms += r.solve_ms
        rec.solves += 1
        rec.solve_ms += r.solve_ms
        plan = r.plan() if r.status in OK else {}
        dep_e = {v.id: round(v.current_energy_kwh + float(np.sum(plan.get(v.id, [0.0]))) * self.dt, 2)
                 for v in conn}
        cost = self._cost(r.schedule, tou, pv, bess) if r.status in OK else float("nan")
        return {"option": name, "priority_ids": sorted(pri), "horizon_cost": round(cost, 3),
                "soft": soft, "departure_energy_kwh": dep_e}

    def evaluate(self, t: int, trigger: str, conn: List[Vehicle], tou, pv, bess,
                 choice: str = "recommended") -> PreviewRecord:
        rec = PreviewRecord(t=t, trigger=trigger)
        base = {v.id for v in conn if v.is_priority}
        np_cands = sorted((v for v in conn if not v.is_priority and v.mission_remaining > 0),
                          key=lambda v: (-v.atc_score, v.id))
        keep = set(base) | ({np_cands[0].id} if np_cands else set())
        pri_sorted = sorted((v for v in conn if v.is_priority),
                            key=lambda v: (v.mission_remaining > 1e-6, v.atc_score, v.id))
        down = set(base) - ({pri_sorted[0].id} if pri_sorted else set())
        opts = [self._what_if("recommended", t, conn, base, tou, pv, bess, rec),
                self._what_if("keep_all_urgent", t, conn, keep, tou, pv, bess, rec),
                self._what_if("downgrade", t, conn, down, tou, pv, bess, rec)]
        for o in opts:
            o["cost_delta"] = round(o["horizon_cost"] - opts[0]["horizon_cost"], 3)
        rec.options = opts
        chosen = next((o for o in opts if o["option"] == choice), opts[0])
        rec.chosen = chosen["option"]
        rec.extra_cost = max(0.0, chosen["cost_delta"]) if np.isfinite(chosen["cost_delta"]) else 0.0
        chosen_set = set(chosen["priority_ids"])
        for v in conn:
            v.is_priority = v.id in chosen_set
        self.log.append(rec)
        return rec
