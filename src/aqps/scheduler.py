"""
AQPS scheduler variants (revision v3).

  AQPS-static : day-ahead flags only; Layer 1 Tier 1 only; Layer 2 every step.
  AQPS-RT     : Layer 0 + Layer 1 (Tiers 1-4) + Layer 2 at EVERY step (time-triggered).
  AQPS-RT-ET  : same layers, run only on events (table below) + operator preview on injections.

Event routing for AQPS-RT-ET:
  event              Layer 0   Layer 1   Layer 2
  arrival             yes       yes      re-solve
  departure/finish    no        yes      re-solve
  operator injection  yes       yes      re-solve (+ preview)
  predicted shortfall yes       yes      re-solve   (connected priority EV under-planned, or a
                                                    waiting priority EV's mission slack <= delta)
  PV deviation        no        no       re-solve
  periodic T_max      no        no       re-solve
Without an event the next column of the cached plan is applied.
If hard Eq. 10 is infeasible: RT modes re-run Layer 0 + Layer 1 once; if still infeasible (always
the case for static) Layer 2 is solved in the logged soft mode (Eq. 10 moved into the penalty).
The scheduler never changes vehicle energy; the runner applies commands.
"""

from typing import Dict, List, Optional, Set

import numpy as np

from .data_structures import Vehicle
from .impact_preview import ImpactPreview
from .priority_selection import PrioritySelector
from .queue_manager import QueueManager
from .tou_optimization import TOUOptimizer, OK

MODES = ("AQPS-static", "AQPS-RT", "AQPS-RT-ET")


def new_stats() -> Dict:
    return {"l0_solves": 0, "l2_solves": 0, "preview_solves": 0, "l2_ms": [], "all_ms": 0.0,
            "soft_solves": 0, "infeasible_first": 0, "committed_infeasible": 0,
            "preempted": [], "safe_released": [], "rotated": [], "escalation_new": [],
            "event_reasons": [], "preview": None, "status": []}


class AQPSScheduler:
    def __init__(self, cfg: Dict, n_evse: int, site_kw: float, mode: str):
        if mode not in MODES:
            raise ValueError(mode)
        o, l0, l1, et = cfg["optimizer"], cfg["layer0"], cfg["layer1"], cfg["event_trigger"]
        self.mode = mode
        self.dt = cfg["time"]["step_minutes"] / 60.0
        self.opt = TOUOptimizer(site_kw, self.dt, o["alpha1"], o["alpha2"], o["alpha3"],
                                o["x_min_priority_kw"], o["solver"], o["mission_soft_weight"])
        self.sel = PrioritySelector(self.opt, n_evse, self.dt, l0["k_atc"], l0["w_min_candidate"],
                                    l0["feas_tol_kwh"])
        self.qm = QueueManager(n_evse, self.dt, l1["rotation_margin"], l1.get("preempt_slack_h"))
        self.preview = ImpactPreview(self.opt, self.dt)
        self.t_max = int(et["t_max_steps"])
        self.pv_thr = float(et["pv_dev_threshold_kw"])
        self.preview_on_injection = bool(et["preview_on_injection"])
        self.preview_on_escalation = bool(et["preview_on_escalation"])
        self.site_kw = float(site_kw)
        self.plan: Dict[str, np.ndarray] = {}
        self.plan_t0 = -1
        self.plan_H = 0
        self.last_solve = -10 ** 9
        self.pv_at_solve: Optional[np.ndarray] = None
        self.esc_open: Set[str] = set()

    # ------------------------------------------------------------------
    def _events(self, t, present, arrivals, departures, pv_act_now, injection) -> List[str]:
        r = []
        conn = self.qm.connected
        if arrivals:
            r.append("arrival")
        if any(d in conn for d in departures) or any(v.done for v in present if v.id in conn):
            r.append("departure")
        if injection is not None:
            r.append("injection")
        off = t - self.plan_t0
        if not self.plan or off >= self.plan_H:
            r.append("replan")
        else:
            for v in present:
                if v.id in conn and v.is_priority and v.mission_remaining > 0.05:
                    planned = float(np.sum(self.plan.get(v.id, np.zeros(0))[off:])) * self.dt
                    if planned < v.mission_remaining - 0.05:
                        r.append("shortfall")
                        break
            # a waiting priority EV whose mission slack reached the preemption threshold
            if self.qm.preempt_slack_h is not None and "shortfall" not in r:
                for v in present:
                    if (v.id not in conn and v.is_priority and v.mission_remaining > 0.05
                            and self.qm.mission_slack_h(v, t) <= self.qm.preempt_slack_h):
                        r.append("shortfall")
                        break
            if self.pv_at_solve is not None and off < len(self.pv_at_solve):
                if abs(pv_act_now - float(self.pv_at_solve[off])) > self.pv_thr:
                    r.append("pv_dev")
        if t - self.last_solve >= self.t_max:
            r.append("periodic")
        return r

    def _layer0_1(self, t, present, tou, pv, bess, st, run_l0: bool):
        rt = self.mode in ("AQPS-RT", "AQPS-RT-ET")
        l0_esc: List[str] = []
        if self.mode == "AQPS-static":
            for v in present:
                v.is_priority = v.day_ahead_priority
        elif run_l0:
            res = self.sel.select(present, set(self.qm.connected), t, tou, pv, bess)
            st["l0_solves"] += res.solves
            st["all_ms"] += res.solve_ms
            st["committed_infeasible"] += int(res.committed_infeasible)
            l0_esc = res.escalated_ids
        else:
            self.sel.score(present, t)
        conn, ev = self.qm.assign(t, present, safe_release=rt, rotation=rt, atc_order=rt)
        for k in ("preempted", "safe_released", "rotated"):
            st[k] += ev[k]
        waiting_now = set(l0_esc) | set(ev["waiting_priority"])
        st["escalation_new"] += sorted(waiting_now - self.esc_open)
        self.esc_open = waiting_now
        return conn

    def _solve(self, t, conn: List[Vehicle], tou, pv, bess, soft=False):
        evs = [{"id": v.id, "session_rem": v.session_remaining, "mission_rem": v.mission_remaining,
                "p_max": v.p_max_kw, "dep_idx": v.departure_idx, "is_priority": v.is_priority}
               for v in conn if not v.done]
        return self.opt.solve(t, evs, tou, pv, bess, soft=soft)

    # ------------------------------------------------------------------
    def step(self, t: int, present: List[Vehicle], arrivals: List[str], departures: List[str],
             tou: np.ndarray, pv: np.ndarray, bess: np.ndarray, pv_act_now: float,
             injection: Optional[Dict] = None) -> Dict:
        st = new_stats()
        et = self.mode == "AQPS-RT-ET"
        if et:
            reasons = self._events(t, present, arrivals, departures, pv_act_now, injection)
        else:
            reasons = ["every_step"]
        st["event_reasons"] = reasons
        if et and not reasons:
            return self._apply_plan(t, present, st)

        if not et:
            conn = self._layer0_1(t, present, tou, pv, bess, st, run_l0=self.mode == "AQPS-RT")
        elif set(reasons) & {"arrival", "injection", "shortfall", "departure", "replan"}:
            run_l0 = bool(set(reasons) & {"arrival", "injection", "shortfall"})
            conn = self._layer0_1(t, present, tou, pv, bess, st, run_l0=run_l0)
        else:
            conn = [v for v in present if v.id in self.qm.connected]

        if et and injection is not None and self.preview_on_injection and conn:
            rec = self.preview.evaluate(t, ",".join(reasons), conn, tou, pv, bess,
                                        injection.get("operator_choice", "recommended"))
            st["preview_solves"] += rec.solves
            st["all_ms"] += rec.solve_ms
            st["preview"] = rec
        elif et and st["escalation_new"] and self.preview_on_escalation and conn:
            rec = self.preview.evaluate(t, "escalation", conn, tou, pv, bess, "recommended")
            st["preview_solves"] += rec.solves
            st["all_ms"] += rec.solve_ms
            st["preview"] = rec

        res = self._solve(t, conn, tou, pv, bess)
        self._count(st, res)
        if res.status not in OK and res.status != "empty":
            st["infeasible_first"] += 1
            if self.mode != "AQPS-static":
                conn = self._layer0_1(t, present, tou, pv, bess, st, run_l0=True)
                res = self._solve(t, conn, tou, pv, bess)
                self._count(st, res)
            if res.status not in OK and res.status != "empty":
                res = self._solve(t, conn, tou, pv, bess, soft=True)
                self._count(st, res)
                st["soft_solves"] += 1

        self.plan = res.plan() if res.status in OK else {}
        self.plan_t0, self.plan_H = t, res.horizon
        self.last_solve = t
        self.pv_at_solve = np.asarray(pv, dtype=float).copy()
        return self._apply_plan(t, present, st)

    def _count(self, st, res):
        if res.status == "empty":
            return
        st["l2_solves"] += 1
        st["l2_ms"].append(res.solve_ms)
        st["all_ms"] += res.solve_ms
        st["status"].append(res.status)

    def _apply_plan(self, t, present, st) -> Dict:
        off = t - self.plan_t0
        cmds = {}
        for v in present:
            if v.id not in self.qm.connected or v.done:
                continue
            prof = self.plan.get(v.id)
            p = float(prof[off]) if prof is not None and 0 <= off < len(prof) else 0.0
            cmds[v.id] = max(0.0, min(p, v.session_remaining / self.dt, v.p_max_kw))
        tot = sum(cmds.values())
        if tot > self.site_kw + 1e-6:
            for k in cmds:
                cmds[k] *= self.site_kw / tot
        st["commands"] = cmds
        st["connected"] = sorted(self.qm.connected)
        return st
