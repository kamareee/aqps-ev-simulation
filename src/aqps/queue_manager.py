"""
Layer 1: EVSE admission with safe release (revision v3).

Queues:  Q_act (connected priority EVs, FIFO by admission), Q_p (waiting priority EVs, ATC order),
         Q_n (non-priority EVs, FIFO by arrival).
Just-in-time preemption (config layer1.preempt_slack_h = delta): a waiting priority EV whose mission
slack (d - t) dt - E_mission_rem / xbar is still > delta waits for a naturally freed EVSE; Tiers 1-3
act only once its slack reaches delta (delta = null gives eager preemption, as in the submitted method).
When a waiting priority EV needs an EVSE and all M are occupied:
  Tier 1  Non-priority preemption: disconnect the connected non-priority EV with the largest laxity.
  Tier 2  Safe release (automatic): disconnect the connected priority EV that satisfies all of
          (a) E_curr >= E_trip + E_ret + r   and   E_mission met (Eq. 10 quantity)
          (b) E_further_today <= E_curr - (E_trip + E_ret + r)
          (c) xbar * (d_next_day - a_return) * dt >= E_next_day - surplus   (skipped for overnight EVs)
          choosing the largest surplus, then the latest departure. It leaves with >= mission energy.
  Tier 3  Guarded FIFO rotation (SIMULATED OPERATOR DECISION, stands in for the operator preview):
          from the head of Q_act, rotate out the first EV j with pi_new > margin * pi_j and
          (E_curr_j >= E_trip + E_ret + r  or  laxity_j > dt). If j has met its mission it becomes
          non-priority; otherwise it returns to Q_p with its priority.
  Tier 4  Otherwise disconnect nobody; the incoming EV waits and an escalation episode is opened.
Proposition 2 (no disconnection below mission energy) covers Tiers 1-2 (automatic tiers).
Rotations are counted separately; their effect appears in the mission-sufficiency metric.
"""

from typing import Dict, List, Set, Tuple

from .data_structures import Vehicle

MISSION_TOL_KWH = 1e-3  # safe release requires E_mission met (same quantity as Eq. 10)


class QueueManager:
    def __init__(self, n_evse: int, dt_h: float, rotation_margin: float = 1.02,
                 preempt_slack_h: float = None):
        self.preempt_slack_h = preempt_slack_h  # None = eager preemption (submitted design)
        self.M = int(n_evse)
        self.dt = float(dt_h)
        self.margin = float(rotation_margin)
        self.connected: Set[str] = set()
        self.q_act: List[str] = []
        self.open_escalations: Set[str] = set()

    # ---------------------------------------------------------------- helpers
    def laxity(self, v: Vehicle, t: int) -> float:
        rem_h = (v.departure_idx - t) * self.dt
        v.laxity_h = rem_h - v.session_remaining / max(v.p_max_kw, 1e-6)
        return v.laxity_h

    def mission_slack_h(self, v: Vehicle, t: int) -> float:
        return (v.departure_idx - t) * self.dt - v.mission_remaining / max(v.p_max_kw, 1e-6)

    def safe_release_ok(self, v: Vehicle) -> Tuple[bool, float]:
        surplus = v.current_energy_kwh - v.mission_level_kwh
        if surplus < -1e-6 or v.mission_remaining > MISSION_TOL_KWH:   # (a), physical level and Eq. 10 target
            return False, surplus
        if v.further_trips_today_kwh > surplus + 1e-6:         # (b)
            return False, surplus
        if not v.overnight:                                    # (c)
            window = max(0, v.next_day_dep_idx - v.evening_return_idx) * self.dt * v.p_max_kw
            if window < v.next_day_req_kwh - surplus:
                return False, surplus
        return True, surplus

    def drop(self, vid: str):
        self.connected.discard(vid)
        if vid in self.q_act:
            self.q_act.remove(vid)

    # ---------------------------------------------------------------- assign
    def assign(self, t: int, present: List[Vehicle], safe_release: bool, rotation: bool,
               atc_order: bool) -> Tuple[List[Vehicle], Dict[str, List[str]]]:
        ev = {"preempted": [], "safe_released": [], "rotated": [], "escalation_new": [],
              "waiting_priority": [], "deferred_priority": []}
        active = {v.id: v for v in present if not v.done and v.departure_idx > t}
        for vid in list(self.connected):
            if vid not in active:
                self.drop(vid)
        for v in active.values():
            self.laxity(v, t)

        self.q_act = [vid for vid in self.q_act if vid in self.connected and active[vid].is_priority]
        for vid in sorted(self.connected):
            if active[vid].is_priority and vid not in self.q_act:
                self.q_act.append(vid)

        waiting = [v for v in active.values() if v.is_priority and v.id not in self.connected]
        if atc_order:
            waiting.sort(key=lambda v: (-v.atc_score, v.arrival_idx, v.id))
        else:
            waiting.sort(key=lambda v: (v.arrival_idx, v.id))

        for p in waiting:
            if len(self.connected) < self.M:
                self._admit(p)
                continue
            if self.preempt_slack_h is not None:
                if self.mission_slack_h(p, t) > self.preempt_slack_h:
                    ev["deferred_priority"].append(p.id)  # enough slack: wait for a natural free EVSE
                    continue
            conn_np = [active[vid] for vid in self.connected if not active[vid].is_priority]
            if conn_np:                                                    # Tier 1
                victim = max(conn_np, key=lambda v: (v.laxity_h, v.id))
                self.drop(victim.id)
                victim.preempted_count += 1
                ev["preempted"].append(victim.id)
                self._admit(p)
                continue
            if safe_release:                                               # Tier 2
                rel = []
                for vid in self.q_act:
                    ok, surplus = self.safe_release_ok(active[vid])
                    if ok:
                        rel.append((surplus, active[vid].departure_idx, vid))
                if rel:
                    _, _, vid = max(rel)
                    v = active[vid]
                    self.drop(vid)
                    v.safe_released = True
                    v.is_priority = False
                    ev["safe_released"].append(vid)
                    self._admit(p)
                    continue
            if rotation:                                                   # Tier 3 (simulated operator)
                victim = None
                for vid in list(self.q_act):
                    h = active[vid]
                    guard = (h.current_energy_kwh >= h.mission_level_kwh - 1e-6) or (h.laxity_h > self.dt)
                    if p.atc_score > self.margin * h.atc_score and guard:
                        victim = h
                        break
                if victim is not None:
                    self.drop(victim.id)
                    victim.rotated_count += 1
                    if victim.mission_remaining <= 1e-6:
                        victim.is_priority = False
                    ev["rotated"].append(victim.id)
                    self._admit(p)
                    continue
            if p.id not in self.open_escalations:                         # Tier 4
                self.open_escalations.add(p.id)
                ev["escalation_new"].append(p.id)
            ev["waiting_priority"].append(p.id)

        # close escalation episodes for EVs now connected, departed, or no longer priority
        self.open_escalations = {vid for vid in self.open_escalations
                                 if vid in active and vid not in self.connected and active[vid].is_priority}

        for v in sorted((v for v in active.values() if not v.is_priority and v.id not in self.connected),
                        key=lambda v: (v.arrival_idx, v.id)):
            if len(self.connected) >= self.M:
                break
            self.connected.add(v.id)

        return [active[vid] for vid in sorted(self.connected)], ev

    def _admit(self, v: Vehicle):
        self.connected.add(v.id)
        if v.id not in self.q_act:
            self.q_act.append(v.id)
