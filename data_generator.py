"""
Scenario + roster generator (revision v3).

Outputs
  simulation_data_{45,90,180}EVs.json   — S1_Baseline, S2_Contention, S3_PeakStress
  data/operator_injections.json         — scripted E2 cases (IDs chosen from the 90-EV S2 data)

Key definitions (all energies in kWh, never % SoC)
  E_session : session target (energy the operator would like added this visit)
  E_mission : next trip + return + reserve - arrival energy  (hard target, Eq. 10, priority EVs)
              always <= E_session
  reserve   : 0.10 x battery capacity
Time base: 120 x 15-min steps from 06:00 AEST (day D) to 12:00 AEST (day D+1).
PV: real AEMO VIC1 reference day from data/aemo_vic1_selected_days.json (forecast + actual).
Contention is NOT tuned to any baseline outcome; the a-priori ratios are written to metadata.

Usage: python data_generator.py [--pv-json data/aemo_vic1_selected_days.json]
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import yaml

CFG = yaml.safe_load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "experiments.yaml")))
DT_H = CFG["time"]["step_minutes"] / 60.0
N_STEPS = CFG["time"]["n_steps"]
START_H = CFG["time"]["start_hour"]
INFRA = {int(k): v for k, v in CFG["fleets"].items()}
SCENARIOS = ["S1_Baseline", "S2_Contention", "S3_PeakStress"]

# Commercial van archetypes (Vales-Alonso & Alcaraz 2026; Fiori & Marzano 2018)
ARCHETYPES = {
    "Light":  {"cap": 45.0, "p_max": 11.0},
    "Medium": {"cap": 68.0, "p_max": 22.0},
    "Heavy":  {"cap": 85.0, "p_max": 22.0},
}


def clock_hour(step: int) -> float:
    return (START_H + step * DT_H) % 24.0


def tou_profile():
    prices = []
    for s in range(N_STEPS):
        h = clock_hour(s)
        price = None
        for tier in ("peak", "shoulder", "off_peak"):
            for a, b in CFG["tariff"][tier]["windows"]:
                if a <= h < b:
                    price = CFG["tariff"][tier]["price"]
        if price is None:
            raise RuntimeError(f"No tariff window covers hour {h}")
        prices.append(price)
    return prices


def bess_profile():
    b = CFG["site"]["bess"]
    c0, c1 = b["charge_window"]
    d0, d1 = b["discharge_window"]
    e_dis = b["capacity_kwh"] * b["round_trip_eff"]
    p_dis = e_dis / (d1 - d0)
    prof = []
    for s in range(N_STEPS):
        h = clock_hour(s)
        day1 = s < (24 - START_H) / DT_H  # only on day D
        if day1 and c0 <= h < c1:
            prof.append(-b["charge_kw"])
        elif day1 and d0 <= h < d1:
            prof.append(round(p_dis, 4))
        else:
            prof.append(0.0)
    return prof


def h2i(hour: float) -> int:
    """Clock hour on day D (or >24 for D+1) -> step index."""
    return int(round((hour - START_H) / DT_H))


class FleetBuilder:
    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self.n = 0

    def ev(self, archetype, arr, dep, efrac, is_pri, overnight=False):
        rng = self.rng
        a = ARCHETYPES[archetype]
        cap = a["cap"]
        arr = int(np.clip(arr, 0, N_STEPS - 9))
        dep = int(np.clip(dep, arr + 8, N_STEPS - 1))
        e_session = round(max(8.0, cap * efrac), 2)
        reserve = round(0.10 * cap, 2)
        eod = round(float(rng.uniform(0.12, 0.25) * cap), 2)
        ret = round(min(8.0, 0.18 * e_session), 2)
        # next trip sized so that the mission need is ~70-85 % of the session target
        trip = round(max(4.0, float(rng.uniform(0.70, 0.85)) * e_session + eod - ret - reserve), 2)
        e_mission = round(min(e_session, max(0.0, trip + ret + reserve - eod)), 2)
        next_day_dep = int(rng.integers(h2i(29), h2i(33) + 1))  # 05:00-09:00 on D+1
        if overnight:
            dep = min(dep, next_day_dep)
        evening_return = int(min(N_STEPS - 1, dep + int(rng.integers(8, 17))))  # 2-4 h trip
        ev = {
            "id": f"EV_{self.n:03d}",
            "arrival_idx": arr,
            "departure_idx": dep,
            "archetype": archetype,
            "battery_cap_kwh": cap,
            "p_max": a["p_max"],
            "eod_soc_kwh": eod,
            "e_session_kwh": e_session,
            "e_mission_kwh": e_mission,
            "next_trip_kwh": trip,
            "return_kwh": ret,
            "reserve_kwh": reserve,
            "further_trips_today_kwh": round(float(rng.uniform(0.0, 4.0)) if (is_pri and not overnight) else 0.0, 2),
            "evening_return_idx": evening_return,
            "next_day_dep_idx": next_day_dep,
            "next_day_req_kwh": round(float(cap * rng.uniform(0.30, 0.55)), 2),
            "trip_criticality_w": round(float(rng.uniform(2.5, 4.5) if is_pri else rng.uniform(0.5, 1.4)), 2),
            "priority": "High" if is_pri else "Low",
            "overnight": bool(overnight),
        }
        self.n += 1
        return ev

    def batch(self, n, archetype, arr_h, dwell_steps, efrac, is_pri, overnight=False):
        out = []
        for _ in range(n):
            arr = int(self.rng.integers(h2i(arr_h[0]), h2i(arr_h[1]) + 1))
            if overnight:
                dep = N_STEPS - 1  # clipped to next-day departure inside ev()
            else:
                dep = arr + int(self.rng.integers(dwell_steps[0], dwell_steps[1] + 1))
            e = float(self.rng.uniform(*efrac))
            out.append(self.ev(archetype, arr, dep, e, is_pri, overnight))
        return out


def finalize(evs):
    evs.sort(key=lambda e: (e["arrival_idx"], e["id"]))
    return evs


def gen_s1(n, seed):
    """S1 Baseline: moderate load; 27 % priority; half the non-priority vans return overnight."""
    fb = FleetBuilder(seed)
    n_pri = int(round(n * 0.27))
    n_np = n - n_pri
    evs = []
    for i in range(n_np):
        if i % 2 == 0:
            evs += fb.batch(1, "Light", (15, 19), None, (0.40, 0.70), False, overnight=True)
        else:
            evs += fb.batch(1, "Medium", (6, 16), (20, 44), (0.25, 0.50), False)
    for i in range(n_pri):
        evs += fb.batch(1, "Light" if i % 2 == 0 else "Medium", (7, 15), (20, 40), (0.30, 0.52), True)
    return finalize(evs)


def gen_s2(n, seed):
    """S2 Contention: early high-demand non-priority cohort vs mid-day priority arrivals."""
    fb = FleetBuilder(seed)
    n_pri = int(round(n * 0.50))
    n_np = n - n_pri
    n_block = int(round(n_np * 0.80))
    evs = []
    evs += fb.batch(n_block, "Medium", (7, 10), (16, 24), (0.72, 0.88), False)
    evs += fb.batch(n_np - n_block, "Light", (15, 19), None, (0.40, 0.70), False, overnight=True)
    evs += fb.batch(n_pri, "Medium", (10, 13), (20, 30), (0.40, 0.58), True)
    return finalize(evs)


def gen_s3(n, seed):
    """S3 Peak stress: pre-peak non-priority blockers vs priority arrivals in the ToU peak."""
    fb = FleetBuilder(seed)
    n_pri = int(round(n * 0.50))
    n_np = n - n_pri
    n_block = int(round(n_np * 0.75))
    evs = []
    evs += fb.batch(n_block, "Medium", (13, 15), (16, 24), (0.72, 0.86), False)
    evs += fb.batch(n_np - n_block, "Light", (17, 20), None, (0.40, 0.70), False, overnight=True)
    evs += fb.batch(n_pri, "Medium", (15, 18), (16, 26), (0.45, 0.64), True)
    return finalize(evs)


GENERATORS = {"S1_Baseline": gen_s1, "S2_Contention": gen_s2, "S3_PeakStress": gen_s3}


def contention_metrics(evs, n_evse, site_kw):
    present = np.zeros(N_STEPS)
    for e in evs:
        present[e["arrival_idx"]:e["departure_idx"]] += 1
    e_tot = sum(e["e_session_kwh"] for e in evs)
    day_steps = int((24 - START_H) / DT_H)  # 06:00-24:00 on day D
    e_day = sum(e["e_session_kwh"] for e in evs if not e["overnight"])
    return {
        "peak_present_over_evse": round(float(present.max() / n_evse), 3),
        "daytime_energy_over_site_capacity": round(e_day / (site_kw * day_steps * DT_H), 3),
        "total_session_energy_kwh": round(e_tot, 1),
        "priority_share": round(sum(e["priority"] == "High" for e in evs) / len(evs), 3),
    }


def pick_injections(evs90):
    """Choose E2 vehicle IDs deterministically from the 90-EV S2 scenario."""
    t_brk, t_abs, t_swp, t_pv = h2i(12), h2i(13), h2i(11.5), h2i(12)
    np_present = [e for e in evs90 if e["priority"] == "Low" and e["arrival_idx"] <= t_brk
                  and e["departure_idx"] >= t_brk + 8 and not e["overnight"]]
    pr_present = [e for e in evs90 if e["priority"] == "High" and e["arrival_idx"] <= t_abs
                  and e["departure_idx"] > t_abs + 4]
    pr_swp = [e for e in evs90 if e["priority"] == "High" and e["arrival_idx"] <= t_swp
              and e["departure_idx"] > t_swp + 8]
    np_swp = [e for e in evs90 if e["priority"] == "Low" and e["arrival_idx"] <= t_swp
              and e["departure_idx"] > t_swp + 8 and not e["overnight"]]
    if not (np_present and pr_present and pr_swp and np_swp):
        sys.exit("Cannot construct E2 injections from 90-EV S2 data")
    promote = np_present[0]
    return {
        "case_1_breakdown": {
            "time_idx": t_brk, "type": "vehicle_breakdown",
            "description": f"A priority van breaks down on route; standby non-priority van {promote['id']} "
                           f"is promoted with a 32 kWh mission and a 15:30 departure.",
            "promote_id": promote["id"], "new_mission_kwh": 32.0,
            "new_dep_idx": h2i(15.5), "new_w": 4.2, "operator_choice": "recommended",
        },
        "case_2_driver_unavailable": {
            "time_idx": t_abs, "type": "driver_unavailable",
            "description": f"Driver absence delays priority van {pr_present[0]['id']} by 2 h.",
            "delay_id": pr_present[0]["id"], "delay_steps": 8, "operator_choice": "recommended",
        },
        "case_3_roster_swap": {
            "time_idx": t_swp, "type": "roster_swap",
            "description": f"FMS swaps the afternoon route from priority {pr_swp[0]['id']} to non-priority "
                           f"{np_swp[0]['id']}; operator keeps all urgent vehicles.",
            "swap_from_id": pr_swp[0]["id"], "swap_to_id": np_swp[0]["id"],
            "operator_choice": "keep_all_urgent",
        },
        "case_4_pv_shortfall": {
            "time_idx": t_pv, "type": "pv_shortfall",
            "description": "Cloud front: realised PV falls to 25 % of the day-ahead forecast for 2 h.",
            "pv_drop_factor": 0.25, "duration_steps": 8, "operator_choice": "recommended",
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pv-json", default=os.path.join("data", "aemo_vic1_selected_days.json"))
    args = ap.parse_args()
    if not os.path.exists(args.pv_json):
        sys.exit(f"{args.pv_json} not found. Run scripts/download_aemo_pv.py and "
                 f"scripts/prepare_aemo_pv.py first (real AEMO data; no synthetic fallback).")
    pv = json.load(open(args.pv_json))
    ref = [d for d in pv["days"] if d.get("reference")]
    if len(ref) != 1:
        sys.exit("Expected exactly one reference day in the PV file")
    ref = ref[0]

    start = datetime.fromisoformat(ref["date"]).replace(hour=START_H)
    env = {
        "start_time_aest": start.isoformat(sep=" "),
        "step_hours": DT_H,
        "n_steps": N_STEPS,
        "tou_tariff": tou_profile(),
        "pv_forecast": ref["pv_forecast_kw"],
        "pv_actual": ref["pv_actual_kw"],
        "pv_day": ref["date"],
        "pv_source": pv["metadata"]["source"],
        "bess": bess_profile(),
    }
    evs90_s2 = None
    for size, infra in INFRA.items():
        sc, meta = {}, {}
        for k, (name, fn) in enumerate(GENERATORS.items()):
            evs = fn(size, seed=1000 + 10 * size + k)
            sc[name] = evs
            meta[name] = contention_metrics(evs, infra["n_evse"], infra["site_kw"])
            if size == 90 and name == "S2_Contention":
                evs90_s2 = evs
        out = {"metadata": {"fleet_size": size, "generator": "data_generator.py v3",
                            "contention": meta},
               "infrastructure": infra, "environment": env, "scenarios": sc}
        fname = f"simulation_data_{size}EVs.json"
        with open(fname, "w") as f:
            json.dump(out, f, indent=1)
        print(f"Saved {fname}: " + "; ".join(f"{k} {v}" for k, v in meta.items()))

    with open(os.path.join("data", "operator_injections.json"), "w") as f:
        json.dump(pick_injections(evs90_s2), f, indent=2)
    print("Saved data/operator_injections.json")


if __name__ == "__main__":
    main()
