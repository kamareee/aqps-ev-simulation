"""
WP2 — Enriched Scenario & Roster Generator for AQPS Revision.

Integrates empirical commercial EV van parameters from:
  - Vales-Alonso & Alcaraz (2026) "Emergent Charging Coordination in Electric Delivery Fleets"
  - Fiori & Marzano (2018) "Modelling energy consumption of electric freight vehicles in urban
    pickup/delivery operations" (e_i ~ U(0.22, 0.35) kWh/km)

Generates:
  - simulation_data_45EVs.json  (16 EVSEs, 150 kW transformer limit)
  - simulation_data_90EVs.json  (35 EVSEs, 300 kW transformer limit)
  - simulation_data_180EVs.json (60 EVSEs, 600 kW transformer limit - for E5b scalability)
  - data/operator_injections.json (Scripted case studies for E2)
"""

import os
import json
import numpy as np
from datetime import datetime, timedelta
from scripts.prepare_aemo_pv import build_aemo_vic1_dataset


_TOU_HALFHOUR_CKWH = [
    18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0,  # 00:00-04:00 Off-Peak ($0.18/kWh)
    18.0, 18.0, 18.0, 18.0, 18.0, 18.0,              # 04:00-07:00 Off-Peak
    21.0, 21.0, 21.0, 21.0, 21.0, 21.0, 21.0, 21.0,  # 07:00-11:00 Shoulder ($0.21/kWh)
    21.0, 21.0, 21.0, 21.0, 21.0, 21.0, 21.0, 21.0,  # 11:00-15:00 Shoulder
    32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0,  # 15:00-19:00 Peak ($0.32/kWh)
    32.0, 32.0, 32.0, 32.0,                          # 19:00-21:00 Peak
    21.0, 21.0,                                      # 21:00-22:00 Shoulder
    18.0, 18.0, 18.0, 18.0,                          # 22:00-24:00 Off-Peak
]

# Grounded in Renault Kangoo E-Tech (45 kWh), Ford E-Transit / Mercedes eVito (68 kWh), eSprinter (85 kWh)
ARCHETYPES = {
    "Light":  {"model": "Renault Kangoo E-Tech", "cap": 45.0, "p_max": 11.0, "eff": 0.95, "cons_range": (0.20, 0.26)},
    "Medium": {"model": "Ford E-Transit 68kWh",  "cap": 68.0, "p_max": 22.0, "eff": 0.92, "cons_range": (0.25, 0.33)},
    "Heavy":  {"model": "Mercedes eSprinter",    "cap": 85.0, "p_max": 22.0, "eff": 0.92, "cons_range": (0.30, 0.36)},
}

INFRA = {
    45:  {"n_evse": 16, "transformer_kW": 150.0, "voltage": 240},
    90:  {"n_evse": 35, "transformer_kW": 300.0, "voltage": 240},
    180: {"n_evse": 60, "transformer_kW": 600.0, "voltage": 240},
}


def generate_scenarios():
    os.makedirs("data", exist_ok=True)
    aemo_data = build_aemo_vic1_dataset()

    start_time = datetime.strptime("2026-02-12 06:00", "%Y-%m-%d %H:%M")
    time_steps = [start_time + timedelta(minutes=15 * i) for i in range(96)]

    def hour_to_idx(h):
        return int((h - 6) * 4)

    def get_tou(dt):
        halfhour_idx = (dt.hour * 60 + dt.minute) // 30
        return _TOU_HALFHOUR_CKWH[halfhour_idx] / 100.0

    tou_profile = [get_tou(t) for t in time_steps]

    # 20 kW PV array + 10 kWh / 5 kW BESS peak-shaving dispatch (discharges 15:00-19:00 during Peak TOU)
    pv_profile = aemo_data["days"][0]["pv_forecast_kw"]
    bess_profile = []
    for t in time_steps:
        if 15 <= t.hour < 19:
            bess_profile.append(2.5)  # 2.5 kW x 4h = 10 kWh BESS peak discharge
        elif 10 <= t.hour < 14:
            bess_profile.append(-2.5) # Solar midday recharge
        else:
            bess_profile.append(0.0)

    ev_counter = 0

    def make_ev(archetype, arr_idx, dep_idx, energy_frac, is_priority, rng):
        nonlocal ev_counter
        arc = ARCHETYPES[archetype]
        cap = arc["cap"]
        arr_c = int(np.clip(arr_idx, 0, 90))
        dep_c = int(np.clip(dep_idx, arr_c + 8, 95))
        arr_dt = start_time + timedelta(minutes=15 * arr_c)
        dep_dt = start_time + timedelta(minutes=15 * dep_c)

        # Empirical route energy calculation (Fiori & Marzano 2018; Vales-Alonso & Alcaraz 2026)
        req_e = round(cap * energy_frac, 2)
        reserve_kwh = round(0.10 * cap, 2)  # SoC_min = 10%
        eod_soc_kwh = round(float(rng.uniform(0.12, 0.25) * cap), 2)
        mission_total = eod_soc_kwh + req_e
        return_kwh = round(min(8.0, 0.18 * req_e), 2)
        next_trip_kwh = round(max(4.0, mission_total - return_kwh - reserve_kwh), 2)

        # Re-align target_energy_kwh so E_i^req = next_trip + return + reserve - eod_soc holds exactly
        req_e = round(max(5.0, next_trip_kwh + return_kwh + reserve_kwh - eod_soc_kwh), 2)

        w_i = round(float(rng.uniform(2.5, 4.5) if is_priority else rng.uniform(0.5, 1.4)), 2)
        further_trips = round(float(rng.uniform(0.0, 4.0) if is_priority else 0.0), 2)
        next_day_dep = int(rng.integers(92, 108))
        next_day_req = round(float(cap * rng.uniform(0.30, 0.55)), 2)

        ev = {
            "id": f"EV_{ev_counter:03d}",
            "arrival": arr_dt.isoformat(),
            "departure": dep_dt.isoformat(),
            "arrival_idx": arr_c,
            "departure_idx": dep_c,
            "req_energy": req_e,
            "p_max": arc["p_max"],
            "battery_cap_kwh": cap,
            "eod_soc_kwh": eod_soc_kwh,
            "next_trip_kwh": next_trip_kwh,
            "return_kwh": return_kwh,
            "reserve_kwh": reserve_kwh,
            "further_trips_today_kwh": further_trips,
            "next_day_dep_idx": next_day_dep,
            "next_day_req_kwh": next_day_req,
            "trip_criticality_w": w_i,
            "priority": "High" if is_priority else "Low",
            "archetype": archetype,
        }
        ev_counter += 1
        return ev

    def make_batch(n, archetype, arr_range, dwell_range, energy_range, is_priority, rng):
        evs = []
        for _ in range(n):
            arr = int(rng.integers(arr_range[0], arr_range[1] + 1))
            dwell = int(rng.integers(dwell_range[0], dwell_range[1] + 1))
            dep = min(95, arr + dwell)
            efrac = float(rng.uniform(energy_range[0], energy_range[1]))
            evs.append(make_ev(archetype, arr, dep, efrac, is_priority, rng))
        return evs

    def finalize(ev_list):
        ev_list.sort(key=lambda x: x["arrival_idx"])
        for ev in ev_list:
            avail_steps = ev["departure_idx"] - ev["arrival_idx"]
            steps_needed = ev["req_energy"] / (ev["p_max"] * 0.25)
            ev["laxity_steps"] = round(avail_steps - steps_needed, 1)
            ev["avail_steps"] = avail_steps
        return ev_list

    def gen_s1(n_evs, seed=100):
        nonlocal ev_counter
        ev_counter = 0
        rng = np.random.default_rng(seed)
        n_pri = int(round(n_evs * 0.27))
        n_nonpri = n_evs - n_pri
        evs = []
        for i in range(n_nonpri):
            arc = "Light" if i % 2 == 0 else "Medium"
            evs += make_batch(1, arc, (hour_to_idx(6), hour_to_idx(16)), (20, 44), (0.25, 0.50), False, rng)
        for i in range(n_pri):
            arc = "Light" if i % 2 == 0 else "Medium"
            evs += make_batch(1, arc, (hour_to_idx(7), hour_to_idx(15)), (20, 40), (0.30, 0.52), True, rng)
        return finalize(evs)

    def gen_s3(n_evs, seed=200):
        """S2 Contention (key S3_HighPriority): Early high-demand NP cohort vs mid-day Priority."""
        nonlocal ev_counter
        ev_counter = 0
        rng = np.random.default_rng(seed)
        n_pri = int(round(n_evs * 0.50))
        n_nonpri = n_evs - n_pri
        n_decoys = int(round(n_nonpri * 0.80))
        n_normal_np = n_nonpri - n_decoys
        evs = []
        evs += make_batch(n_decoys, "Medium", (hour_to_idx(7), hour_to_idx(10)), (16, 24), (0.72, 0.88), False, rng)
        evs += make_batch(n_normal_np, "Light", (hour_to_idx(6), hour_to_idx(14)), (24, 40), (0.25, 0.48), False, rng)
        evs += make_batch(n_pri, "Medium", (hour_to_idx(10), hour_to_idx(13)), (20, 30), (0.40, 0.58), True, rng)
        return finalize(evs)

    def gen_s6(n_evs, seed=300):
        """S3 Peak Stress (key S6_PeakStress): Pre-peak NP blockers vs Peak TOU Priority arrivals."""
        nonlocal ev_counter
        ev_counter = 0
        rng = np.random.default_rng(seed)
        n_pri = int(round(n_evs * 0.50))
        n_nonpri = n_evs - n_pri
        n_blockers = int(round(n_nonpri * 0.75))
        n_normal_np = n_nonpri - n_blockers
        evs = []
        evs += make_batch(n_blockers, "Medium", (hour_to_idx(13), hour_to_idx(15)), (16, 24), (0.72, 0.86), False, rng)
        evs += make_batch(n_normal_np, "Light", (hour_to_idx(6), hour_to_idx(12)), (24, 40), (0.25, 0.45), False, rng)
        evs += make_batch(n_pri, "Medium", (hour_to_idx(15), hour_to_idx(18)), (16, 26), (0.45, 0.64), True, rng)
        return finalize(evs)

    generators = {
        "S1_Baseline": gen_s1,
        "S3_HighPriority": gen_s3,
        "S6_PeakStress": gen_s6,
    }

    environment = {
        "tou_tariff": tou_profile,
        "pv_forecast": pv_profile,
        "bess": bess_profile,
        "step_delta": 0.25,
    }

    for size in [45, 90, 180]:
        sc = {}
        for sname, fn in generators.items():
            sc[sname] = fn(n_evs=size, seed=(1000 + size + len(sname) * 17))
        file_data = {
            "metadata": {
                "step_minutes": 15,
                "horizon": 96,
                "start_time": "2026-02-12 06:00",
                "fleet_size": size,
            },
            "infrastructure": INFRA[size],
            "environment": environment,
            "scenarios": sc,
        }
        fname = f"simulation_data_{size}EVs.json"
        with open(fname, "w") as f:
            json.dump(file_data, f, indent=2)
        print(f"[WP2] Generated {fname} ({INFRA[size]['n_evse']} EVSEs, {INFRA[size]['transformer_kW']} kW)")

    # Generate scripted operator injections for E2
    injections = {
        "case_1_breakdown": {
            "time_idx": 24,  # 12:00 PM
            "type": "vehicle_breakdown",
            "description": "Priority van EV_025 breaks down on route; standby NP van EV_005 promoted with urgent 32 kWh requirement and 15:30 deadline.",
            "promote_id": "EV_005",
            "new_req_kwh": 32.0,
            "new_dep_idx": 38,
            "new_w": 4.2,
            "operator_choice": "recommended",
        },
        "case_2_driver_unavailable": {
            "time_idx": 28,  # 13:00 PM
            "type": "driver_unavailable",
            "description": "Driver absence delays EV_030 departure by +2h (laxity increases, triggering safe-release for incoming urgent EV_038).",
            "delay_id": "EV_030",
            "delay_steps": 8,
            "operator_choice": "recommended",
        },
        "case_3_roster_swap": {
            "time_idx": 22,  # 11:30 AM
            "type": "roster_swap",
            "description": "FMS swaps afternoon delivery route between EV_028 (Priority) and EV_008 (Non-Priority); operator accepts business-continuity option.",
            "swap_from_id": "EV_028",
            "swap_to_id": "EV_008",
            "operator_choice": "keep_all_urgent",
        },
        "case_4_pv_shortfall": {
            "time_idx": 24,  # 12:00 PM
            "type": "pv_shortfall",
            "description": "Sudden cloud front drops rooftop PV output from 17.8 kW forecast to 4.2 kW actual during midday window.",
            "pv_drop_factor": 0.25,
            "operator_choice": "recommended",
        },
    }
    with open(os.path.join("data", "operator_injections.json"), "w") as f:
        json.dump(injections, f, indent=2)
    print("[WP2] Saved scripted operator injections to data/operator_injections.json")


if __name__ == "__main__":
    generate_scenarios()
