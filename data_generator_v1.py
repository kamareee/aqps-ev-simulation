import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta


def generate_scenarios(output_file="simulation_data.json"):
    """
    Generates deterministic simulation scenarios (S1, S3, S6) for AQPS.
    """

    # 1. Configuration
    # 96 steps (15 mins) starting from 06:00 to 06:00 next day
    start_time = datetime.strptime("2026-02-12 06:00", "%Y-%m-%d %H:%M")
    time_steps = [start_time + timedelta(minutes=15 * i) for i in range(96)]

    # Archetypes (Battery Cap kWh, Max Power kW)
    archetypes = [
        {"name": "Light", "cap": 40, "p_max": 11, "eff": 0.95},
        {"name": "Medium", "cap": 75, "p_max": 22, "eff": 0.92},
    ]

    # 2. Environmental Forecasts
    # ToU Tariff (Peak 15:00-21:00)
    def get_tou(dt):
        h = dt.hour
        if 15 <= h < 21:
            return 0.35  # Peak
        if 7 <= h < 15 or 21 <= h < 22:
            return 0.22  # Shoulder
        return 0.15  # Off-Peak

    tou_profile = [get_tou(t) for t in time_steps]

    # PV Profile (Bell Curve peaking at 12pm)
    # Peak 50kW scaled by sine wave
    pv_profile = []
    for t in time_steps:
        # Solar window roughly 7am to 7pm
        if 7 <= t.hour < 19:
            # Normalized 0 to 1
            x = (t.hour + t.minute / 60.0 - 7) / 12.0
            pv_val = 50.0 * np.sin(x * np.pi)
        else:
            pv_val = 0.0
        pv_profile.append(max(0.0, pv_val))

    # Base Load (Background building load)
    base_load = [10.0 + (15.0 if 8 <= t.hour < 18 else 0.0) for t in time_steps]

    # 3. Scenario Logic
    scenarios = {}

    def generate_ev_list(n_evs, priority_pct, arrival_mode):
        ev_list = []
        np.random.seed(42)  # Reproducible

        for i in range(n_evs):
            # Select Archetype
            arc = archetypes[i % len(archetypes)]

            # Arrival Time
            if arrival_mode == "uniform":
                # Uniform between 06:00 (0) and 18:00 (48)
                arr_idx = np.random.randint(0, 48)
            elif arrival_mode == "pm_cluster":
                # 70% between 14:00 (32) and 18:00 (48)
                if np.random.rand() < 0.7:
                    arr_idx = np.random.randint(32, 49)
                else:
                    arr_idx = np.random.randint(0, 32)

            # Dwell Time (4 to 12 hours converted to steps)
            dwell_steps = np.random.randint(16, 48)
            dep_idx = min(95, arr_idx + dwell_steps)

            # Energy Needs
            req_energy = arc["cap"] * np.random.uniform(0.3, 0.8)

            # Priority
            is_priority = i < n_evs * priority_pct

            ev_list.append(
                {
                    "id": f"EV_{i}",
                    "arrival_idx": int(arr_idx),
                    "departure_idx": int(dep_idx),
                    "req_energy_kwh": round(req_energy, 2),
                    "max_power_kw": arc["p_max"],
                    "is_priority": bool(is_priority),
                }
            )

        # Sort by arrival for the simulation loop
        ev_list.sort(key=lambda x: x["arrival_idx"])
        return ev_list

    # --- S1: Baseline ---
    # 27% Priority, Uniform Arrival
    scenarios["S1"] = generate_ev_list(
        n_evs=50, priority_pct=0.27, arrival_mode="uniform"
    )

    # --- S3: High Priority ---
    # 50% Priority, Uniform Arrival
    scenarios["S3"] = generate_ev_list(
        n_evs=50, priority_pct=0.50, arrival_mode="uniform"
    )

    # --- S6: Peak Stress ---
    # 50% Priority, PM Clustering (Conflict with Peak Pricing & waning PV)
    scenarios["S6"] = generate_ev_list(
        n_evs=60, priority_pct=0.50, arrival_mode="pm_cluster"
    )

    # 4. Output
    data = {
        "metadata": {"step_minutes": 15, "horizon": 96},
        "environment": {
            "tou": tou_profile,
            "pv": pv_profile,
            "base_load": base_load,
            "bess": [0.0]
            * 96,  # Placeholder, scheduler handles BESS logic or reads this
        },
        "scenarios": scenarios,
    }

    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Generated {output_file} with S1, S3, S6.")


if __name__ == "__main__":
    generate_scenarios()
