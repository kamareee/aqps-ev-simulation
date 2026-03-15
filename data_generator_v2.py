# pylint: disable=wrong-import-order,missing-module-docstring,missing-function-docstring

import pandas as pd
import numpy as np
import json
import random
from datetime import datetime, timedelta


def generate_scenarios_for_fleet(fleet_size):
    """
    Generates deterministic simulation scenarios for the AQPC framework.
    Outputs scenarios S1, S3, and S6 for a given fleet size.
    """

    # 1. Define Fleet Archetypes
    archetypes = {
        "Light": {"cap": 40, "p_max": 11, "eff": 0.95},
        "Medium": {"cap": 75, "p_max": 22, "eff": 0.92},
    }

    def get_arc_props(index):
        # Cyclic mix between Light and Medium
        keys = list(archetypes.keys())
        name = keys[index % len(keys)]
        props = archetypes[name].copy()
        props["name"] = name
        return props

    # 2. Environmental Forecast Profiles (24 hours, 15-min steps)
    start_time = datetime(2026, 2, 12, 0, 0)
    time_steps = pd.date_range(start_time, periods=96, freq="15min")

    # ToU Tariff (Victorian Default Market Offer approx)
    def get_tou(dt):
        hour = dt.hour + dt.minute / 60.0
        if (8 <= hour < 10) or (16 <= hour < 18):
            return 0.26668  # Peak
        return 0.05623  # Off-Peak

    tou_profile = [get_tou(t) for t in time_steps]

    # PV Profile (Bell Curve - Peak around noon)
    pv_peak = (
        fleet_size / 45
    ) * 50.0  # Scale PV peak with fleet size (e.g., 50kW for 45 EVs)
    pv_profile = [
        pv_peak * max(0, np.sin(np.pi * (t.hour + t.minute / 60 - 6) / 12))
        for t in time_steps
    ]

    # Base Load (scaled with fleet size)
    base_load_idle = (fleet_size / 45) * 5.0
    base_load_active = (fleet_size / 45) * 10.0
    base_load = [
        base_load_idle + (base_load_active if 9 <= t.hour < 17 else 0.0)
        for t in time_steps
    ]

    # 3. Scenario Logic
    scenarios = {}

    # Configuration for requested scenarios
    scenario_configs = {
        "S1_Baseline": {"priority_pct": 0.27, "arrival_pattern": "uniform"},
        "S3_HighPriority": {"priority_pct": 0.50, "arrival_pattern": "uniform"},
        "S6_PeakStress": {"priority_pct": 0.50, "arrival_pattern": "pm_cluster"},
    }

    for sc_name, config in scenario_configs.items():
        # Set a fixed seed for reproducible vehicle generation per scenario
        random.seed(hash(sc_name) % 10000)
        np.random.seed(hash(sc_name) % 10000)

        ev_list = []

        # Determine exactly which EVs get Priority based on target percentage
        num_priority = int(fleet_size * config["priority_pct"])
        priority_indices = set(random.sample(range(fleet_size), num_priority))

        for i in range(fleet_size):
            arc = get_arc_props(i)

            # --- Arrival Time Generation ---
            if config["arrival_pattern"] == "uniform":
                # Uniform arrivals between 6:00 AM and 6:00 PM (18.0)
                arrival_hour = np.random.uniform(6.0, 18.0)
            elif config["arrival_pattern"] == "pm_cluster":
                # 70% arrive during peak PM (2:00 PM - 6:00 PM), 30% uniform
                if random.random() < 0.70:
                    arrival_hour = np.random.uniform(14.0, 18.0)
                else:
                    arrival_hour = np.random.uniform(6.0, 14.0)

            # Convert float hour to datetime
            arrival_dt = start_time + timedelta(hours=arrival_hour)
            # Round to nearest 15 mins
            arrival_dt = arrival_dt.replace(second=0, microsecond=0)
            minute = (arrival_dt.minute // 15) * 15
            arrival_dt = arrival_dt.replace(minute=minute)

            # --- Departure Time Generation ---
            # Dwell time between 4 and 10 hours
            dwell_hours = np.random.uniform(4.0, 10.0)
            departure_dt = arrival_dt + timedelta(hours=dwell_hours)

            # --- Energy Request ---
            # Random request between 20% and 80% of capacity
            req_energy = arc["cap"] * np.random.uniform(0.2, 0.8)
            init_energy = arc["cap"] * np.random.uniform(0.1, 0.2)

            ev_list.append(
                {
                    "id": f"EV_{i:03d}",
                    "arrival": arrival_dt.isoformat(),
                    "departure": departure_dt.isoformat(),
                    "req_energy": round(req_energy, 2),
                    "init_energy": round(init_energy, 2),
                    "p_max": arc["p_max"],
                    "eff": arc["eff"],
                    "archetype": arc["name"],
                    "priority": "High" if i in priority_indices else "Low",
                }
            )

        scenarios[sc_name] = ev_list

    # --- Final Compilation ---
    output = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "fleet_size": fleet_size,
            "step_delta": 0.25,
            "horizon": 96,
        },
        "environment": {
            "tou_tariff": tou_profile,
            "pv_forecast": pv_profile,
            "base_load": base_load,
        },
        "scenarios": scenarios,
    }

    filename = f"simulation_data_{fleet_size}EVs.json"
    with open(filename, "w") as f:
        json.dump(output, f, indent=4)

    print(f"Generated data for {fleet_size} EVs saved to {filename}")


if __name__ == "__main__":
    # Generate files for both fleet configurations
    generate_scenarios_for_fleet(fleet_size=45)
    generate_scenarios_for_fleet(fleet_size=90)
