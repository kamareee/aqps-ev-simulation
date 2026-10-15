# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from src.aqps.scheduler import AQPSScheduler
from src.aqps.data_structures import Vehicle


# --- Baseline LLF Scheduler for Comparison ---
class LLFScheduler:
    def __init__(self, num_chargers, grid_limit):
        self.num_chargers = num_chargers
        self.grid_limit = grid_limit
        self.active_vehicles = []

    def step(self, time_idx, arriving, departing_ids, current_cap):
        # 1. Update List
        self.active_vehicles = [
            v for v in self.active_vehicles if v.id not in departing_ids
        ]
        self.active_vehicles.extend(arriving)

        # 2. Calculate Laxity
        # Laxity = (Time Remaining) - (Time needed to charge at max power)
        for v in self.active_vehicles:
            time_rem = (v.departure_time_idx - time_idx) * 0.25
            if time_rem <= 0:
                laxity = -999
            else:
                time_to_charge = v.energy_needed / v.max_charging_power_kw
                laxity = time_rem - time_to_charge
            v.laxity = laxity

        # 3. Sort by Laxity (Ascending)
        self.active_vehicles.sort(key=lambda x: x.laxity)

        # 4. Allocate
        commands = {}
        total_power = 0.0

        # Simple greedy allocation up to grid limit
        candidates = self.active_vehicles[: self.num_chargers]

        remaining_cap = current_cap

        for v in candidates:
            if remaining_cap <= 0:
                commands[v.id] = 0.0
                continue

            p_cmd = min(v.max_charging_power_kw, remaining_cap)

            # Simple logic: if fully charged, stop
            if v.energy_needed <= 0.01:
                p_cmd = 0.0

            commands[v.id] = p_cmd
            remaining_cap -= p_cmd
            total_power += p_cmd

            # Update state
            v.energy_delivered += p_cmd * 0.25
            v.energy_needed = max(0.0, v.target_energy_kwh - v.energy_delivered)

        return {"commands": commands, "grid_power": total_power}


# --- Main Simulation Runner ---
def run_scenario(scenario_name, ev_data, env_data):
    # Utilize scenario_name for clear console logging
    print(f"\n[{scenario_name}] Initializing simulation horizon...")

    horizon = len(env_data["tou"])

    # Configuration
    config = {
        "station": {"num_chargers": 20},
        "grid": {"transformer_limit_kw": 100.0},
        "simulation": {
            "dt_hours": 0.25,
            "planning_horizon_steps": 12,  # 3 hour lookahead for MPC
        },
    }

    # ==========================================
    # 1. Run AQPS
    # ==========================================
    aqps = AQPSScheduler(config)

    # Tag results with the scenario name
    aqps_results = {
        "scenario": scenario_name,
        "cost": 0,
        "priority_unmet": 0,
        "grid_peak": 0,
        "power_trace": [],
    }

    # Reset Vehicles for AQPS
    vehicles_map = {}
    for ev in ev_data:
        v = Vehicle(
            id=ev["id"],
            arrival_time_idx=ev["arrival_idx"],
            departure_time_idx=ev["departure_idx"],
            target_energy_kwh=ev["req_energy_kwh"],
            max_charging_power_kw=ev["max_power_kw"],
            is_priority=ev["is_priority"],
        )
        vehicles_map[v.id] = v

    for t in range(horizon):
        arriving = [v for v in vehicles_map.values() if v.arrival_time_idx == t]
        departing_ids = [
            v.id for v in vehicles_map.values() if v.departure_time_idx == t
        ]

        # Forecasts
        tou = np.array(env_data["tou"][t:] + [env_data["tou"][-1]] * t)
        pv = np.array(env_data["pv"][t:] + [0.0] * t)
        bess = np.array([0.0] * len(tou))

        step_res = aqps.step(t, arriving, departing_ids, tou, pv, bess)

        # FIX: Calculate realized step cost identically to LLF to avoid adding multi-step MPC horizon objectives
        total_ev_power = sum(step_res["commands"].values())
        grid_p = max(0.0, total_ev_power - env_data["pv"][t])
        realized_cost = grid_p * env_data["tou"][t] * config["simulation"]["dt_hours"]

        aqps_results["cost"] += realized_cost
        aqps_results["grid_peak"] = max(aqps_results["grid_peak"], grid_p)
        aqps_results["power_trace"].append(grid_p)

    for v in vehicles_map.values():
        if v.is_priority and v.energy_needed > 0.5:
            aqps_results["priority_unmet"] += 1

    # ==========================================
    # 2. Run LLF
    # ==========================================
    llf = LLFScheduler(
        config["station"]["num_chargers"], config["grid"]["transformer_limit_kw"]
    )

    # Tag results with the scenario name
    llf_results = {
        "scenario": scenario_name,
        "cost": 0,
        "priority_unmet": 0,
        "grid_peak": 0,
        "power_trace": [],
    }

    vehicles_map_llf = {}
    for ev in ev_data:
        v = Vehicle(
            id=ev["id"],
            arrival_time_idx=ev["arrival_idx"],
            departure_time_idx=ev["departure_idx"],
            target_energy_kwh=ev["req_energy_kwh"],
            max_charging_power_kw=ev["max_power_kw"],
            is_priority=ev["is_priority"],
        )
        vehicles_map_llf[v.id] = v

    for t in range(horizon):
        arriving = [v for v in vehicles_map_llf.values() if v.arrival_time_idx == t]
        departing_ids = [
            v.id for v in vehicles_map_llf.values() if v.departure_time_idx == t
        ]

        current_cap = config["grid"]["transformer_limit_kw"] + env_data["pv"][t]

        step_res = llf.step(t, arriving, departing_ids, current_cap)

        grid_p = step_res["grid_power"] - env_data["pv"][t]
        grid_p = max(0, grid_p)
        cost = grid_p * env_data["tou"][t] * config["simulation"]["dt_hours"]

        llf_results["cost"] += cost
        llf_results["grid_peak"] = max(llf_results["grid_peak"], grid_p)
        llf_results["power_trace"].append(grid_p)

    for v in vehicles_map_llf.values():
        if v.is_priority and v.energy_needed > 0.5:
            llf_results["priority_unmet"] += 1

    print(f"[{scenario_name}] Simulation complete.")
    return aqps_results, llf_results


def main():
    # Load Data safely
    try:
        with open("simulation_data.json", "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: simulation_data.json not found in the root directory.")
        return

    scenarios = ["S1", "S3", "S6"]
    metrics = {
        "Scenario": [],
        "Algo": [],
        "Cost": [],
        "Priority_Failures": [],
        "Peak_Load": [],
    }

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    for i, s_name in enumerate(scenarios):
        evs = data["scenarios"][s_name]
        env = data["environment"]

        res_aqps, res_llf = run_scenario(s_name, evs, env)

        # Record Metrics using the tagged scenario name
        metrics["Scenario"].extend([res_aqps["scenario"], res_llf["scenario"]])
        metrics["Algo"].extend(["AQPS", "LLF"])
        metrics["Cost"].extend([res_aqps["cost"], res_llf["cost"]])
        metrics["Priority_Failures"].extend(
            [res_aqps["priority_unmet"], res_llf["priority_unmet"]]
        )
        metrics["Peak_Load"].extend([res_aqps["grid_peak"], res_llf["grid_peak"]])

        # Plot Power Trace
        ax = axes[i]
        ax.plot(res_aqps["power_trace"], label="AQPS", color="green", linewidth=2)
        ax.plot(
            res_llf["power_trace"],
            label="LLF",
            color="red",
            linestyle="--",
            linewidth=2,
        )
        ax.set_title(f"Grid Power Profile: {s_name}")
        ax.set_ylabel("Power (kW)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    axes[2].set_xlabel("Time Step (15 min)")

    # Use the explicitly created 'fig' object directly for safe memory handling
    fig.tight_layout()
    fig.savefig("comparison_results.png")
    plt.close(fig)

    print("\nPlot saved successfully to comparison_results.png")

    # Print clean summary table
    df = pd.DataFrame(metrics)
    print("\nSimulation Results Summary:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
