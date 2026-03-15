# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring

import os
import json
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from src.aqps.scheduler import AQPSScheduler
from src.aqps.data_structures import Vehicle

RESULTS_DIR = "results_resubmission"
os.makedirs(RESULTS_DIR, exist_ok=True)


# =============================================================================
# Baseline LLF Scheduler (from v1)
# =============================================================================
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

        candidates = self.active_vehicles[: self.num_chargers]
        remaining_cap = current_cap

        for v in candidates:
            if remaining_cap <= 0:
                commands[v.id] = 0.0
                continue

            p_cmd = min(v.max_charging_power_kw, remaining_cap)

            if v.energy_needed <= 0.01:
                p_cmd = 0.0

            commands[v.id] = p_cmd
            remaining_cap -= p_cmd
            total_power += p_cmd

            # Update state internally for LLF
            v.energy_delivered += p_cmd * 0.25
            v.energy_needed = max(0.0, v.target_energy_kwh - v.energy_delivered)

        return {"commands": commands, "grid_power": total_power}


# =============================================================================
# Helper Functions
# =============================================================================
def parse_dt_to_idx(dt_str):
    dt = datetime.fromisoformat(dt_str)
    return int((dt.hour * 60 + dt.minute) // 15)


def create_vehicle_map(ev_data):
    """Creates a fresh map of Vehicle objects for clean simulation state."""
    vehicles_map = {}
    for ev in ev_data:
        v = Vehicle(
            id=ev["id"],
            arrival_time_idx=parse_dt_to_idx(ev["arrival"]),
            departure_time_idx=parse_dt_to_idx(ev["departure"]),
            target_energy_kwh=ev["req_energy"],
            max_charging_power_kw=ev["p_max"],
            is_priority=(ev["priority"] == "High"),
        )
        v.energy_delivered = 0.0
        v.energy_needed = v.target_energy_kwh
        vehicles_map[v.id] = v
    return vehicles_map


# =============================================================================
# Simulation Runner
# =============================================================================
def run_scenario(fleet_size, scenario_key, scenario_label, ev_data, env_data):
    print(
        f"\n[{scenario_label}] Initializing simulation horizon for {fleet_size} EVs..."
    )

    horizon = len(env_data["tou_tariff"])
    dt_hours = env_data.get("step_delta", 0.25)

    num_chargers = int(fleet_size / 3)
    transformer_limit_kw = (fleet_size / 45) * 150.0

    config = {
        "station": {"num_chargers": num_chargers},
        "grid": {"transformer_limit_kw": transformer_limit_kw},
        "simulation": {"dt_hours": dt_hours, "planning_horizon_steps": 12},
    }

    # Initialize Schedulers
    aqps = AQPSScheduler(config)
    llf = LLFScheduler(num_chargers, transformer_limit_kw)

    # Initialize distinct vehicle sets to prevent cross-contamination
    vehicles_aqps = create_vehicle_map(ev_data)
    vehicles_llf = create_vehicle_map(ev_data)

    # Metrics Tracking
    time_steps = []
    priority_arrivals = [
        v.arrival_time_idx for v in vehicles_aqps.values() if v.is_priority
    ]

    aqps_cost_history, llf_cost_history = [], []
    current_cost_aqps, current_cost_llf = 0.0, 0.0

    aqps_preemptions = []
    llf_starvations = []

    prev_cmds_aqps = {}

    for t in range(horizon):
        tou = np.array(env_data["tou_tariff"][t:] + [env_data["tou_tariff"][-1]] * t)
        pv = np.array(env_data["pv_forecast"][t:] + [0.0] * t)
        bess = np.array([0.0] * len(tou))
        current_cap_llf = transformer_limit_kw + pv[0]

        # ---------------------------------------------------------------------
        # 1. Run AQPS
        # ---------------------------------------------------------------------
        arriving_aqps = [v for v in vehicles_aqps.values() if v.arrival_time_idx == t]
        departing_aqps = [
            v.id for v in vehicles_aqps.values() if v.departure_time_idx == t
        ]

        step_res_aqps = aqps.step(t, arriving_aqps, departing_aqps, tou, pv, bess)
        cmds_aqps = step_res_aqps.get("commands", {})

        # Update AQPS Energy & Track Preemptions
        priority_arriving_now = any(v.is_priority for v in arriving_aqps)
        for v_id, p_cmd in cmds_aqps.items():
            if v_id in vehicles_aqps:
                vehicles_aqps[v_id].energy_delivered += p_cmd * dt_hours

        for v in vehicles_aqps.values():
            if not v.is_priority and v.arrival_time_idx <= t < v.departure_time_idx:
                still_needs_energy = (v.target_energy_kwh - v.energy_delivered) > 0.5
                was_charging = prev_cmds_aqps.get(v.id, 0) > 0.0
                is_cut_off = cmds_aqps.get(v.id, 0) == 0.0
                if (
                    was_charging
                    and is_cut_off
                    and still_needs_energy
                    and priority_arriving_now
                ):
                    aqps_preemptions.append(t)
        prev_cmds_aqps = cmds_aqps

        grid_p_aqps = max(0.0, sum(cmds_aqps.values()) - pv[0])
        current_cost_aqps += grid_p_aqps * tou[0] * dt_hours
        aqps_cost_history.append(current_cost_aqps)

        # ---------------------------------------------------------------------
        # 2. Run LLF
        # ---------------------------------------------------------------------
        arriving_llf = [v for v in vehicles_llf.values() if v.arrival_time_idx == t]
        departing_llf = [
            v.id for v in vehicles_llf.values() if v.departure_time_idx == t
        ]

        step_res_llf = llf.step(t, arriving_llf, departing_llf, current_cap_llf)
        cmds_llf = step_res_llf.get("commands", {})

        # Track LLF Starvation Events (Priority EV needs charge but gets 0 power)
        for v in vehicles_llf.values():
            if v.is_priority and v.arrival_time_idx <= t < v.departure_time_idx:
                if v.energy_needed > 0.5 and cmds_llf.get(v.id, 0) == 0.0:
                    llf_starvations.append(t)

        grid_p_llf = max(0.0, step_res_llf["grid_power"] - pv[0])
        current_cost_llf += grid_p_llf * tou[0] * dt_hours
        llf_cost_history.append(current_cost_llf)

        time_steps.append(t)

    # ---------------------------------------------------------------------
    # Fulfillment Calculation
    # ---------------------------------------------------------------------
    def calc_fulfillment(vehicles_dict, is_priority_filter):
        req = sum(
            v.target_energy_kwh
            for v in vehicles_dict.values()
            if v.is_priority == is_priority_filter
        )
        deliv = sum(
            v.energy_delivered
            for v in vehicles_dict.values()
            if v.is_priority == is_priority_filter
        )
        return (deliv / req * 100) if req > 0 else 100.0

    priority_ratio = (
        len([v for v in vehicles_aqps.values() if v.is_priority])
        / len(vehicles_aqps)
        * 100
    )

    print(f"[{scenario_label}] Simulation complete.")
    return {
        "fleet_size": fleet_size,
        "scenario_key": scenario_key,
        "scenario_label": scenario_label,
        "priority_ratio": priority_ratio,
        "time_steps": time_steps,
        "priority_arrivals": priority_arrivals,
        "aqps": {
            "priority_fulfillment": calc_fulfillment(vehicles_aqps, True),
            "non_priority_fulfillment": calc_fulfillment(vehicles_aqps, False),
            "cumulative_cost": aqps_cost_history,
            "preemptions": list(set(aqps_preemptions)),
        },
        "llf": {
            "priority_fulfillment": calc_fulfillment(vehicles_llf, True),
            "non_priority_fulfillment": calc_fulfillment(vehicles_llf, False),
            "cumulative_cost": llf_cost_history,
            "starvations": list(set(llf_starvations)),
        },
    }


def execute_configuration(fleet_size: int):
    filename = f"simulation_data_{fleet_size}EVs.json"
    try:
        with open(filename, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {filename} not found. Please run data_generator.py first.")
        return []

    env_data = data["environment"]
    scenarios = [
        ("S1_Baseline", "S1 (27%)"),
        ("S3_HighPriority", "S3 (50%)"),
        ("S6_PeakStress", "S6 (50% Peak Stress)"),
    ]

    results = []
    for sc_key, sc_label in scenarios:
        if sc_key in data["scenarios"]:
            ev_data = data["scenarios"][sc_key]
            res = run_scenario(fleet_size, sc_key, sc_label, ev_data, env_data)
            results.append(res)

    return results


# =============================================================================
# Plotting and Export Functions
# =============================================================================
def export_data_to_csv(results_45, results_90):
    all_results = results_45 + results_90
    if not all_results:
        return

    # Fulfillment
    with open(
        os.path.join(RESULTS_DIR, "fulfillment_data.csv"), mode="w", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Fleet Size",
                "Scenario Label",
                "Priority Pct",
                "Algorithm",
                "Priority Fulfillment",
                "Non-Priority Fulfillment",
            ]
        )
        for r in all_results:
            writer.writerow(
                [
                    r["fleet_size"],
                    r["scenario_label"],
                    r["priority_ratio"],
                    "AQPS",
                    r["aqps"]["priority_fulfillment"],
                    r["aqps"]["non_priority_fulfillment"],
                ]
            )
            writer.writerow(
                [
                    r["fleet_size"],
                    r["scenario_label"],
                    r["priority_ratio"],
                    "LLF",
                    r["llf"]["priority_fulfillment"],
                    r["llf"]["non_priority_fulfillment"],
                ]
            )

    # Cumulative Cost
    with open(
        os.path.join(RESULTS_DIR, "cumulative_cost_data.csv"), mode="w", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Fleet Size",
                "Scenario Label",
                "Time Step",
                "Hour",
                "AQPS Cost",
                "LLF Cost",
            ]
        )
        for r in all_results:
            for t, aqps_c, llf_c in zip(
                r["time_steps"],
                r["aqps"]["cumulative_cost"],
                r["llf"]["cumulative_cost"],
            ):
                writer.writerow(
                    [r["fleet_size"], r["scenario_label"], t, t * 0.25, aqps_c, llf_c]
                )


def plot_fulfillment_stacked(results_45, results_90):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    width = 0.35

    def plot_stacked_axis(ax, results, title):
        labels = [r["scenario_label"] for r in results]
        x = np.arange(len(labels))

        # Calculate Weighted Percentages for accurate stacking
        aqps_p_weighted, aqps_np_weighted = [], []
        llf_p_weighted, llf_np_weighted = [], []

        for r in results:
            p_ratio = r["priority_ratio"] / 100.0
            np_ratio = 1.0 - p_ratio

            # AQPS Bars
            aqps_p_weighted.append(r["aqps"]["priority_fulfillment"] * p_ratio)
            aqps_np_weighted.append(r["aqps"]["non_priority_fulfillment"] * np_ratio)

            # LLF Bars
            llf_p_weighted.append(r["llf"]["priority_fulfillment"] * p_ratio)
            llf_np_weighted.append(r["llf"]["non_priority_fulfillment"] * np_ratio)

        # Plot AQPS (Left Bar in group)
        ax.bar(
            x - width / 2,
            aqps_np_weighted,
            width,
            label="AQPS: Non-Priority Met",
            color="#D32F2F",
            alpha=0.9,
        )
        ax.bar(
            x - width / 2,
            aqps_p_weighted,
            width,
            bottom=aqps_np_weighted,
            label="AQPS: Priority Met",
            color="#2E7D32",
            alpha=0.9,
        )

        # Plot LLF (Right Bar in group)
        ax.bar(
            x + width / 2,
            llf_np_weighted,
            width,
            label="LLF: Non-Priority Met",
            color="#E57373",
            alpha=0.9,
        )
        ax.bar(
            x + width / 2,
            llf_p_weighted,
            width,
            bottom=llf_np_weighted,
            label="LLF: Priority Met",
            color="#81C784",
            alpha=0.9,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylabel(
            "Total Fleet Demand Fulfilled (%)", fontsize=12, fontweight="bold"
        )
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim([0, 110])
        # Add a baseline for 100% total demand
        ax.axhline(y=100, color="black", linestyle="--", alpha=0.5)

    plot_stacked_axis(ax1, results_45, "45 EVs Fleet (15 EVSEs)")
    plot_stacked_axis(ax2, results_90, "90 EVs Fleet (30 EVSEs)")

    # Deduplicate legend
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        fontsize=11,
        bbox_to_anchor=(0.5, -0.05),
    )

    fig.suptitle(
        "Total EV Fulfillment Breakdown: AQPS vs LLF",
        fontsize=16,
        fontweight="bold",
        y=1.05,
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_DIR, "fig_fulfillment_stacked.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


# def plot_priority_fulfillment(results_45, results_90):
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

#     width = 0.35

#     # --- Subplot 1: 45 EVs ---
#     labels_45 = [r["scenario_label"] for r in results_45]
#     aqps_45 = [r["aqps"]["priority_fulfillment"] for r in results_45]
#     llf_45 = [r["llf"]["priority_fulfillment"] for r in results_45]
#     x_45 = np.arange(len(labels_45))

#     ax1.bar(x_45 - width / 2, aqps_45, width, label="AQPS", color="#2E7D32", alpha=0.9)
#     ax1.bar(x_45 + width / 2, llf_45, width, label="LLF", color="#81C784", alpha=0.9)
#     ax1.set_xticks(x_45)
#     ax1.set_xticklabels(labels_45, fontsize=11)
#     ax1.set_ylabel("Priority Demand Met (%)", fontsize=12, fontweight="bold")
#     ax1.set_title("45 EVs Fleet (15 EVSEs)", fontsize=13, fontweight="bold")
#     ax1.grid(True, alpha=0.3, axis="y")
#     ax1.legend(fontsize=11)
#     ax1.set_ylim([0, 110])

#     # --- Subplot 2: 90 EVs ---
#     labels_90 = [r["scenario_label"] for r in results_90]
#     aqps_90 = [r["aqps"]["priority_fulfillment"] for r in results_90]
#     llf_90 = [r["llf"]["priority_fulfillment"] for r in results_90]
#     x_90 = np.arange(len(labels_90))

#     ax2.bar(x_90 - width / 2, aqps_90, width, label="AQPS", color="#1565C0", alpha=0.9)
#     ax2.bar(x_90 + width / 2, llf_90, width, label="LLF", color="#64B5F6", alpha=0.9)
#     ax2.set_xticks(x_90)
#     ax2.set_xticklabels(labels_90, fontsize=11)
#     ax2.set_ylabel("Priority Demand Met (%)", fontsize=12, fontweight="bold")
#     ax2.set_title("90 EVs Fleet (30 EVSEs)", fontsize=13, fontweight="bold")
#     ax2.grid(True, alpha=0.3, axis="y")
#     ax2.legend(fontsize=11)
#     ax2.set_ylim([0, 110])

#     fig.suptitle(
#         "Priority EV Fulfillment vs. Scenarios (AQPS vs LLF)",
#         fontsize=16,
#         fontweight="bold",
#     )
#     plt.tight_layout()
#     plt.savefig(os.path.join(RESULTS_DIR, "fig1_priority_fulfillment.png"), dpi=300)
#     plt.close()


# def plot_non_priority_fulfillment(results_45, results_90):
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

#     width = 0.35

#     # --- Subplot 1: 45 EVs ---
#     labels_45 = [
#         f"{r['scenario_key'].split('_')[0]}\n({100 - r['priority_ratio']:.0f}% Non-Priority)"
#         for r in results_45
#     ]
#     aqps_45 = [r["aqps"]["non_priority_fulfillment"] for r in results_45]
#     llf_45 = [r["llf"]["non_priority_fulfillment"] for r in results_45]
#     x_45 = np.arange(len(labels_45))

#     ax1.bar(x_45 - width / 2, aqps_45, width, label="AQPS", color="#D32F2F", alpha=0.9)
#     ax1.bar(x_45 + width / 2, llf_45, width, label="LLF", color="#E57373", alpha=0.9)
#     ax1.set_xticks(x_45)
#     ax1.set_xticklabels(labels_45, fontsize=11)
#     ax1.set_ylabel("Non-Priority Demand Met (%)", fontsize=12, fontweight="bold")
#     ax1.set_title("45 EVs Fleet (15 EVSEs)", fontsize=13, fontweight="bold")
#     ax1.grid(True, alpha=0.3, axis="y")
#     ax1.legend(fontsize=11)
#     ax1.set_ylim([0, 110])

#     # --- Subplot 2: 90 EVs ---
#     labels_90 = [
#         f"{r['scenario_key'].split('_')[0]}\n({100 - r['priority_ratio']:.0f}% Non-Priority)"
#         for r in results_90
#     ]
#     aqps_90 = [r["aqps"]["non_priority_fulfillment"] for r in results_90]
#     llf_90 = [r["llf"]["non_priority_fulfillment"] for r in results_90]
#     x_90 = np.arange(len(labels_90))

#     ax2.bar(x_90 - width / 2, aqps_90, width, label="AQPS", color="#F57C00", alpha=0.9)
#     ax2.bar(x_90 + width / 2, llf_90, width, label="LLF", color="#FFB74D", alpha=0.9)
#     ax2.set_xticks(x_90)
#     ax2.set_xticklabels(labels_90, fontsize=11)
#     ax2.set_ylabel("Non-Priority Demand Met (%)", fontsize=12, fontweight="bold")
#     ax2.set_title("90 EVs Fleet (30 EVSEs)", fontsize=13, fontweight="bold")
#     ax2.grid(True, alpha=0.3, axis="y")
#     ax2.legend(fontsize=11)
#     ax2.set_ylim([0, 110])

#     fig.suptitle(
#         "Non-Priority EV Fulfillment vs. Scenarios (AQPS vs LLF)",
#         fontsize=16,
#         fontweight="bold",
#     )
#     plt.tight_layout()
#     plt.savefig(os.path.join(RESULTS_DIR, "fig2_non_priority_fulfillment.png"), dpi=300)
#     plt.close()


def plot_cumulative_cost(results, fleet_size):
    plt.figure(figsize=(10, 6))
    colors = ["#4CAF50", "#2196F3", "#F44336"]

    for idx, r in enumerate(results):
        hours = [t * 0.25 for t in r["time_steps"]]
        c = colors[idx % len(colors)]
        plt.plot(
            hours,
            r["aqps"]["cumulative_cost"],
            linestyle="-",
            linewidth=2,
            color=c,
            label=f'AQPS: {r["scenario_label"]}',
        )
        plt.plot(
            hours,
            r["llf"]["cumulative_cost"],
            linestyle="--",
            linewidth=2,
            color=c,
            label=f'LLF: {r["scenario_label"]}',
        )

    plt.xlabel("Time of Day (Hours)", fontsize=12, fontweight="bold")
    plt.ylabel("Cumulative Energy Cost ($)", fontsize=12, fontweight="bold")
    plt.title(
        f"Total Charging Cost Over the Day ({fleet_size} EVs)",
        fontsize=14,
        fontweight="bold",
    )
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, ncol=2)

    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_DIR, f"fig3_cumulative_cost_{fleet_size}EVs.png"), dpi=300
    )
    plt.close()


def plot_preemption_vs_arrival(result):
    plt.figure(figsize=(12, 6))

    # Convert indices to hours
    aqps_preempts = sorted([t * 0.25 for t in result["aqps"]["preemptions"]])
    llf_starves = sorted([t * 0.25 for t in result["llf"]["starvations"]])
    priority_arrivals = sorted([t * 0.25 for t in result["priority_arrivals"]])

    # Create time array and cumulative counts
    time_axis = np.arange(0, 24.25, 0.25)

    cum_preempts = [sum(1 for p in aqps_preempts if p <= t) for t in time_axis]
    cum_starves = [sum(1 for s in llf_starves if s <= t) for t in time_axis]

    # Plot Cumulative Events as Step charts (Standard in Control systems)
    plt.step(
        time_axis,
        cum_preempts,
        where="post",
        color="#2E7D32",
        linewidth=3,
        label="Cumulative AQPS Preemptions (Layer 1 Action)",
    )
    plt.step(
        time_axis,
        cum_starves,
        where="post",
        color="#D32F2F",
        linewidth=3,
        linestyle="--",
        label="Cumulative LLF Priority Starvations (Failures)",
    )

    # Use secondary axis to show the "Stress Driver" (Priority Arrivals)
    ax2 = plt.twinx()
    ax2.hist(
        priority_arrivals,
        bins=24,
        range=(0, 24),
        alpha=0.2,
        color="#1565C0",
        label="Priority EV Arrival Volume",
    )
    ax2.set_ylabel(
        "Number of Priority Arrivals", color="#1565C0", fontsize=12, fontweight="bold"
    )

    # Formatting
    plt.xlim([0, 24])
    plt.xlabel("Time of Day (Hours)", fontsize=12, fontweight="bold")

    # Primary Y axis formatting
    ax = plt.gca()  # gets the twinx axis, need the primary
    fig = plt.gcf()
    ax1 = fig.axes[0]
    ax1.set_ylabel("Cumulative Event Occurrences", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    plt.title(
        f'Mechanism Validation: AQPS Preemption preventing Priority Starvation\n({result["scenario_label"]}, {result["fleet_size"]} EVs)',
        fontsize=14,
        fontweight="bold",
    )

    # Combine legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left", fontsize=11)

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            f'fig4_mechanism_validation_{result["scenario_key"]}_{result["fleet_size"]}EVs.png',
        ),
        dpi=300,
    )
    plt.close()


def main():
    print("Starting AQPC Scenario Simulations with AQPS vs LLF Comparison...")

    results_45 = execute_configuration(fleet_size=45)
    results_90 = execute_configuration(fleet_size=90)

    if not results_45 or not results_90:
        print("Missing required JSON data files. Simulation halted.")
        return

    print("Exporting data to CSV files...")
    export_data_to_csv(results_45, results_90)

    print("Generating Figures...")
    plot_fulfillment_stacked(results_45, results_90)
    # plot_priority_fulfillment(results_45, results_90)
    # plot_non_priority_fulfillment(results_45, results_90)

    plot_cumulative_cost(results_45, 45)
    plot_cumulative_cost(results_90, 90)

    # Extract S6 (Peak Stress) from 90 EVs configuration for the preemption plot to show maximum stress
    peak_stress_90 = next(
        (r for r in results_90 if r["scenario_key"] == "S6_PeakStress"), None
    )
    if peak_stress_90:
        plot_preemption_vs_arrival(peak_stress_90)

    print(f"All figures and data CSVs saved to directory: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
