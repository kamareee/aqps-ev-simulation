# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring

"""
AQPS vs LLF Simulation Studies — Publication-Quality Figures
Uses AQPSScheduler (QueueManager + TOUOptimizer/cvxpy two-layer).
Requires: cvxpy, numpy, pandas, matplotlib
"""

import os, json, csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from datetime import datetime
from collections import defaultdict

from src.aqps.scheduler import AQPSScheduler
from src.aqps.data_structures import Vehicle

RESULTS_DIR = "results_resubmission"
os.makedirs(RESULTS_DIR, exist_ok=True)

# =============================================================================
# Journal-Quality Color Palette (Tol's bright, colorblind-safe)
# =============================================================================
COLORS = {
    "aqps_priority": "#0077BB",
    "aqps_nonpriority": "#33BBEE",
    "llf_priority": "#EE7733",
    "llf_nonpriority": "#FFCC66",
    "gantt_window": "#DDDDDD",
    "gantt_np_charge": "#33BBEE",
    "gantt_p_charge": "#0077BB",
    "gantt_preempted": "#CC3311",
    "cost_lines": ["#0077BB", "#EE7733", "#009988"],
    "text": "#333333",
}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth": 1.2,
        "axes.grid": False,
        "text.color": COLORS["text"],
        "axes.labelcolor": COLORS["text"],
        "xtick.color": COLORS["text"],
        "ytick.color": COLORS["text"],
    }
)


# =============================================================================
# Baseline LLF Scheduler (from v1)
# =============================================================================
class LLFScheduler:
    def __init__(self, num_chargers, grid_limit):
        self.num_chargers = num_chargers
        self.grid_limit = grid_limit
        self.active_vehicles = []

    def step(self, time_idx, arriving, departing_ids, current_cap):
        self.active_vehicles = [
            v for v in self.active_vehicles if v.id not in departing_ids
        ]
        self.active_vehicles.extend(arriving)

        for v in self.active_vehicles:
            time_rem = (v.departure_time_idx - time_idx) * 0.25
            if time_rem <= 0:
                laxity = -999
            else:
                time_to_charge = v.energy_needed / max(v.max_charging_power_kw, 0.01)
                laxity = time_rem - time_to_charge
            v.laxity = laxity

        self.active_vehicles.sort(key=lambda x: x.laxity)
        commands = {}
        remaining_cap = current_cap

        candidates = self.active_vehicles[: self.num_chargers]
        for v in candidates:
            if remaining_cap <= 0:
                commands[v.id] = 0.0
                continue
            p_cmd = min(v.max_charging_power_kw, remaining_cap)
            if v.energy_needed <= 0.01:
                p_cmd = 0.0
            commands[v.id] = p_cmd
            remaining_cap -= p_cmd
            v.energy_delivered += p_cmd * 0.25
            v.energy_needed = max(0.0, v.target_energy_kwh - v.energy_delivered)

        return {"commands": commands, "grid_power": sum(commands.values())}


# =============================================================================
# Helpers
# =============================================================================
def parse_dt_to_idx(dt_str):
    dt = datetime.fromisoformat(dt_str)
    return int((dt.hour * 60 + dt.minute) // 15)


def create_vehicle_map(ev_data):
    vehicles = {}
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
        vehicles[v.id] = v
    return vehicles


# =============================================================================
# Simulation Runner (uses real AQPSScheduler + per-EV tracking for Gantt)
# =============================================================================
def run_scenario(fleet_size, scenario_key, scenario_label, ev_data, env_data):
    print(f"\n[{scenario_label}] Running {fleet_size} EVs...")

    horizon = len(env_data["tou_tariff"])
    dt_hours = env_data.get("step_delta", 0.25)
    num_chargers = max(3, int(fleet_size / 3))
    transformer_limit_kw = (fleet_size / 45) * 150.0

    config = {
        "station": {"num_chargers": num_chargers},
        "grid": {"transformer_limit_kw": transformer_limit_kw},
        "simulation": {"dt_hours": dt_hours, "planning_horizon_steps": 12},
    }

    # Initialize Schedulers
    aqps = AQPSScheduler(config)
    llf = LLFScheduler(num_chargers, transformer_limit_kw)

    # Distinct vehicle sets
    vehicles_aqps = create_vehicle_map(ev_data)
    vehicles_llf = create_vehicle_map(ev_data)

    time_steps = []
    aqps_cost_history, llf_cost_history = [], []
    current_cost_aqps, current_cost_llf = 0.0, 0.0

    # Per-EV timeline tracking (for Gantt chart)
    ev_power_timeline = defaultdict(lambda: np.zeros(horizon))  # AQPS
    llf_power_timeline = defaultdict(lambda: np.zeros(horizon))  # LLF
    ev_info = {}
    for v in vehicles_aqps.values():
        ev_info[v.id] = {
            "arrival": v.arrival_time_idx,
            "departure": v.departure_time_idx,
            "is_priority": v.is_priority,
            "target_energy": v.target_energy_kwh,
        }

    # Preemption event tracking (AQPS)
    preemption_events = []
    prev_cmds_aqps = {}

    # Starvation event tracking (LLF — priority EV needs charge but gets 0)
    llf_starvation_events = []

    for t in range(horizon):
        tou = np.array(env_data["tou_tariff"][t:] + [env_data["tou_tariff"][-1]] * t)
        pv = np.array(env_data["pv_forecast"][t:] + [0.0] * t)
        bess = np.array([0.0] * len(tou))
        pv_now = env_data["pv_forecast"][t] if t < len(env_data["pv_forecast"]) else 0.0
        current_cap_llf = transformer_limit_kw + pv_now

        # -----------------------------------------------------------------
        # AQPS (Two-Layer: QueueManager + TOUOptimizer)
        # -----------------------------------------------------------------
        arriving_aqps = [v for v in vehicles_aqps.values() if v.arrival_time_idx == t]
        departing_aqps = [
            v.id for v in vehicles_aqps.values() if v.departure_time_idx == t
        ]

        step_res_aqps = aqps.step(t, arriving_aqps, departing_aqps, tou, pv, bess)
        cmds_aqps = step_res_aqps.get("commands", {})

        # Record per-EV power timeline
        for v_id, p_cmd in cmds_aqps.items():
            if v_id in vehicles_aqps:
                ev_power_timeline[v_id][t] = p_cmd

        # Update AQPS energy state
        for v_id, p_cmd in cmds_aqps.items():
            if v_id in vehicles_aqps:
                vehicles_aqps[v_id].energy_delivered += p_cmd * dt_hours

        # Detect preemption events (NP was charging -> now cut off when priority arrives)
        priority_arriving_now = [v for v in arriving_aqps if v.is_priority]
        for v in vehicles_aqps.values():
            if not v.is_priority and v.arrival_time_idx <= t < v.departure_time_idx:
                still_needs = (v.target_energy_kwh - v.energy_delivered) > 0.5
                was_charging = prev_cmds_aqps.get(v.id, 0) > 0.5
                is_cut = cmds_aqps.get(v.id, 0) < 0.5
                if was_charging and is_cut and still_needs and priority_arriving_now:
                    for pv_ev in priority_arriving_now:
                        preemption_events.append(
                            {
                                "time": t,
                                "priority_id": pv_ev.id,
                                "preempted_id": v.id,
                            }
                        )
        prev_cmds_aqps = dict(cmds_aqps)

        grid_p_aqps = max(0.0, sum(cmds_aqps.values()) - pv_now)
        current_cost_aqps += grid_p_aqps * tou[0] * dt_hours
        aqps_cost_history.append(current_cost_aqps)

        # -----------------------------------------------------------------
        # LLF Baseline
        # -----------------------------------------------------------------
        arriving_llf = [v for v in vehicles_llf.values() if v.arrival_time_idx == t]
        departing_llf = [
            v.id for v in vehicles_llf.values() if v.departure_time_idx == t
        ]

        step_res_llf = llf.step(t, arriving_llf, departing_llf, current_cap_llf)
        cmds_llf = step_res_llf.get("commands", {})

        # Record LLF per-EV power timeline
        for v_id, p_cmd in cmds_llf.items():
            if v_id in vehicles_llf:
                llf_power_timeline[v_id][t] = p_cmd

        # Detect LLF starvation (priority EV active + needs energy but gets 0 power)
        for v in vehicles_llf.values():
            if v.is_priority and v.arrival_time_idx <= t < v.departure_time_idx:
                if v.energy_needed > 0.5 and cmds_llf.get(v.id, 0) < 0.5:
                    llf_starvation_events.append(
                        {
                            "time": t,
                            "starved_id": v.id,
                        }
                    )

        grid_p_llf = max(0.0, step_res_llf["grid_power"] - pv_now)
        current_cost_llf += grid_p_llf * tou[0] * dt_hours
        llf_cost_history.append(current_cost_llf)

        time_steps.append(t)

    # -----------------------------------------------------------------
    # Fulfillment Calculation
    # -----------------------------------------------------------------
    def calc_fulfillment(vdict, is_p):
        req = sum(v.target_energy_kwh for v in vdict.values() if v.is_priority == is_p)
        deliv = sum(
            min(v.energy_delivered, v.target_energy_kwh)
            for v in vdict.values()
            if v.is_priority == is_p
        )
        return (deliv / req * 100) if req > 0 else 100.0

    priority_ratio = (
        len([v for v in vehicles_aqps.values() if v.is_priority])
        / len(vehicles_aqps)
        * 100
    )

    print(f"[{scenario_label}] Done. Preemption events: {len(preemption_events)}")
    return {
        "fleet_size": fleet_size,
        "scenario_key": scenario_key,
        "scenario_label": scenario_label,
        "priority_ratio": priority_ratio,
        "time_steps": time_steps,
        "aqps": {
            "priority_fulfillment": calc_fulfillment(vehicles_aqps, True),
            "non_priority_fulfillment": calc_fulfillment(vehicles_aqps, False),
            "cumulative_cost": aqps_cost_history,
        },
        "llf": {
            "priority_fulfillment": calc_fulfillment(vehicles_llf, True),
            "non_priority_fulfillment": calc_fulfillment(vehicles_llf, False),
            "cumulative_cost": llf_cost_history,
        },
        "ev_power_timeline": dict(ev_power_timeline),
        "llf_power_timeline": dict(llf_power_timeline),
        "ev_info": ev_info,
        "preemption_events": preemption_events,
        "llf_starvation_events": llf_starvation_events,
        "horizon": horizon,
    }


def execute_configuration(fleet_size):
    filename = f"simulation_data_{fleet_size}EVs.json"
    try:
        with open(filename, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {filename} not found. Run data_generator.py first.")
        return []

    env_data = data["environment"]
    scenarios = [
        ("S1_Baseline", "S1 (27%)"),
        ("S3_HighPriority", "S3 (50%)"),
        ("S6_PeakStress", "S6 (50% Peak)"),
    ]

    results = []
    for sc_key, sc_label in scenarios:
        if sc_key in data["scenarios"]:
            res = run_scenario(
                fleet_size, sc_key, sc_label, data["scenarios"][sc_key], env_data
            )
            results.append(res)
    return results


# =============================================================================
# FIGURE 1: Fulfillment Stacked Bar with Priority % Labels
# =============================================================================
def plot_fulfillment_stacked(results_45, results_90):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0))
    width = 0.32

    def plot_axis(ax, results, title):
        labels = [r["scenario_label"] for r in results]
        x = np.arange(len(labels))

        aqps_p_w, aqps_np_w = [], []
        llf_p_w, llf_np_w = [], []
        aqps_p_raw, llf_p_raw = [], []

        for r in results:
            p_ratio = r["priority_ratio"] / 100.0
            np_ratio = 1.0 - p_ratio

            aqps_p_val = r["aqps"]["priority_fulfillment"] * p_ratio
            aqps_np_val = r["aqps"]["non_priority_fulfillment"] * np_ratio
            llf_p_val = r["llf"]["priority_fulfillment"] * p_ratio
            llf_np_val = r["llf"]["non_priority_fulfillment"] * np_ratio

            aqps_p_w.append(aqps_p_val)
            aqps_np_w.append(aqps_np_val)
            llf_p_w.append(llf_p_val)
            llf_np_w.append(llf_np_val)
            aqps_p_raw.append(r["aqps"]["priority_fulfillment"])
            llf_p_raw.append(r["llf"]["priority_fulfillment"])

        # AQPS bars
        ax.bar(
            x - width / 2,
            aqps_np_w,
            width,
            color=COLORS["aqps_nonpriority"],
            edgecolor="white",
            linewidth=0.3,
        )
        ax.bar(
            x - width / 2,
            aqps_p_w,
            width,
            bottom=aqps_np_w,
            color=COLORS["aqps_priority"],
            edgecolor="white",
            linewidth=0.3,
        )

        # LLF bars
        ax.bar(
            x + width / 2,
            llf_np_w,
            width,
            color=COLORS["llf_nonpriority"],
            edgecolor="white",
            linewidth=0.3,
        )
        ax.bar(
            x + width / 2,
            llf_p_w,
            width,
            bottom=llf_np_w,
            color=COLORS["llf_priority"],
            edgecolor="white",
            linewidth=0.3,
        )

        # Priority fulfillment % labels on priority segments
        for i in range(len(x)):
            y_center_a = aqps_np_w[i] + aqps_p_w[i] / 2
            ax.text(
                x[i] - width / 2,
                y_center_a,
                f"{aqps_p_raw[i]:.0f}%",
                ha="center",
                va="center",
                fontsize=6,
                fontweight="bold",
                color="white",
            )

            y_center_l = llf_np_w[i] + llf_p_w[i] / 2
            ax.text(
                x[i] + width / 2,
                y_center_l,
                f"{llf_p_raw[i]:.0f}%",
                ha="center",
                va="center",
                fontsize=6,
                fontweight="bold",
                color="white" if llf_p_raw[i] > 50 else COLORS["text"],
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Fleet Demand Fulfilled (%)")
        ax.set_title(title, fontweight="bold")
        ax.set_ylim([0, 115])
        ax.axhline(y=100, color="#999999", linestyle="--", linewidth=0.5, alpha=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, alpha=0.2, linewidth=0.4)

    plot_axis(ax1, results_45, "45 EVs (15 EVSEs)")
    plot_axis(ax2, results_90, "90 EVs (30 EVSEs)")

    legend_elements = [
        mpatches.Patch(
            facecolor=COLORS["aqps_priority"], edgecolor="white", label="AQPS: Priority"
        ),
        mpatches.Patch(
            facecolor=COLORS["aqps_nonpriority"],
            edgecolor="white",
            label="AQPS: Non-Priority",
        ),
        mpatches.Patch(
            facecolor=COLORS["llf_priority"], edgecolor="white", label="LLF: Priority"
        ),
        mpatches.Patch(
            facecolor=COLORS["llf_nonpriority"],
            edgecolor="white",
            label="LLF: Non-Priority",
        ),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.06),
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    path = os.path.join(RESULTS_DIR, "fig_fulfillment_stacked.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  Saved: {path}")


# =============================================================================
# FIGURE 2: Cumulative Cost Comparison
# =============================================================================
def plot_cumulative_cost(results, fleet_size):
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    colors = COLORS["cost_lines"]

    for idx, r in enumerate(results):
        hours = [t * 0.25 for t in r["time_steps"]]
        c = colors[idx % len(colors)]
        ax.plot(
            hours,
            r["aqps"]["cumulative_cost"],
            "-",
            color=c,
            label=f'AQPS: {r["scenario_label"]}',
        )
        ax.plot(
            hours,
            r["llf"]["cumulative_cost"],
            "--",
            color=c,
            label=f'LLF: {r["scenario_label"]}',
            alpha=0.7,
        )

    ax.set_xlabel("Time of Day (Hours)")
    ax.set_ylabel("Cumulative Cost ($)")
    ax.set_title(f"Charging Cost ({fleet_size} EVs)", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.2, linewidth=0.4)
    ax.legend(fontsize=6, ncol=2, frameon=False, loc="upper left")

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"fig_cumulative_cost_{fleet_size}EVs.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  Saved: {path}")


# =============================================================================
# FIGURE 3: Preemption Gantt Chart (per scenario/fleet)
# =============================================================================
def _draw_gantt_panel(
    ax,
    involved_list,
    ev_info,
    timeline_dict,
    horizon,
    t_min,
    t_max,
    event_markers,
    event_connections,
    title,
):
    """Helper: draw one Gantt panel (used for both AQPS and LLF)."""
    n_evs = len(involved_list)
    bar_height = 0.6

    for yi, eid in enumerate(involved_list):
        info = ev_info[eid]
        arr = info["arrival"]
        dep = info["departure"]
        is_p = info["is_priority"]

        # Connection window
        ax.barh(
            yi,
            (dep - arr) * 0.25,
            left=arr * 0.25,
            height=bar_height,
            color=COLORS["gantt_window"],
            edgecolor="none",
            zorder=1,
        )

        # Charging blocks
        timeline = timeline_dict.get(eid, np.zeros(horizon))
        charge_color = COLORS["gantt_p_charge"] if is_p else COLORS["gantt_np_charge"]

        t_start = None
        for t in range(t_min, t_max):
            if t < len(timeline) and timeline[t] > 0.5:
                if t_start is None:
                    t_start = t
            else:
                if t_start is not None:
                    ax.barh(
                        yi,
                        (t - t_start) * 0.25,
                        left=t_start * 0.25,
                        height=bar_height * 0.7,
                        color=charge_color,
                        edgecolor="none",
                        zorder=2,
                        alpha=0.9,
                    )
                    t_start = None
        if t_start is not None:
            ax.barh(
                yi,
                (t_max - t_start) * 0.25,
                left=t_start * 0.25,
                height=bar_height * 0.7,
                color=charge_color,
                edgecolor="none",
                zorder=2,
                alpha=0.9,
            )

        # Event markers (preemption or starvation)
        for pt in event_markers.get(eid, []):
            ax.plot(
                pt * 0.25,
                yi,
                marker="v",
                color=COLORS["gantt_preempted"],
                markersize=5,
                zorder=4,
                markeredgecolor="white",
                markeredgewidth=0.3,
            )

    # Connection lines between event pairs
    for id_a, id_b, t_ev in event_connections:
        if id_a in involved_list and id_b in involved_list:
            yi_a = involved_list.index(id_a)
            yi_b = involved_list.index(id_b)
            ax.plot(
                [t_ev * 0.25, t_ev * 0.25],
                [yi_a, yi_b],
                "--",
                color=COLORS["gantt_preempted"],
                linewidth=0.5,
                alpha=0.5,
                zorder=3,
            )

    # Y-axis labels
    y_labels = []
    for eid in involved_list:
        tag = "P" if ev_info[eid]["is_priority"] else "NP"
        short_id = eid.replace("EV_", "")
        y_labels.append(f"{short_id} [{tag}]")

    ax.set_yticks(range(n_evs))
    ax.set_yticklabels(y_labels, fontsize=5.5)
    ax.set_xlim(t_min * 0.25, t_max * 0.25)
    ax.set_ylim(-0.5, n_evs - 0.5)
    ax.invert_yaxis()
    ax.set_title(title, fontweight="bold", fontsize=7.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.grid(True, alpha=0.15, linewidth=0.4)


def plot_preemption_gantt(result):
    """Two-row Gantt: AQPS (top, with preemption) vs LLF (bottom, with starvation).
    Same EVs shown in both panels for direct comparison."""

    preemptions = result["preemption_events"]
    starvations = result["llf_starvation_events"]

    if not preemptions and not starvations:
        print(
            f"  No preemption/starvation events for {result['scenario_label']} "
            f"{result['fleet_size']}EVs — skipping Gantt."
        )
        return

    ev_info = result["ev_info"]
    aqps_timeline = result["ev_power_timeline"]
    llf_timeline = result["llf_power_timeline"]
    horizon = result["horizon"]

    # Collect involved EVs from both AQPS preemptions and LLF starvations
    pairs_seen = set()
    ordered_ids = []
    for pe in sorted(preemptions, key=lambda x: x["time"]):
        pair = (pe["preempted_id"], pe["priority_id"])
        if pair not in pairs_seen:
            pairs_seen.add(pair)
            if pe["preempted_id"] not in ordered_ids:
                ordered_ids.append(pe["preempted_id"])
            if pe["priority_id"] not in ordered_ids:
                ordered_ids.append(pe["priority_id"])

    # Also include starved priority EVs from LLF
    for se in starvations:
        if se["starved_id"] not in ordered_ids:
            ordered_ids.append(se["starved_id"])

    involved_list = ordered_ids[:10]
    n_evs = len(involved_list)
    if n_evs == 0:
        return

    # Time range
    all_arrivals = [ev_info[eid]["arrival"] for eid in involved_list]
    all_departures = [ev_info[eid]["departure"] for eid in involved_list]
    t_min = max(0, min(all_arrivals) - 4)
    t_max = min(horizon, max(all_departures) + 4)

    # --- Build event marker dicts ---
    # AQPS: preemption markers on preempted NP EVs
    aqps_markers = defaultdict(list)
    aqps_connections = []
    for pe in preemptions:
        if pe["preempted_id"] in involved_list:
            aqps_markers[pe["preempted_id"]].append(pe["time"])
        aqps_connections.append((pe["preempted_id"], pe["priority_id"], pe["time"]))

    # LLF: starvation markers on starved priority EVs
    llf_markers = defaultdict(list)
    for se in starvations:
        if se["starved_id"] in involved_list:
            llf_markers[se["starved_id"]].append(se["time"])

    # --- Create 2-row figure ---
    fig_height = max(3.0, 0.3 * n_evs * 2 + 1.5)
    fig, (ax_aqps, ax_llf) = plt.subplots(2, 1, figsize=(3.5, fig_height), sharex=True)

    _draw_gantt_panel(
        ax_aqps,
        involved_list,
        ev_info,
        aqps_timeline,
        horizon,
        t_min,
        t_max,
        aqps_markers,
        aqps_connections,
        f"AQPS (Preemption)",
    )

    _draw_gantt_panel(
        ax_llf,
        involved_list,
        ev_info,
        llf_timeline,
        horizon,
        t_min,
        t_max,
        llf_markers,
        [],
        f"LLF (Priority Starvation)",
    )

    ax_llf.set_xlabel("Time of Day (Hours)")

    fig.suptitle(
        f"Scheduling Comparison — {result['scenario_label']}, {result['fleet_size']} EVs",
        fontweight="bold",
        fontsize=8.5,
        y=1.02,
    )

    # Shared legend below
    legend_elements = [
        mpatches.Patch(facecolor=COLORS["gantt_window"], label="Connected"),
        mpatches.Patch(facecolor=COLORS["gantt_p_charge"], label="Priority Charging"),
        mpatches.Patch(
            facecolor=COLORS["gantt_np_charge"], label="Non-Priority Charging"
        ),
        Line2D(
            [0],
            [0],
            marker="v",
            color="w",
            markerfacecolor=COLORS["gantt_preempted"],
            markersize=5,
            label="Preemption / Starvation",
        ),
    ]
    fig.legend(
        handles=legend_elements,
        fontsize=5.5,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.03),
    )

    plt.tight_layout(rect=[0, 0.06, 1, 1.0])
    safe_label = result["scenario_key"]
    path = os.path.join(
        RESULTS_DIR, f"fig_preemption_gantt_{safe_label}_{result['fleet_size']}EVs.png"
    )
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  Saved: {path}")


# =============================================================================
# CSV Export
# =============================================================================
def export_data_to_csv(all_results):
    if not all_results:
        return

    with open(os.path.join(RESULTS_DIR, "fulfillment_data.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "Fleet",
                "Scenario",
                "Priority%",
                "Algo",
                "PriorityFulfill",
                "NonPriorityFulfill",
            ]
        )
        for r in all_results:
            for algo in ["aqps", "llf"]:
                w.writerow(
                    [
                        r["fleet_size"],
                        r["scenario_label"],
                        r["priority_ratio"],
                        algo.upper(),
                        r[algo]["priority_fulfillment"],
                        r[algo]["non_priority_fulfillment"],
                    ]
                )

    with open(
        os.path.join(RESULTS_DIR, "cumulative_cost_data.csv"), "w", newline=""
    ) as f:
        w = csv.writer(f)
        w.writerow(["Fleet", "Scenario", "Step", "Hour", "AQPS_Cost", "LLF_Cost"])
        for r in all_results:
            for t, ac, lc in zip(
                r["time_steps"],
                r["aqps"]["cumulative_cost"],
                r["llf"]["cumulative_cost"],
            ):
                w.writerow([r["fleet_size"], r["scenario_label"], t, t * 0.25, ac, lc])

    print(f"  CSV files saved to {RESULTS_DIR}/")


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 60)
    print("AQPS vs LLF — Publication Figure Generation")
    print("Uses AQPSScheduler (QueueManager + TOUOptimizer/cvxpy)")
    print("=" * 60)

    results_45 = execute_configuration(fleet_size=45)
    results_90 = execute_configuration(fleet_size=90)

    if not results_45 or not results_90:
        print("Missing JSON data files. Run data_generator.py first.")
        return

    all_results = results_45 + results_90

    print("\nExporting CSV data...")
    export_data_to_csv(all_results)

    print("\nGenerating figures...")

    # Fig 1: Fulfillment stacked bar
    plot_fulfillment_stacked(results_45, results_90)

    # Fig 2: Cumulative cost
    plot_cumulative_cost(results_45, 45)
    plot_cumulative_cost(results_90, 90)

    # Fig 3: Preemption Gantt (all combos)
    for r in all_results:
        plot_preemption_gantt(r)

    print(f"\nAll outputs saved to: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
