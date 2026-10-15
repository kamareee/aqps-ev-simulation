# Save directly as: generate_all_figures.py
"""
WP7 — Publication-Quality Figures & EPSR LaTeX Table Generator.

Style specifications:
- Paul Tol bright colorblind-safe palette
- Serif fonts, 300 DPI, 3.5" (single-column) or 7.0" (double-column) widths
- Legends placed cleanly below plots
- Option A for Fig. 3 (Grouped direct 0-100% Priority and Non-Priority Fulfillment bars,
  fixing the stacked-vs-label mismatch and updating titles to 16 / 35 EVSEs).
- Regenerates Figs. 6 & 7 (Per-vehicle schedules / Preemption Gantt for S2 & S3).
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from collections import defaultdict

FIG_DIR = "figs_epsr"
TAB_DIR = "tables_epsr"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

# Paul Tol Bright Colorblind-Safe Palette
TOL_BRIGHT = {
    "blue": "#0077BB",
    "cyan": "#33BBEE",
    "teal": "#009988",
    "orange": "#EE7733",
    "red": "#CC3311",
    "magenta": "#EE3377",
    "yellow": "#CCBB44",
    "grey": "#BBBBBB",
    "dark": "#333333",
}

METHOD_COLORS = {
    "AQPS-RT-ET": "#0077BB",
    "AQPS-RT": "#33BBEE",
    "AQPS-static": "#009988",
    "MPC-only": "#EE3377",
    "EDF": "#AA3377",
    "LLF": "#EE7733",
    "Offline-Optimal": "#333333",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9.5,
    "legend.fontsize": 7,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.25,
})


def plot_fig2_architecture():
    """Fig. 2: Three-Layer AQPS + Operator Impact-Preview Loop Diagram."""
    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.2)
    ax.axis("off")

    boxes = [
        (0.2, 2.5, 2.1, 1.3, "#E8F4F8", "#0077BB",
         "Layer 0: Priority Selection\n• Session $E_i^{req}$ (kWh)\n• ATC Score $\\pi_i(t)$\n• QP Bisection Check"),
        (2.8, 2.5, 2.2, 1.3, "#E8F8F5", "#009988",
         "Layer 1: Multi-Queue ($Q_k^{act}, Q_k^p, Q_k^n$)\n• Laxity NP Preemption\n• Safe Release & $Q_k^{act}$ FIFO Rotation"),
        (5.5, 2.5, 2.2, 1.3, "#FEF9E7", "#EE7733",
         "Layer 2: Event-Triggered MPC\n• Receding-Horizon QP\n• Plan Cache $X^*$ ($T_{max}$)\n• Gross Limit $\\sum x_{i,t} \\leq P_{site}^{max}$"),
        (8.1, 2.5, 1.7, 1.3, "#F4ECF7", "#AA3377",
         "Depot EVSEs ($M$)\n• Pilot Signals $x_{i,k}$\n• 20 kW PV + BESS\n• 3-Tier TOU Tariff"),
        (2.8, 0.35, 4.9, 1.25, "#FDEDEC", "#CC3311",
         "Human-in-the-Loop Operator Impact Preview (FMS Integration)\n• 3 What-If QP Options (Recommended / Keep All / Downgrade)\n• Provisional Execution Unless Vetoed (Parasuraman et al., 2000)"),
    ]
    for x, y, w, h, fc, ec, text in boxes:
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.08", facecolor=fc, edgecolor=ec, linewidth=1.2
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7, color="#222222")

    arrow_kw = dict(arrowstyle="->,head_width=0.22,head_length=0.22", lw=1.1, color="#333333")
    ax.annotate("", xy=(2.8, 3.15), xytext=(2.3, 3.15), arrowprops=arrow_kw)
    ax.annotate("", xy=(5.5, 3.15), xytext=(5.0, 3.15), arrowprops=arrow_kw)
    ax.annotate("", xy=(8.1, 3.15), xytext=(7.7, 3.15), arrowprops=arrow_kw)

    ax.annotate(
        "Shortfall $>0$ Feedback",
        xy=(1.25, 2.5),
        xytext=(6.2, 2.5),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.28", lw=0.9, ls="--", color="#0077BB"),
        fontsize=6.5,
        color="#0077BB",
        ha="center",
    )
    ax.annotate(
        "Escalation / Preview",
        xy=(4.2, 1.6),
        xytext=(4.2, 2.5),
        arrowprops=dict(arrowstyle="->", lw=1.0, color="#CC3311"),
        fontsize=6.5,
        color="#CC3311",
    )
    ax.annotate(
        "Operator Choice",
        xy=(6.4, 2.5),
        xytext=(6.4, 1.6),
        arrowprops=dict(arrowstyle="->", lw=1.0, color="#CC3311"),
        fontsize=6.5,
        color="#CC3311",
    )

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig2_system_architecture.png"), dpi=300)
    plt.close()


def plot_fig3_fulfillment_grouped(df_e1: pd.DataFrame):
    """
    Fig. 3 (Option A - R1.5 & R3.2 Fixed):
    Direct 0-100% Priority and Non-Priority Demand Fulfillment (%) across key methods.
    """
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6), sharey=True)
    methods_show = ["AQPS-RT-ET", "AQPS-static", "MPC-only", "EDF", "LLF"]
    width = 0.15

    for col_idx, (fleet, evse) in enumerate([(45, 16), (90, 35)]):
        sub = df_e1[df_e1["fleet_size"] == fleet]
        scenarios = ["S1 (27%)", "S2 (50%)", "S3 (50% Peak)"]
        x = np.arange(len(scenarios))

        for row_idx, (metric_col, row_title) in enumerate([
            ("priority_fulfillment_pct", "Priority Demand Fulfilled (%)"),
            ("non_priority_fulfillment_pct", "Non-Priority Demand Fulfilled (%)"),
        ]):
            ax = axes[row_idx, col_idx]
            for m_idx, m in enumerate(methods_show):
                vals = []
                for sc in scenarios:
                    row = sub[(sub["scenario_label"] == sc) & (sub["method"] == m)]
                    vals.append(float(row[metric_col].iloc[0]) if len(row) > 0 else 0.0)
                offset = (m_idx - len(methods_show) / 2 + 0.5) * width
                bars = ax.bar(
                    x + offset, vals, width, label=m, color=METHOD_COLORS[m], edgecolor="white", linewidth=0.3
                )
                if m in ["AQPS-RT-ET", "LLF"]:
                    for b, v in zip(bars, vals):
                        ax.text(
                            b.get_x() + b.get_width() / 2,
                            v + 1.5,
                            f"{v:.0f}%",
                            ha="center",
                            va="bottom",
                            fontsize=5.5,
                            fontweight="bold",
                        )

            if row_idx == 0:
                ax.set_title(f"{fleet} EVs ({evse} EVSEs)", fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(row_title)
            ax.set_xticks(x)
            ax.set_xticklabels(scenarios)
            ax.set_ylim(0, 114)
            ax.axhline(100, color="#888888", ls="--", lw=0.5, alpha=0.7)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.yaxis.grid(True, alpha=0.2)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.03))
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(os.path.join(FIG_DIR, "fig_fulfillment_stacked.png"), dpi=300)
    plt.close()


def plot_figs4_5_cumulative_cost():
    """Figs. 4 & 5: Cumulative Charging Cost over 24h for 45 EVs (16 EVSEs) and 90 EVs (35 EVSEs)."""
    for fleet, evse in [(45, 16), (90, 35)]:
        fig, ax = plt.subplots(figsize=(3.5, 2.7))
        sc_colors = {"S1_Baseline": "#009988", "S3_HighPriority": "#0077BB", "S6_PeakStress": "#CC3311"}
        for sc_key, sc_label in [("S1_Baseline", "S1"), ("S3_HighPriority", "S2"), ("S6_PeakStress", "S3")]:
            for m, ls in [("AQPS-RT-ET", "-"), ("MPC-only", "-."), ("LLF", "--")]:
                p = os.path.join("results", "E1_main", "raw", f"{m}_{sc_key}_{fleet}EVs.json")
                if os.path.exists(p):
                    with open(p) as f:
                        r = json.load(f)
                    hours = np.arange(len(r["cumulative_cost_trace"])) * 0.25 + 6.0
                    ax.plot(
                        hours,
                        r["cumulative_cost_trace"],
                        ls=ls,
                        color=sc_colors[sc_key],
                        label=f"{m} ({sc_label})",
                        alpha=0.9 if ls == "-" else 0.7,
                    )

        ax.set_xlabel("Time of Day (Hours from 06:00)")
        ax.set_ylabel("Cumulative Cost ($)")
        ax.set_title(f"Charging Cost — {fleet} EVs ({evse} EVSEs)", fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, alpha=0.2)
        ax.legend(fontsize=5.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f"fig_cumulative_cost_{fleet}EVs.png"), dpi=300)
        plt.close()


def plot_figs6_7_preemption_gantt():
    """Figs. 6 & 7: Per-vehicle scheduling Gantt comparison (AQPS-RT-ET vs LLF) for S2 & S3 (90 EVs)."""
    with open("simulation_data_90EVs.json") as f:
        data90 = json.load(f)

    for sc_key, safe_tag, title_tag in [
        ("S3_HighPriority", "S2", "S2 (50% Contention)"),
        ("S6_PeakStress", "S3", "S3 (50% Peak Stress)"),
    ]:
        p_aqps = os.path.join("results", "E1_main", "raw", f"AQPS-RT-ET_{sc_key}_90EVs.json")
        p_llf = os.path.join("results", "E1_main", "raw", f"LLF_{sc_key}_90EVs.json")
        if not (os.path.exists(p_aqps) and os.path.exists(p_llf)):
            continue
        with open(p_aqps) as fa, open(p_llf) as fl:
            res_a = json.load(fa)
            res_l = json.load(fl)

        ev_list = data90["scenarios"][sc_key]
        ev_info = {
            ev["id"]: {
                "arrival": int(ev["arrival_idx"]),
                "departure": int(ev["departure_idx"]),
                "is_priority": (ev["priority"] == "High"),
            }
            for ev in ev_list
        }

        # Pick 1 preempted NP EV and 2 starved Priority EVs
        np_cands = [e["preempted_id"] for e in res_a.get("preemption_events", []) if e["preempted_id"] in ev_info]
        p_cands = [e["starved_id"] for e in res_l.get("starvation_events", []) if e["starved_id"] in ev_info]
        selected = []
        for vid in np_cands:
            if vid not in selected:
                selected.append(vid)
                break
        for vid in p_cands:
            if vid not in selected and len(selected) < 3:
                selected.append(vid)
        if len(selected) < 3:
            for vid, inf in ev_info.items():
                if inf["is_priority"] and vid not in selected:
                    selected.append(vid)
                    if len(selected) == 3:
                        break

        fig, ax = plt.subplots(figsize=(3.5, 3.6))
        GROUP = 2.2
        ytick_pos, ytick_lab = [], []
        t_min = max(0, min(ev_info[vid]["arrival"] for vid in selected) - 2)
        t_max = min(96, max(ev_info[vid]["departure"] for vid in selected) + 2)

        for i, vid in enumerate(selected):
            inf = ev_info[vid]
            arr, dep, is_p = inf["arrival"], inf["departure"], inf["is_priority"]
            y_a = i * GROUP
            y_l = i * GROUP + 0.85
            tag = "P" if is_p else "NP"
            short_id = vid.replace("EV_", "")
            ytick_pos.extend([y_a, y_l])
            ytick_lab.extend([f"{short_id} [{tag}]\nAQPS", f"{short_id} [{tag}]\nLLF"])

            for y_row, tl_dict in [(y_a, res_a["ev_power_timeline"]), (y_l, res_l["ev_power_timeline"])]:
                ax.barh(
                    y_row,
                    (dep - arr) * 0.25,
                    left=6.0 + arr * 0.25,
                    height=0.6,
                    color="#DDDDDD",
                    edgecolor="none",
                    zorder=1,
                )
                tl = tl_dict.get(vid, [0.0] * 96)
                c_bar = "#0077BB" if is_p else "#33BBEE"
                for t in range(arr, min(dep, len(tl))):
                    if tl[t] > 0.5:
                        ax.barh(y_row, 0.25, left=6.0 + t * 0.25, height=0.44, color=c_bar, edgecolor="none", zorder=2)
                    elif (y_row == y_l and is_p) or (y_row == y_a and not is_p and t > arr):
                        ax.plot(6.0 + (t + 0.5) * 0.25, y_row, marker="v", color="#CC3311", markersize=3.5, zorder=3)

        ax.set_yticks(ytick_pos)
        ax.set_yticklabels(ytick_lab, fontsize=6.5)
        ax.set_xlim(6.0 + t_min * 0.25, 6.0 + t_max * 0.25)
        ax.invert_yaxis()
        ax.set_xlabel("Time of Day (Hours)")
        ax.set_title(f"Schedule Comparison — {title_tag}, 90 EVs", fontweight="bold", fontsize=8.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.xaxis.grid(True, alpha=0.2)

        legend_el = [
            mpatches.Patch(facecolor="#DDDDDD", label="Connected"),
            mpatches.Patch(facecolor="#0077BB", label="Priority Charging"),
            mpatches.Patch(facecolor="#33BBEE", label="Non-Priority Charging"),
            Line2D([0], [0], marker="v", color="w", markerfacecolor="#CC3311", markersize=5, label="Preempt / Delay"),
        ]
        ax.legend(handles=legend_el, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f"fig_preemption_gantt_{safe_tag}_90EVs.png"), dpi=300)
        plt.close()


def plot_fig8_cost_breakdown(df_e1: pd.DataFrame):
    """Fig. 8: Stacked Cost Breakdown by TOU Tier (Off-Peak / Shoulder / Peak)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=False)
    methods = ["AQPS-RT-ET", "AQPS-static", "MPC-only", "EDF", "LLF"]

    for ax, fleet, evse in [(ax1, 45, 16), (ax2, 90, 35)]:
        sub = df_e1[(df_e1["fleet_size"] == fleet) & (df_e1["scenario_key"] == "S3_HighPriority")]
        sub = sub.set_index("method").reindex(methods).reset_index()
        x = np.arange(len(methods))
        off_p = sub["cost_off_peak"].values
        sh_p = sub["cost_shoulder"].values
        pk_p = sub["cost_peak"].values

        ax.bar(x, off_p, 0.5, label="Off-Peak ($0.18)", color="#009988")
        ax.bar(x, sh_p, 0.5, bottom=off_p, label="Shoulder ($0.21)", color="#33BBEE")
        ax.bar(x, pk_p, 0.5, bottom=off_p + sh_p, label="Peak ($0.32)", color="#CC3311")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=20, ha="right")
        ax.set_ylabel("Daily Grid Cost ($) — S2 Contention")
        ax.set_title(f"{fleet} EVs ({evse} EVSEs)", fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, alpha=0.2)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.06))
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(os.path.join(FIG_DIR, "fig_cost_breakdown_tiers.png"), dpi=300)
    plt.close()


def plot_fig9_queue_position_injection():
    """Fig. 9: Queue Position & Priority Promotion Trajectory with Operator Injection Marker."""
    raw_p = os.path.join("results", "E2_case_studies", "raw", "case_1_breakdown_AQPS-RT-ET.json")
    if not os.path.exists(raw_p):
        return
    with open(raw_p) as f:
        res = json.load(f)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.0, 3.6), sharex=True)
    hours = np.arange(96) * 0.25 + 6.0
    inj_hour = 6.0 + res.get("injection_time_idx", 24) * 0.25

    tracked = ["EV_005", "EV_025", "EV_012", "EV_040"]
    colors = ["#CC3311", "#0077BB", "#009988", "#EE7733"]

    for vid, c in zip(tracked, colors):
        qpos = res["queue_pos_timeline"].get(vid, [None] * 96)
        q_clean = [np.nan if v is None else float(v) for v in qpos]
        pwr = res["ev_power_timeline"].get(vid, [0.0] * 96)
        ax1.plot(hours, q_clean, marker="o", markersize=2, color=c, label=vid)
        ax2.step(hours, pwr, where="post", color=c, label=vid)

    for ax in (ax1, ax2):
        ax.axvline(inj_hour, color="#CC3311", ls="--", lw=1.1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.2)

    ax1.invert_yaxis()
    ax1.set_ylabel("Queue Priority Rank (1=Top)")
    ax1.set_title("Real-Time Layer-0 Re-Ranking & Power Allocation Under Breakdown Injection (12:00)", fontweight="bold")
    ax1.text(inj_hour + 0.15, 3.0, "Breakdown Injection\n(EV_005 Promoted)", color="#CC3311", fontsize=7, fontweight="bold")
    ax2.set_ylabel("Charging Power (kW)")
    ax2.set_xlabel("Time of Day (Hours from 06:00)")
    ax1.legend(loc="lower center", bbox_to_anchor=(0.5, -1.42), ncol=4, frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig_queue_trajectory_injection.png"), dpi=300)
    plt.close()


def plot_fig10_monte_carlo(df_mc: pd.DataFrame):
    """Fig. 10: 30-Run Random Disruption Monte Carlo (Mean +/- 95% CI)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.7))
    methods = ["AQPS-RT-ET", "AQPS-static", "AQPS-static", "MPC-only", "EDF", "LLF"]
    x = np.arange(len(methods))

    for ax, metric, ylabel, title in [
        (ax1, "priority_fulfillment_pct", "Priority Fulfillment (%)", "(a) Priority Service Robustness (30 Runs)"),
        (ax2, "total_cost", "Total Daily Cost ($)", "(b) Daily Charging Cost (30 Runs)"),
    ]:
        sub = df_mc[df_mc["metric"] == metric].set_index("method").reindex(methods).reset_index()
        colors = [METHOD_COLORS[m] for m in methods]
        ax.bar(x, sub["mean"], yerr=sub["ci95"], capsize=3, width=0.55, color=colors, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=22, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig_monte_carlo_disruptions.png"), dpi=300)
    plt.close()


def plot_fig11_aemo_pv(df_pv: pd.DataFrame):
    """Fig. 11: Real AEMO VIC1 PV Forecast vs Actual Profiles & Daily Cost vs Error."""
    with open(os.path.join("data", "aemo_vic1_pv_profiles.json")) as f:
        aemo = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.7))
    hours = np.arange(96) * 0.25 + 6.0
    for idx, color in [(0, "#009988"), (2, "#0077BB"), (5, "#CC3311")]:
        d = aemo["days"][idx]
        ax1.plot(hours, d["pv_forecast_kw"], ls="--", color=color, alpha=0.75)
        ax1.plot(hours, d["pv_actual_kw"], ls="-", color=color, label=f"{d['date']} ({d['error_tier']})")

    ax1.set_xlim(6, 20)
    ax1.set_xlabel("Time of Day (Hours)")
    ax1.set_ylabel("PV Power (kW) [Solid: Actual, Dashed: Forecast]")
    ax1.set_title("(a) AEMO VIC1 20 kW Rooftop PV Profiles", fontweight="bold")
    ax1.legend(fontsize=6, frameon=False, loc="upper right")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    sub90 = df_pv[df_pv["fleet_size"] == 90]
    for m in ["AQPS-RT-ET", "AQPS-static", "MPC-only", "LLF"]:
        m_df = sub90[sub90["method"] == m].sort_values("pv_nmae_pct")
        ax2.plot(m_df["pv_nmae_pct"], m_df["total_cost"], marker="o", color=METHOD_COLORS[m], label=m)

    ax2.set_xlabel("Daily PV Forecast Error nMAE (% of Capacity)")
    ax2.set_ylabel("Realized Daily Grid Cost ($) — 90 EVs")
    ax2.set_title("(b) Cost Sensitivity to AEMO Forecast Error", fontweight="bold")
    ax2.legend(fontsize=6.5, frameon=False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.yaxis.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig_aemo_pv_uncertainty.png"), dpi=300)
    plt.close()


def plot_fig12_event_triggered(df_et: pd.DataFrame):
    """Fig. 12: Event-Triggered vs Every-Step MPC Solve Counts & Priority Fulfillment."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.6))
    x = np.arange(len(df_et))
    labels = [
        f"$T_{{max}}={int(m)}$m\n({'Every-Step' if int(s)==1 else 'Event-Trig'})"
        for s, m in zip(df_et["t_max_steps"], df_et["t_max_minutes"])
    ]

    ax1.bar(x, df_et["l2_solve_count"], width=0.5, color="#0077BB", label="Layer-2 QP Solves")
    for i, v in enumerate(df_et["l2_solve_count"]):
        ax1.text(i, v + 1.5, str(int(v)), ha="center", va="bottom", fontsize=7, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=6.5)
    ax1.set_ylim(0, 110)
    ax1.set_ylabel("Daily Layer-2 QP Solves")
    ax1.set_title("(a) QP Solve Count Reduction", fontweight="bold")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2.plot(x, df_et["priority_fulfillment_pct"], marker="s", color="#009988", label="Priority Fulfillment (%)")
    ax2_r = ax2.twinx()
    ax2_r.plot(x, df_et["total_cost"], marker="^", ls="--", color="#EE7733", label="Total Cost ($)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=6.5)
    ax2.set_ylim(85, 103)
    ax2.set_ylabel("Priority Fulfillment (%)", color="#009988")
    ax2_r.set_ylabel("Total Cost ($)", color="#EE7733")
    ax2.set_title("(b) Schedule Optimality Preservation", fontweight="bold")
    ax2.spines["top"].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig_event_triggered_solves.png"), dpi=300)
    plt.close()


def plot_fig13_sensitivity(df_sens: pd.DataFrame):
    """Fig. 13: 4-Panel Sensitivity Analysis (alpha2, alpha3, ATC k, reserve r)."""
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.2))
    specs = [
        ("alpha2", r"$\alpha_2$ (Non-Completion Weight)", axes[0, 0]),
        ("alpha3", r"$\alpha_3$ (Tikhonov Conditioning Weight)", axes[0, 1]),
        ("k_atc", r"ATC Lookahead Parameter $k$", axes[1, 0]),
        ("reserve_kwh", r"Reserve Margin $r$ (kWh)", axes[1, 1]),
    ]
    for pname, xlabel, ax in specs:
        sub = df_sens[df_sens["param_name"] == pname].sort_values("param_val")
        x = np.arange(len(sub))
        ax.plot(x, sub["priority_fulfillment_pct"], marker="o", color="#0077BB", label="Priority Fulfill (%)")
        ax.plot(x, sub["non_priority_fulfillment_pct"], marker="s", ls="--", color="#33BBEE", label="Non-Priority Fulfill (%)")
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in sub["param_val"]])
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Fulfillment (%)")
        ax.set_ylim(35, 105)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, alpha=0.2)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.03))
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(os.path.join(FIG_DIR, "fig_sensitivity_panels.png"), dpi=300)
    plt.close()


def export_latex_tables(df_e1: pd.DataFrame, df_scale: pd.DataFrame):
    """Exports EPSR-ready LaTeX tables for Section 4."""
    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\caption{Master benchmark comparison across scenarios S1--S3 and fleet sizes (45 EVs / 16 EVSEs and 90 EVs / 35 EVSEs).}",
        r"\label{tab:master_results}",
        r"\footnotesize",
        r"\begin{tabular}{llcccccccc}",
        r"\toprule",
        r"Fleet & Scenario & Method & Pri. Fulfill (\%) & Mission Suff. (\%) & Non-Pri. (\%) & Cost (\$) & Gap to Opt. (\%) & QP Solves & Mean Solve (ms) \\",
        r"\midrule",
    ]
    for _, r in df_e1.iterrows():
        lines.append(
            f"{int(r['fleet_size'])} EVs & {r['scenario_label']} & {r['method']} & "
            f"{r['priority_fulfillment_pct']:.1f} & {r['mission_sufficiency_pct']:.1f} & "
            f"{r['non_priority_fulfillment_pct']:.1f} & {r['total_cost']:.1f} & "
            f"{r['gap_to_offline_opt_pct']:+.1f} & {int(r['l2_solve_count'])} & {r['solve_time_mean_ms']:.2f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    with open(os.path.join(TAB_DIR, "tab_master_e1.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    lines_s = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Scalability across fleet sizes (45, 90, 180 EVs) and EV-to-EVSE congestion ratios (2:1, 3:1, 4:1) under S2 Contention.}",
        r"\label{tab:scalability}",
        r"\footnotesize",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"Fleet & Ratio (EVSEs) & Method & Pri. (\%) & Non-Pri. (\%) & Cost (\$) & Mean Solve (ms) \\",
        r"\midrule",
    ]
    for _, r in df_scale[df_scale["method"].isin(["AQPS-RT-ET", "EDF", "LLF"])].iterrows():
        lines_s.append(
            f"{int(r['fleet_size'])} & {r['ev_to_evse_ratio']} ({int(r['n_evse'])}) & {r['method']} & "
            f"{r['priority_fulfillment_pct']:.1f} & {r['non_priority_fulfillment_pct']:.1f} & "
            f"{r['total_cost']:.1f} & {r['solve_time_mean_ms']:.2f} \\\\"
        )
    lines_s += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(TAB_DIR, "tab_scalability_e5b.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines_s))


def main():
    print("Generating publication figures and LaTeX tables...")
    plot_fig2_architecture()

    df_e1 = pd.read_csv(os.path.join("results", "E1_main", "summary", "e1_master_table.csv"))
    plot_fig3_fulfillment_grouped(df_e1)
    plot_figs4_5_cumulative_cost()
    plot_figs6_7_preemption_gantt()
    plot_fig8_cost_breakdown(df_e1)
    plot_fig9_queue_position_injection()

    df_mc = pd.read_csv(os.path.join("results", "E3_monte_carlo", "summary", "e3_mc_summary.csv"))
    plot_fig10_monte_carlo(df_mc)

    df_pv = pd.read_csv(os.path.join("results", "E4_aemo_pv", "summary", "e4_aemo_pv_results.csv"))
    plot_fig11_aemo_pv(df_pv)

    df_et = pd.read_csv(os.path.join("results", "E5_scalability", "summary", "e5a_event_trigger_comparison.csv"))
    plot_fig12_event_triggered(df_et)

    df_sens = pd.read_csv(os.path.join("results", "E6_sensitivity", "summary", "e6_sensitivity.csv"))
    plot_fig13_sensitivity(df_sens)

    df_scale = pd.read_csv(os.path.join("results", "E5_scalability", "summary", "e5b_scalability_ratios.csv"))
    export_latex_tables(df_e1, df_scale)
    print(f"All figures saved to {FIG_DIR}/ and LaTeX tables saved to {TAB_DIR}/")


if __name__ == "__main__":
    main()