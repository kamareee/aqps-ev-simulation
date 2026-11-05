"""
Paper figures 3-8 (Figs 1-2 are diagrams drawn in the manuscript).

  fig3_service       E1: mission sufficiency and non-priority fulfilment, S1-S3 x {45, 90} EVs
  fig4a_cumulative_cost  E1 raw traces: cumulative grid energy cost, S2 and S3 x {45, 90} EVs
  fig4_cost_tiers    E1: grid energy cost by tariff tier (stacked)
  fig5_swimlane_s3   E1 raw traces: 90-EV S3, EVSE state of 3 vehicles, AQPS-RT-ET / LLF / EDF
  fig7_aemo          AEMO PV forecast vs actual (3 of 6 days) and E4 $/kWh vs daily nMAE
  fig8_sensitivity   E6: all six one-at-a-time sweeps (AQPS-RT-ET), 3 metrics x 6 parameters

Run from the repo root:  python -m postprocess.make_figures
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from . import common as C

def bar(ax, x, h, m, w, bottom=None, **kw):
    return ax.bar(x, h, w, bottom=bottom, color=C.COLOR[m], hatch=C.HATCH.get(m),
                  edgecolor="white", linewidth=0.6, **kw)


def method_legend(fig, methods, y=1.0, ncol=None):
    handles = [Patch(facecolor=C.COLOR[m], hatch=C.HATCH.get(m), edgecolor="white", linewidth=0.6,
                     label=C.LABEL[m]) for m in methods]
    fig.legend(handles=handles, loc="upper center", ncol=ncol or len(methods), bbox_to_anchor=(0.5, y),
               handlelength=1.4, handleheight=0.9, columnspacing=1.2)


# ----------------------------------------------------------------------------- Fig 3
def fig3_service():
    d = C.runs("E1")
    metrics = [("mission_sufficiency_pct", "Priority vehicle\ndemand met (%)"),
               ("nonpriority_fulfil_pct", "Non-priority energy\nfulfilment (%)")]
    fl = C.cfg()["fleets"]
    fig, axes = plt.subplots(2, 2, figsize=(C.W2, 3.45), sharey=True)
    n = len(C.FIG_METHODS)
    w = 0.8 / n
    for j, size in enumerate([45, 90]):
        for i, (col, ylab) in enumerate(metrics):
            ax = axes[i, j]
            for k, m in enumerate(C.FIG_METHODS):
                vals = [d[(d.fleet_size == size) & (d.scenario == s) & (d.method == m)][col].iloc[0] for s in C.SCEN]
                bar(ax, np.arange(3) + (k - (n - 1) / 2) * w, vals, m, w)
            ax.set_ylim(0, 105)
            ax.set_yticks([0, 25, 50, 75, 100])
            ax.set_xticks(range(3))
            ax.set_xticklabels([C.SCEN_LABEL[s] for s in C.SCEN] if i == 1 else [])
            ax.tick_params(axis="x", length=0)
            if j == 0:
                ax.set_ylabel(ylab)
            if i == 0:
                f = fl[size] if size in fl else fl[str(size)]
                ax.set_title(f"{size} EVs, {f['n_evse']} EVSEs, {int(f['site_kw'])} kW site limit", pad=4)
    method_legend(fig, C.FIG_METHODS)
    fig.subplots_adjust(left=0.085, right=0.995, top=0.875, bottom=0.075, hspace=0.18, wspace=0.06)
    C.save(fig, "fig3_service")


# ----------------------------------------------------------------------------- Fig 4a
def tariff_per_step(T):
    """ToU price ($/kWh) of each 15-min step from config (windows in local hours, wrap at 24)."""
    tar = C.cfg()["tariff"]
    hrs = C.step_to_hour(np.arange(T)) % 24
    price = np.zeros(T)
    for tier in tar.values():
        for a, b in tier["windows"]:
            price[(hrs >= a) & (hrs < b)] = tier["price"]
    return price


CUM_LS = {"EDF": (0, (5, 1.5, 1, 1.5)), "Offline-Optimal": (0, (3, 1.5))}


def fig4a_cumulative_cost():
    """Cumulative grid energy cost: sum over steps of net import (kW) x price x 0.25 h. The end value
    equals cost_total in E1/summary/runs.csv (checked below)."""
    d = C.runs("E1")
    fig, axes = plt.subplots(2, 2, figsize=(C.W2, 3.45), sharex=True)
    for i, s in enumerate(["S2_Contention", "S3_PeakStress"]):
        for j, size in enumerate([45, 90]):
            ax = axes[i, j]
            ax.axvspan(15, 21, color="#f4f4f4", zorder=0, linewidth=0)
            for m in C.FIG_METHODS:
                r = C.raw("E1", f"{m}_{s}_{size}EV")
                net = np.maximum(0.0, np.asarray(r["trace"]["net"], dtype=float))
                T = net.size
                cum = np.concatenate([[0.0], np.cumsum(net * tariff_per_step(T) * 0.25)])
                ref = d[(d.fleet_size == size) & (d.scenario == s) & (d.method == m)].cost_total.iloc[0]
                assert abs(cum[-1] - ref) < 0.01, (m, s, size, cum[-1], ref)
                ax.plot(C.step_to_hour(np.arange(T + 1)), cum, color=C.COLOR[m], linewidth=1.1,
                        linestyle=CUM_LS.get(m, "-"), zorder=3 if m == "AQPS-RT-ET" else 2)
            C.hour_ticks(ax, 6, 36, every=6)
            ax.set_xlim(6, 36)
            if i == 0:
                ax.set_title(f"{size} EVs", pad=4)
                ax.text(18, 0.03, "peak\n15:00\u201321:00", ha="center", va="bottom", fontsize=6.5,
                        color=C.INK2, transform=ax.get_xaxis_transform())
            if j == 0:
                ax.set_ylabel(f"{C.SCEN_LABEL[s]}\ncumulative cost ($)")
            if i == 1:
                ax.set_xlabel("Time (AEST)")
    handles = [Line2D([], [], color=C.COLOR[m], linewidth=1.1, linestyle=CUM_LS.get(m, "-"), label=C.LABEL[m])
               for m in C.FIG_METHODS]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), bbox_to_anchor=(0.5, 1.0),
               handlelength=1.8, columnspacing=1.2)
    fig.subplots_adjust(left=0.085, right=0.99, top=0.885, bottom=0.11, hspace=0.16, wspace=0.12)
    C.save(fig, "fig4a_cumulative_cost")


# ----------------------------------------------------------------------------- Fig 4
def fig4_cost_tiers():
    d = C.runs("E1")
    tiers = ["off_peak", "shoulder", "peak"]
    fig, axes = plt.subplots(2, 2, figsize=(C.W2, 3.45))
    x = np.arange(len(C.FIG_METHODS))
    for i, s in enumerate(["S2_Contention", "S3_PeakStress"]):
        for j, size in enumerate([45, 90]):
            ax = axes[i, j]
            sub = d[(d.fleet_size == size) & (d.scenario == s)].set_index("method").loc[C.FIG_METHODS]
            bottom = np.zeros(len(x))
            for tr in tiers:
                v = sub[f"cost_{tr}"].to_numpy()
                ax.bar(x, v, 0.62, bottom=bottom, color=C.TIER_COLOR[tr], edgecolor="white", linewidth=0.8)
                bottom += v
            top = bottom.max()
            ax.set_xticks(x)
            two_line = {"Offline-Optimal": "Offline-\nOpt.", "AQPS-static": "AQPS-\nstatic", "AQPS-RT-ET": "AQPS-\nRT-ET"}
            ax.set_xticklabels([two_line.get(m, C.LABEL[m]) for m in C.FIG_METHODS] if i == 1 else [], rotation=0)
            ax.tick_params(axis="x", length=0)
            ax.set_ylim(0, top * 1.06)
            if j == 0:
                ax.set_ylabel(f"{C.SCEN_LABEL[s]}\ngrid energy cost ($)")
            if i == 0:
                ax.set_title(f"{size} EVs", pad=4)
    for ax in axes[1]:
        for lab in ax.get_xticklabels():
            lab.set_fontsize(6.5)
    handles = [Patch(facecolor=C.TIER_COLOR[t], edgecolor="white", label=C.TIER_LABEL[t]) for t in tiers]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0), columnspacing=1.5)
    fig.subplots_adjust(left=0.085, right=0.995, top=0.885, bottom=0.095, hspace=0.16, wspace=0.12)
    C.save(fig, "fig4_cost_tiers")


# ----------------------------------------------------------------------------- queue events
EVENT_MARK = {1: ("X", "Preempted (Tier 1)"), 2: ("s", "Safe release (Tier 2)"), 3: ("P", "Rotated (Tier 3)")}


def disconnect_events(res):
    """{vehicle: [(t, tier)]} for EVSE disconnections of a vehicle that is still at the depot and has
    not finished its session. Tier from the trace: non-priority before -> Tier 1; priority before and
    safe-released (non-priority after, safe_released flag) -> Tier 2; otherwise priority -> Tier 3."""
    tr, per = res["trace"], {e["id"]: e for e in res["per_ev"]}
    T = len(tr["gross"])
    conn = [set(c) for c in tr["connected"]]
    pri = [set(p) for p in tr["priority"]]
    out = {}
    for v, e in per.items():
        cum = np.cumsum(np.asarray(tr["power"].get(v, [0.0] * T)) * 0.25)
        for t in range(max(1, e["arrival_idx"] + 1), e["departure_idx"]):
            if v in conn[t - 1] and v not in conn[t] and cum[t - 1] < e["e_session"] - 0.01:
                if v not in pri[t - 1]:
                    tier = 1
                elif e["safe_released"] and v not in pri[t]:
                    tier = 2
                else:
                    tier = 3
                out.setdefault(v, []).append((t, tier))
    return out


def event_handles(tiers):
    return [Line2D([], [], marker=EVENT_MARK[k][0], linestyle="none", color=C.INK, markeredgecolor="white",
                   markeredgewidth=0.5, markersize=5, label=EVENT_MARK[k][1]) for k in sorted(tiers)]


def plot_events(ax, hrs, y, evs):
    for t, tier in evs:
        ax.plot(hrs[t], y, marker=EVENT_MARK[tier][0], markersize=5, color=C.INK, markeredgecolor="white",
                markeredgewidth=0.5, zorder=5)


# ----------------------------------------------------------------------------- Fig 5
STATE_COLOR = {"wait": "#e6e6e6", "np": "#86b6ef", "p": "#1c5cab", "done": "#ffffff"}
STATE_EDGE = {"done": "#9a9a9a"}
STATE_LABEL = {"wait": "Waiting (no EVSE)", "np": "On EVSE, non-priority", "p": "On EVSE, priority",
               "done": "Unplugged, charging complete"}
SWIM_RUN = "S3_PeakStress_90EV"
SWIM_METHODS = ["AQPS-RT-ET", "LLF", "EDF"]


def _swim_selection(P):
    """E1 S3 90 EVs. 2 priority EVs that AQPS-RT-ET serves to their mission, with the largest shortfall
    under both LLF and EDF (rank by min of the two shortfalls; ties: id), then 1 non-priority EV most
    often preempted by AQPS-RT-ET (ties: earliest arrival, id)."""
    A, L, D = P["AQPS-RT-ET"], P["LLF"], P["EDF"]
    short = lambda R, i: max(0.0, R[i]["e_mission"] - R[i]["delivered"])
    pri = [i for i, e in A.items() if e["roster_priority"] and e["delivered"] >= e["e_mission"] - 0.1]
    pri.sort(key=lambda i: (-min(short(L, i), short(D, i)), i))
    npr = [i for i, e in A.items() if not e["roster_priority"] and e["preempted"] > 0]
    npr.sort(key=lambda i: (-A[i]["preempted"], A[i]["arrival_idx"], i))
    return pri[:2] + npr[:1]


def swim_runs():
    R = {m: C.raw("E1", f"{m}_{SWIM_RUN}") for m in SWIM_METHODS}
    P = {m: {e["id"]: e for e in R[m]["per_ev"]} for m in SWIM_METHODS}
    return R, P, _swim_selection(P)


def fig5_swimlane_s3():
    """EVSE state of 3 EVs under AQPS-RT-ET, LLF and EDF, E1 S3 90 EVs (no operator injection). On-EVSE
    state as priority uses the method's priority set for AQPS and the roster flag for LLF/EDF (no
    priority set). End label: mission outcome (P) or delivered share of the session energy (NP)."""
    R, P, sel = swim_runs()
    T = len(R["AQPS-RT-ET"]["trace"]["gross"])
    hrs = C.step_to_hour(np.arange(T + 1))
    h0 = float(np.floor(min(C.step_to_hour(P[m][v]["arrival_idx"]) for m in SWIM_METHODS for v in sel)))
    h1 = float(np.ceil(max(C.step_to_hour(P[m][v]["departure_idx"]) for m in SWIM_METHODS for v in sel))) + 2.0
    fig, axes = plt.subplots(len(SWIM_METHODS), 1, figsize=(C.W2, 3.6), sharex=True)
    used, used_st = set(), set()
    for k, (ax, m) in enumerate(zip(axes, SWIM_METHODS)):
        r, per = R[m], P[m]
        tr = r["trace"]
        conn = [set(c) for c in tr["connected"]]
        pri = [set(p) for p in tr["priority"]]
        aqps = m.startswith("AQPS")
        events = disconnect_events(r)
        ax.axvspan(15, 21, color="#f4f4f4", zorder=0, linewidth=0)
        for y, v in enumerate(sel):
            e = per[v]
            cum = np.cumsum(np.asarray(tr["power"].get(v, [0.0] * T)) * 0.25)
            for t in range(e["arrival_idx"], e["departure_idx"]):
                if v not in conn[t]:
                    st = "done" if t > 0 and cum[t - 1] >= e["e_session"] - 0.01 else "wait"
                elif aqps:
                    st = "p" if v in pri[t] else "np"
                else:
                    st = "p" if e["roster_priority"] else "np"
                ax.broken_barh([(hrs[t], hrs[t + 1] - hrs[t])], (y - 0.32, 0.64), facecolor=STATE_COLOR[st],
                               edgecolor=STATE_EDGE.get(st, "white"), linewidth=0.3, zorder=2)
                used_st.add(st)
            plot_events(ax, hrs, y, events.get(v, []))
            used |= {kk for _, kk in events.get(v, [])}
            ax.plot([hrs[e["departure_idx"]]] * 2, [y - 0.38, y + 0.38], color=C.INK, linewidth=0.9, zorder=3)
            if e["roster_priority"]:
                hit = np.nonzero(cum >= e["e_mission"] - 0.1)[0]
                if hit.size:
                    ax.plot(hrs[hit[0] + 1], y - 0.42, marker="v", markersize=5, color="#eda100",
                            markeredgecolor=C.INK, markeredgewidth=0.5, zorder=6)
                miss = e["e_mission"] - e["delivered"]
                txt, col = ("demand met", C.INK) if miss <= 0.1 else (f"short {miss:.1f} kWh", C.COLOR["LLF"])
            else:
                txt, col = f"{100 * min(1, e['delivered'] / e['e_session']):.0f} % of demand", C.INK2
            ax.text(hrs[e["departure_idx"]] + 0.12, y, txt, va="center", ha="left", fontsize=6.5, color=col)
        ax.set_yticks(range(len(sel)))
        ax.set_yticklabels([f"{v.replace('_', ' ')} ({'P' if per[v]['roster_priority'] else 'NP'})" for v in sel])
        ax.set_ylim(len(sel) - 0.5, -0.62)
        ax.grid(False)
        ax.set_title(f"({'abc'[k]}) {C.LABEL[m]}", loc="left", pad=2)
        ax.tick_params(axis="y", length=0)
    axes[0].text(18, -0.62, "peak tariff 15:00\u201321:00", ha="center", va="bottom", fontsize=6.5, color=C.INK2)
    C.hour_ticks(axes[-1], h0=h0, h1=h1 - 1, every=1)
    axes[-1].set_xlim(h0, h1 - 0.5)
    axes[-1].set_xlabel("Time (AEST)")
    handles = [Patch(facecolor=STATE_COLOR[k], edgecolor=STATE_EDGE.get(k, "none"), linewidth=0.5,
                     label=STATE_LABEL[k]) for k in ("wait", "np", "p", "done") if k in used_st]
    handles += event_handles(used)
    handles += [Line2D([], [], marker="v", linestyle="none", color="#eda100", markeredgecolor=C.INK,
                       markeredgewidth=0.5, markersize=5, label="Mission energy reached"),
                Line2D([], [], marker="|", linestyle="none", color=C.INK, markersize=7, label="Departure")]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0),
               columnspacing=1.2, handletextpad=0.4, fontsize=6.5)
    fig.subplots_adjust(left=0.11, right=0.995, top=0.87, bottom=0.11, hspace=0.5)
    C.save(fig, "fig5_swimlane_s3")
    return sel


# ----------------------------------------------------------------------------- Fig 7
def fig7_aemo():
    days = C.pv_days()["days"]
    pick = [10, 50, 97.5]
    show = [next(dd for dd in days if dd["percentile"] == p) for p in pick]
    e4 = C.runs("E4")
    fig = plt.figure(figsize=(C.W2, 3.35))
    gs = fig.add_gridspec(2, 6, height_ratios=[1, 1.05], hspace=0.62, wspace=0.9)
    T = len(show[0]["pv_actual_kw"])
    hrs = C.step_to_hour(np.arange(T))
    m = hrs <= 20
    for k, dd in enumerate(show):
        ax = fig.add_subplot(gs[0, 2 * k:2 * k + 2])
        ax.step(hrs[m], np.asarray(dd["pv_actual_kw"])[m], where="post", color=C.INK, linewidth=1.1)
        ax.step(hrs[m], np.asarray(dd["pv_forecast_kw"])[m], where="post", color=C.COLOR["AQPS-RT-ET"],
                linewidth=1.1, linestyle=(0, (3, 1.5)))
        ax.set_ylim(0, 20.5)
        ax.set_xlim(6, 20)
        C.hour_ticks(ax, 6, 20, every=4)
        ax.set_title(f"({'abc'[k]}) {dd['date']}, p{dd['percentile']:g}, nMAE {dd['nmae_pct']:.1f} %", pad=3,
                     fontsize=7.5)
        if k == 0:
            ax.set_ylabel("PV (kW, 20 kW array)")
    fig.legend(handles=[Line2D([], [], color=C.INK, label="Actual (ROOFTOP_PV_ACTUAL)"),
                        Line2D([], [], color=C.COLOR["AQPS-RT-ET"], linestyle=(0, (3, 1.5)),
                               label="Day-ahead forecast (ROOFTOP_PV_FORECAST)")],
               loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    meths = ["LLF", "EDF", "MPC-only", "AQPS-static", "AQPS-RT-ET"]
    mk = {"LLF": "o", "EDF": "s", "MPC-only": "^", "AQPS-static": "v", "AQPS-RT-ET": "D"}
    axs = []
    for j, size in enumerate([45, 90]):
        ax = fig.add_subplot(gs[1, 3 * j:3 * j + 3])
        axs.append(ax)
        for mth in meths:
            s = e4[(e4.fleet_size == size) & (e4.method == mth)].sort_values("pv_nmae_pct")
            ax.plot(s.pv_nmae_pct, s.unit_cost_per_kwh, marker=mk[mth], markersize=3.8, color=C.COLOR[mth],
                    linewidth=1.0, markeredgecolor="white", markeredgewidth=0.4, label=C.LABEL[mth])
        ax.set_xlabel("Daily forecast error, nMAE (% of 20 kW)")
        ax.set_title(f"({'de'[j]}) {size} EVs, S2", pad=3, fontsize=7.5)
        if j == 0:
            ax.set_ylabel("Unit cost ($/kWh)")
    lo = min(a.get_ylim()[0] for a in axs)
    hi = max(a.get_ylim()[1] for a in axs)
    for a in axs:
        a.set_ylim(lo, hi)
    axs[1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), handlelength=1.6)
    fig.subplots_adjust(left=0.075, right=0.86, top=0.86, bottom=0.11)
    C.save(fig, "fig7_aemo")


# ----------------------------------------------------------------------------- Fig 8
def fig8_sensitivity():
    """E6, AQPS-RT-ET, 90 EVs, S2: one-at-a-time sweeps of all six parameters (columns) and three outcome
    metrics (rows, shared y per row). The k_ATC sweep runs at 4 vehicles per charger (n_evse_override), so
    its level differs from the other columns. Dashed line = default (config)."""
    e6 = C.runs("E6_S2")
    c = C.cfg()
    reserve_default = 10  # % of C_i, data_generator.py (reserve = 0.10 x battery capacity)
    cols = [
        ("alpha2", r"$\alpha_2$", "log", c["optimizer"]["alpha2"]),
        ("alpha3", r"$\alpha_3$", "log", c["optimizer"]["alpha3"]),
        ("pv_dev_threshold_kw", r"$\tau_{\mathrm{pv}}$ (kW)", "lin", c["event_trigger"]["pv_dev_threshold_kw"]),
        ("preempt_slack_h", r"$\delta$ (h)", "cat", c["layer1"]["preempt_slack_h"]),
        ("k_atc", r"$k_{\mathrm{ATC}}$ (4:1)", "lin", c["layer0"]["k_atc"]),
        ("reserve_frac", r"$r$ (% of $C_i$)", "lin", reserve_default),
    ]
    rows = [("mission_sufficiency_pct", "Priority vehicle\ndemand met (%)"),
            ("nonpriority_fulfil_pct", "Non-priority\nfulfilment (%)"),
            ("unit_cost_per_kwh", "Unit cost\n($/kWh)")]
    col = C.COLOR["AQPS-RT-ET"]
    fig, axes = plt.subplots(3, 6, figsize=(C.W2, 3.7), sharey="row")
    for j, (par, xl, kind, dflt) in enumerate(cols):
        sub = e6[e6.param == par].copy()
        if kind == "cat":
            sub["k"] = sub.value.fillna(np.inf)
            sub = sub.sort_values("k")
            x = np.arange(len(sub))
            labels = ["eager" if not np.isfinite(v) else f"{v:g}" for v in sub.k]
            xd = labels.index(f"{dflt:g}")
        else:
            sub = sub.sort_values("value")
            x = sub.value.to_numpy() * (100 if par == "reserve_frac" else 1)
            xd = dflt
        for i, (met, yl) in enumerate(rows):
            ax = axes[i, j]
            ax.plot(x, sub[met].to_numpy(), color=col, marker="o", markersize=3.2, markeredgecolor="white",
                    markeredgewidth=0.4, linewidth=1.0)
            ax.axvline(xd, color=C.INK2, linewidth=0.6, linestyle=(0, (2, 2)), zorder=0)
            if kind == "log":
                ax.set_xscale("log")
                ax.minorticks_off()
                xt = list(x)[::2] if len(x) > 3 else list(x)
                ax.set_xticks(xt)
                ax.set_xticklabels([f"$10^{{{int(round(np.log10(v)))}}}$" if abs(np.log10(v) - round(np.log10(v))) < 1e-6
                                    else f"{v:g}" for v in xt])
            elif kind == "cat":
                ax.set_xticks(x)
                ax.set_xticklabels(labels)
                ax.set_xlim(-0.5, len(x) - 0.5)
            else:
                ax.set_xticks(list(x) if len(x) <= 4 else list(x)[::2])
                ax.set_xticklabels([f"{v:g}" for v in (list(x) if len(x) <= 4 else list(x)[::2])])
            if i < 2:
                ax.tick_params(axis="x", labelbottom=False)
            else:
                ax.set_xlabel(xl)
            if j == 0:
                ax.set_ylabel(yl)
            ax.tick_params(axis="both", labelsize=6.5)
    axes[0, 0].set_ylim(89, 101)
    axes[1, 0].set_ylim(40, 95)
    lo = e6.unit_cost_per_kwh.min()
    hi = e6.unit_cost_per_kwh.max()
    pad = 0.08 * (hi - lo)
    axes[2, 0].set_ylim(lo - pad, hi + pad)
    fig.subplots_adjust(left=0.105, right=0.995, top=0.98, bottom=0.12, hspace=0.18, wspace=0.12)
    C.save(fig, "fig8_sensitivity")


def main():
    C.style()
    fig3_service()
    fig4a_cumulative_cost()
    fig4_cost_tiers()
    s = fig5_swimlane_s3()
    fig7_aemo()
    fig8_sensitivity()
    print("  fig5 vehicles:", s)


if __name__ == "__main__":
    main()
