"""
Paper figures 3-8 (Figs 1-2 are diagrams drawn in the manuscript).

  fig3_service       E1: mission sufficiency and non-priority fulfilment, S1-S3 x {45, 90} EVs
  fig4_cost_tiers    E1: grid energy cost by tariff tier (stacked), $/kWh printed above each bar
  fig5_gantt         E1 raw traces: 90-EV S3, AQPS-RT-ET vs LLF, 6 vehicles (selection rule below)
  fig6_swimlane      E2 raw trace: case 1 (vehicle breakdown), AQPS-RT-ET, EVSE state per vehicle
  fig7_aemo          AEMO PV forecast vs actual (3 of 6 days) and E4 $/kWh vs daily nMAE
  fig8_sensitivity   E6: alpha3 and reserve sweeps (AQPS-RT-ET)

Run from the repo root:  python -m postprocess.make_figures
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from . import common as C

KW_CMAP = LinearSegmentedColormap.from_list(
    "kw", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


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
    metrics = [("mission_sufficiency_pct", "Mission sufficiency (%)"),
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
            for xi, (tot, u) in enumerate(zip(bottom, sub["unit_cost_per_kwh"])):
                ax.text(xi, tot + 0.02 * top, f"{u:.3f}", ha="center", va="bottom", fontsize=6.5, color=C.INK2)
            ax.set_xticks(x)
            two_line = {"Offline-Optimal": "Offline-\nOpt.", "AQPS-static": "AQPS-\nstatic", "AQPS-RT-ET": "AQPS-\nRT-ET"}
            ax.set_xticklabels([two_line.get(m, C.LABEL[m]) for m in C.FIG_METHODS] if i == 1 else [], rotation=0)
            ax.tick_params(axis="x", length=0)
            ax.set_ylim(0, top * 1.14)
            if j == 0:
                ax.set_ylabel(f"{C.SCEN_LABEL[s]}\ngrid energy cost ($)")
            if i == 0:
                ax.set_title(f"{size} EVs", pad=4)
    for ax in axes[1]:
        for lab in ax.get_xticklabels():
            lab.set_fontsize(6.5)
    handles = [Patch(facecolor=C.TIER_COLOR[t], edgecolor="white", label=C.TIER_LABEL[t]) for t in tiers]
    handles.append(Patch(facecolor="none", edgecolor="none",
                         label="Number above bar: unit cost ($/kWh delivered)"))
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0), columnspacing=1.5)
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
def _gantt_selection(aq, ll):
    """3 priority EVs with the largest LLF mission shortfall that AQPS-RT-ET met, then
    3 non-priority EVs most often preempted by AQPS-RT-ET (ties: earliest arrival)."""
    A = {e["id"]: e for e in aq["per_ev"]}
    L = {e["id"]: e for e in ll["per_ev"]}
    pri = [i for i, e in A.items() if e["roster_priority"] and e["delivered"] >= e["e_mission"] - 0.1]
    pri.sort(key=lambda i: (-(L[i]["e_mission"] - L[i]["delivered"]), i))
    npr = [i for i, e in A.items() if not e["roster_priority"] and e["preempted"] > 0]
    npr.sort(key=lambda i: (-A[i]["preempted"], A[i]["arrival_idx"], i))
    return pri[:3] + npr[:3]


def _lanes(ax, res, ids, T, norm, title):
    per = {e["id"]: e for e in res["per_ev"]}
    hrs = C.step_to_hour(np.arange(T + 1))
    evs = disconnect_events(res)
    used = set()
    for y, vid in enumerate(ids):
        e = per[vid]
        a, dep = e["arrival_idx"], e["departure_idx"]
        ax.broken_barh([(hrs[a], hrs[dep] - hrs[a])], (y - 0.34, 0.68), facecolor="#f0f0f0", edgecolor="#bdbdbd",
                       linewidth=0.5)
        p = res["trace"]["power"].get(vid, [0.0] * T)
        for t in range(a, dep):
            if p[t] > 1e-3:
                ax.broken_barh([(hrs[t], hrs[t + 1] - hrs[t])], (y - 0.34, 0.68), facecolor=KW_CMAP(norm(p[t])),
                               edgecolor="none")
        plot_events(ax, hrs, y, evs.get(vid, []))
        used |= {k for _, k in evs.get(vid, [])}
        miss = e["e_mission"] - e["delivered"]
        if e["roster_priority"]:
            txt = "mission met" if miss <= 0.1 else f"short {miss:.1f} kWh"
            col = C.INK if miss <= 0.1 else C.COLOR["LLF"]
        else:
            txt = f"{100 * min(1, e['delivered'] / e['e_session']):.0f} % of session"
            col = C.INK2
        ax.text(hrs[dep] + 0.25, y, txt, va="center", ha="left", fontsize=6.5, color=col)
    ax.set_yticks(range(len(ids)))
    ax.set_yticklabels([f"{v.replace('_', ' ')} ({'P' if per[v]['roster_priority'] else 'NP'})" for v in ids])
    ax.invert_yaxis()
    ax.grid(False)
    ax.set_title(title, loc="left", pad=3)
    return used


def fig5_gantt():
    aq = C.raw("E1", "AQPS-RT-ET_S3_PeakStress_90EV")
    ll = C.raw("E1", "LLF_S3_PeakStress_90EV")
    ids = _gantt_selection(aq, ll)
    T = len(aq["trace"]["gross"])
    pmax = max(max(r["trace"]["power"].get(v, [0])) for r in (aq, ll) for v in ids)
    norm = plt.Normalize(0, pmax)
    per = {e["id"]: e for e in aq["per_ev"]}
    h0 = np.floor(C.step_to_hour(min(per[v]["arrival_idx"] for v in ids)))
    h1 = np.ceil(C.step_to_hour(max(per[v]["departure_idx"] for v in ids))) + 1.6
    fig, axes = plt.subplots(2, 1, figsize=(C.W2, 3.6), sharex=True)
    used = set()
    for ax, res, ttl in [(axes[0], aq, "(a) AQPS-RT-ET"), (axes[1], ll, "(b) LLF")]:
        ax.axvspan(15, 21, color="#f4f4f4", zorder=0, linewidth=0)
        used |= _lanes(ax, res, ids, T, norm, ttl)
        ax.set_xlim(h0, h1)
    axes[0].text(18, -0.62, "peak tariff 15:00\u201321:00", ha="center", va="bottom", fontsize=6.5, color=C.INK2)
    C.hour_ticks(axes[1], h0=h0, h1=np.floor(h1), every=2)
    axes[1].set_xlim(h0, h1)
    axes[1].set_xlabel("Time (AEST)")
    handles = [Patch(facecolor="#f0f0f0", edgecolor="#bdbdbd", linewidth=0.5, label="At depot, not charging")]
    handles += event_handles(used)
    fig.legend(handles=handles, loc="upper left", ncol=len(handles), bbox_to_anchor=(0.11, 1.0),
               columnspacing=1.2, handletextpad=0.4, fontsize=6.5)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=KW_CMAP)
    cax = fig.add_axes([0.83, 0.957, 0.14, 0.02])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(labelsize=6.5, length=2, pad=1)
    fig.text(0.825, 0.967, "Power (kW)", ha="right", va="center", fontsize=6.5, color=C.INK)
    fig.subplots_adjust(left=0.11, right=0.995, top=0.875, bottom=0.11, hspace=0.3)
    C.save(fig, "fig5_gantt")
    return ids


# ----------------------------------------------------------------------------- Fig 6
STATE_COLOR = {"wait": "#e6e6e6", "np": "#86b6ef", "p": "#1c5cab"}
STATE_LABEL = {"wait": "Waiting (no EVSE)", "np": "On EVSE, non-priority", "p": "On EVSE, priority"}


def fig6_swimlane():
    case = "case_1_breakdown"
    r = C.raw("E2", f"AQPS-RT-ET_{case}")
    inj = __import__("json").load(open(C.os.path.join(C.ROOT, "data", "operator_injections.json"), encoding="utf-8"))[case]
    t_inj, promoted = inj["time_idx"], inj["promote_id"]
    tr, per = r["trace"], {e["id"]: e for e in r["per_ev"]}
    T = len(tr["gross"])
    conn = [set(c) for c in tr["connected"]]
    pri = [set(p) for p in tr["priority"]]
    # selection: promoted EV, then EVs disconnected while present within [t_inj, t_inj + 16), then
    # priority EVs connected at t_inj (earliest departure first); 6 lanes in total
    events = disconnect_events(r)
    sel = [promoted]
    near = sorted((v for v, ts in events.items() if v not in sel and any(t_inj <= t < t_inj + 16 for t, _ in ts)),
                  key=lambda v: (min(t for t, _ in events[v] if t >= t_inj), v))
    sel += near[:5 - len(sel) + 1]
    if len(sel) < 6:
        extra = sorted((v for v in per if v not in sel and v in conn[t_inj] and v in pri[t_inj]),
                       key=lambda v: (per[v]["departure_idx"], v))
        sel += extra[:6 - len(sel)]
    hrs = C.step_to_hour(np.arange(T + 1))
    h0 = 9.0
    h1 = float(np.ceil(C.step_to_hour(max(per[v]["departure_idx"] for v in sel)))) + 0.5
    h1 = min(h1, 21.0)
    fig, ax = plt.subplots(figsize=(C.W2, 2.6))
    for y, v in enumerate(sel):
        e = per[v]
        cum = np.cumsum(np.asarray(tr["power"].get(v, [0.0] * T)) * 0.25)
        for t in range(e["arrival_idx"], e["departure_idx"]):
            s = ("p" if v in pri[t] else "np") if v in conn[t] else "wait"
            ax.broken_barh([(hrs[t], hrs[t + 1] - hrs[t])], (y - 0.32, 0.64), facecolor=STATE_COLOR[s],
                           edgecolor="white", linewidth=0.3)
        plot_events(ax, hrs, y, events.get(v, []))
        if e["roster_priority"]:
            hit = np.nonzero(cum >= e["e_mission"] - 0.1)[0]
            if hit.size:
                ax.plot(hrs[hit[0] + 1], y - 0.42, marker="v", markersize=5, color="#eda100", markeredgecolor=C.INK,
                        markeredgewidth=0.5, zorder=6)
        ax.plot([hrs[e["departure_idx"]]] * 2, [y - 0.38, y + 0.38], color=C.INK, linewidth=0.9)
    ax.axvline(hrs[t_inj], color=C.COLOR["LLF"], linewidth=1.0, linestyle=(0, (3, 2)))
    ax.text(hrs[t_inj] + 0.08, -0.62, f"breakdown: {promoted.replace('_', ' ')} promoted", fontsize=6.5,
            color=C.COLOR["LLF"], va="bottom")
    ax.set_yticks(range(len(sel)))
    ax.set_yticklabels([f"{v.replace('_', ' ')}{'*' if v == promoted else ''}" for v in sel])
    ax.invert_yaxis()
    ax.set_ylim(len(sel) - 0.5, -0.85)
    ax.grid(False)
    C.hour_ticks(ax, h0=h0, h1=h1, every=1)
    ax.set_xlim(h0, h1)
    ax.set_xlabel("Time (AEST)")
    handles = [Patch(facecolor=STATE_COLOR[k], edgecolor="none", label=STATE_LABEL[k]) for k in ("wait", "np", "p")]
    handles += event_handles({k for v in sel for _, k in events.get(v, [])})
    handles += [Line2D([], [], marker="v", linestyle="none", color="#eda100", markeredgecolor=C.INK,
                       markeredgewidth=0.5, markersize=5, label="Mission energy reached"),
                Line2D([], [], marker="|", linestyle="none", color=C.INK, markersize=7, label="Departure")]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0), columnspacing=1.2,
               handletextpad=0.4, fontsize=6.5)
    fig.subplots_adjust(left=0.085, right=0.995, top=0.80, bottom=0.15)
    C.save(fig, "fig6_swimlane")
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
    e6 = C.runs("E6")
    a3 = e6[e6.param == "alpha3"].sort_values("value")
    rs = e6[e6.param == "reserve_frac"].sort_values("value")
    base = C.cfg()["optimizer"]["alpha3"]
    col = C.COLOR["AQPS-RT-ET"]
    fig, axes = plt.subplots(1, 4, figsize=(C.W2, 1.95))
    panels = [
        (axes[0], a3.value, a3.unit_cost_per_kwh, r"$\alpha_3$", "Unit cost ($/kWh)", "(a)", True, base),
        (axes[1], a3.value, a3.nonpriority_fulfil_pct, r"$\alpha_3$", "Non-priority fulfilment (%)", "(b)", True, base),
        (axes[2], 100 * rs.value, rs.mission_sufficiency_pct, "Reserve $r$ (% of $C_i$)", "Mission sufficiency (%)",
         "(c)", False, 10),
        (axes[3], 100 * rs.value, rs.unit_cost_per_kwh, "Reserve $r$ (% of $C_i$)", "Unit cost ($/kWh)", "(d)",
         False, 10),
    ]
    for ax, x, y, xl, yl, tag, logx, b in panels:
        ax.plot(x, y, color=col, marker="o", markersize=3.8, markeredgecolor="white", markeredgewidth=0.4)
        ax.axvline(b, color=C.INK2, linewidth=0.6, linestyle=(0, (2, 2)), zorder=0)
        if logx:
            ax.set_xscale("log")
            ax.minorticks_off()
            ax.set_xticks(list(x))
            ax.set_xticklabels([f"$10^{{{int(round(np.log10(v)))}}}$" for v in x])
        else:
            ax.set_xticks(list(x))
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(tag, loc="left", pad=2)
        rng = float(np.ptp(y))
        mid = float(np.mean([np.min(y), np.max(y)]))
        half = max(rng * 0.65, 0.004 * abs(mid) if "cost" in yl.lower() else 1.0)
        ax.set_ylim(mid - half, mid + half)
    axes[2].set_ylim(min(axes[2].get_ylim()[0], 90), 101)
    fig.subplots_adjust(left=0.085, right=0.99, top=0.9, bottom=0.25, wspace=0.75)
    C.save(fig, "fig8_sensitivity")


def main():
    C.style()
    fig3_service()
    fig4_cost_tiers()
    g = fig5_gantt()
    s = fig6_swimlane()
    fig7_aemo()
    fig8_sensitivity()
    print("  fig5 vehicles:", g)
    print("  fig6 vehicles:", s)


if __name__ == "__main__":
    main()
