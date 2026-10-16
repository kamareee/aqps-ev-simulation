"""
Numbers quoted in the manuscript text, as LaTeX macros, with a source map.

  paper/numbers.tex      \\newcommand{\\R...}{value}   (\\input in the preamble; use the macro in the text)
  paper/numbers_map.csv  macro, value, unit, source file, filter/expression

Every number in the text must be one of these macros (or a table/figure value).
Run from the repo root:  python -m postprocess.make_numbers
"""

import csv
import json
import os

import numpy as np
import pandas as pd

from . import common as C

WORD = {45: "FortyFive", 90: "Ninety", 180: "OneEighty"}
SCW = {"S1_Baseline": "SOne", "S2_Contention": "STwo", "S3_PeakStress": "SThree"}
MW = {"LLF": "LLF", "EDF": "EDF", "MPC-only": "MPC", "Offline-Optimal": "Offline", "AQPS-static": "Static",
      "AQPS-RT": "RT", "AQPS-RT-ET": "RTET"}
REG = []


def put(name, value, d, unit, src, expr):
    assert name.isalpha(), name
    txt = f"{value:,.{d}f}".replace(",", "{,}") if isinstance(value, (int, float, np.floating, np.integer)) else str(value)
    REG.append({"macro": "\\R" + name, "value": txt, "unit": unit, "source": src, "expression": expr})


def e1():
    d = C.runs("E1")
    src = "results/E1/summary/runs.csv"
    for n in (45, 90):
        for s in C.SCEN:
            for m in C.TAB_METHODS:
                r = d[(d.fleet_size == n) & (d.scenario == s) & (d.method == m)].iloc[0]
                key = f"{MW[m]}{SCW[s]}{WORD[n]}"
                flt = f"fleet_size={n}, scenario={s}, method={m}"
                put("Mission" + key, r.mission_sufficiency_pct, 1, "%", src, f"mission_sufficiency_pct [{flt}]")
                put("NonPri" + key, r.nonpriority_fulfil_pct, 1, "%", src, f"nonpriority_fulfil_pct [{flt}]")
                put("UnitCost" + key, r.unit_cost_per_kwh, 3, "$/kWh", src, f"unit_cost_per_kwh [{flt}]")
                put("NonPriTwoTwo" + key, r.nonpriority_fulfil_22h_pct, 1, "%", src,
                    f"nonpriority_fulfil_22h_pct [{flt}]")
    gaps = []
    for n in (45, 90):
        g = d[(d.fleet_size == n) & (d.scenario == "S3_PeakStress")].set_index("method")
        gaps.append(g.loc["Offline-Optimal"].nonpriority_fulfil_pct - g.loc["AQPS-RT-ET"].nonpriority_fulfil_pct)
    put("SThreeOfflineGapMin", min(gaps), 1, "pp", src,
        "min over fleets of nonpriority_fulfil_pct Offline-Optimal - AQPS-RT-ET [S3_PeakStress]")
    put("SThreeOfflineGapMax", max(gaps), 1, "pp", src,
        "max over fleets of nonpriority_fulfil_pct Offline-Optimal - AQPS-RT-ET [S3_PeakStress]")
    o = d[d.method == "Offline-Optimal"]
    put("OfflineGapMax", 100 * o.offline_mip_gap_bound.max(), 4, "%", src,
        "100*max(offline_mip_gap_bound) [method=Offline-Optimal]")
    put("OfflineSolveMaxS", o.wall_s.max(), 0, "s", src, "max(wall_s) [method=Offline-Optimal]")
    # unit-cost saving of AQPS-RT-ET vs LLF, per scenario (range over fleets)
    for s in C.SCEN:
        sv = []
        for n in (45, 90):
            a = d[(d.fleet_size == n) & (d.scenario == s) & (d.method == "AQPS-RT-ET")].unit_cost_per_kwh.iloc[0]
            b = d[(d.fleet_size == n) & (d.scenario == s) & (d.method == "LLF")].unit_cost_per_kwh.iloc[0]
            sv.append(100 * (1 - a / b))
        put(f"UnitSaveMin{SCW[s]}", min(sv), 1, "%", src, f"min over fleets of 100*(1 - $/kWh RT-ET / LLF) [{s}]")
        put(f"UnitSaveMax{SCW[s]}", max(sv), 1, "%", src, f"max over fleets of 100*(1 - $/kWh RT-ET / LLF) [{s}]")


def gantt():
    from .make_figures import _gantt_selection
    src = "results/E1/raw/{AQPS-RT-ET,LLF}_S3_PeakStress_90EV.json.gz"
    aq = C.raw("E1", "AQPS-RT-ET_S3_PeakStress_90EV")
    ll = C.raw("E1", "LLF_S3_PeakStress_90EV")
    ids = _gantt_selection(aq, ll)
    A = {e["id"]: e for e in aq["per_ev"]}
    L = {e["id"]: e for e in ll["per_ev"]}
    short = [L[i]["e_mission"] - L[i]["delivered"] for i in ids[:3]]
    put("GanttShortMin", min(short), 1, "kWh", src, "min LLF mission shortfall of the Fig. 5 priority EVs")
    put("GanttShortMax", max(short), 1, "kWh", src, "max LLF mission shortfall of the Fig. 5 priority EVs")
    for nm, R in (("Aqps", A), ("LLF", L)):
        pct = [100 * min(1.0, R[i]["delivered"] / R[i]["e_session"]) for i in ids[3:]]
        put(f"GanttNP{nm}Min", min(pct), 0, "%", src, f"min session % of the Fig. 5 non-priority EVs ({nm})")
        put(f"GanttNP{nm}Max", max(pct), 0, "%", src, f"max session % of the Fig. 5 non-priority EVs ({nm})")


def e2():
    d = C.runs("E2")
    src = "results/E2/summary/runs.csv"
    for m in ("LLF", "EDF", "AQPS-static", "AQPS-RT-ET"):
        r = d[(d.case == "case_1_breakdown") & (d.method == m)].iloc[0]
        put(f"CaseOneMission{MW[m]}", r.mission_sufficiency_pct, 1, "%", src,
            f"mission_sufficiency_pct [case=case_1_breakdown, method={m}]")
    p = pd.read_csv(os.path.join(C.RES, "E2", "summary", "previews.csv"))
    r = p[(p.case == "case_1_breakdown") & (p.option == "downgrade")].iloc[0]
    put("CaseOneDowngradeDelta", r.cost_delta, 2, "$", "results/E2/summary/previews.csv",
        "cost_delta [case=case_1_breakdown, option=downgrade]")


def e3():
    d = C.runs("E3")
    src = "results/E3/summary/runs.csv"
    put("EThreeSeeds", d.seed.nunique(), 0, "", src, "number of seeds")
    mu = d.groupby("method").unit_cost_per_kwh.mean()
    put("EThreeUnitSpreadPct", 100 * (mu.max() / mu.min() - 1), 1, "%", src,
        "100*(max/min - 1) of per-method mean unit_cost_per_kwh")
    for m in ("LLF", "EDF", "MPC-only", "AQPS-static", "AQPS-RT-ET"):
        x = d[d.method == m]
        for col, nm, dd in [("mission_sufficiency_pct", "Mission", 1), ("nonpriority_fulfil_pct", "NonPri", 1),
                            ("unit_cost_per_kwh", "UnitCost", 4)]:
            mu, h = C.ci95(x[col])
            put(f"EThree{nm}{MW[m]}", mu, dd, "", src, f"mean {col} [method={m}]")
            put(f"EThree{nm}CI{MW[m]}", h, dd, "", src, f"95% t half-width {col} [method={m}]")
        put(f"EThreeMissSeeds{MW[m]}", int((x.mission_sufficiency_pct < 100).sum()), 0, "seeds", src,
            f"count(mission_sufficiency_pct < 100) [method={m}]")


def e5():
    d = C.runs("E5")
    src = "results/E5/summary/runs.csv"
    a = d[d.part == "a_et_vs_every_step"]
    l2, tot = [], []
    for (n, s), g in a.groupby(["fleet_size", "scenario"]):
        rt, et = g[g.method == "AQPS-RT"].iloc[0], g[g.method == "AQPS-RT-ET"].iloc[0]
        l2.append(100 * (1 - et.l2_solves / rt.l2_solves))
        tot.append(100 * (1 - et.total_solves / rt.total_solves))
    put("LTwoRedMin", min(l2), 0, "%", src, "min over fleet x scenario of 100*(1 - L2 solves ET/RT) [part=a]")
    put("LTwoRedMax", max(l2), 0, "%", src, "max ... L2 solves [part=a]")
    put("TotRedMin", min(tot), 0, "%", src, "min ... total solves [part=a]")
    put("TotRedMax", max(tot), 0, "%", src, "max ... total solves [part=a]")
    dn, dc = [], []
    for (n, s), g in a.groupby(["fleet_size", "scenario"]):
        rt, et = g[g.method == "AQPS-RT"].iloc[0], g[g.method == "AQPS-RT-ET"].iloc[0]
        dn.append(abs(et.nonpriority_fulfil_pct - rt.nonpriority_fulfil_pct))
        dc.append(abs(1000 * (et.unit_cost_per_kwh - rt.unit_cost_per_kwh)))
    put("ETMaxDeltaNonPri", max(dn), 2, "pp", src, "max |RT-ET - RT| nonpriority_fulfil_pct [part=a]")
    put("ETMaxDeltaCostMWh", max(dc), 2, "$/MWh", src, "max |RT-ET - RT| unit cost x1000 [part=a]")
    b = d[d.part == "b_tmax_sweep"].sort_values("t_max_steps")
    put("TmaxLTwoFirst", int(b.l2_solves.iloc[0]), 0, "", src, "l2_solves at smallest T_max [part=b]")
    put("TmaxLTwoLast", int(b.l2_solves.iloc[-1]), 0, "", src, "l2_solves at largest T_max [part=b]")
    c = d[(d.part == "c_scalability") & (d.method == "AQPS-RT-ET")]
    big = c[c.fleet_size == 180]
    put("ScalMeanMsMin", big.l2_solve_ms_mean.min(), 0, "ms", src, "min l2_solve_ms_mean [part=c, 180 EVs, RT-ET]")
    put("ScalMeanMsMax", big.l2_solve_ms_mean.max(), 0, "ms", src, "max l2_solve_ms_mean [part=c, 180 EVs, RT-ET]")
    put("ScalPNinetyFiveMax", big.l2_solve_ms_p95.max(), 0, "ms", src, "max l2_solve_ms_p95 [part=c, 180 EVs, RT-ET]")
    g = d[(d.part == "c_scalability")]
    low = g[(g.method == "AQPS-RT-ET") & (g.ratio < 4)]
    put("ScalNonPriRTETLowMin", low.nonpriority_fulfil_pct.min(), 1, "%", src,
        "min nonpriority_fulfil_pct [part=c, ratio in {2,3}, AQPS-RT-ET]")
    put("ScalNonPriRTETLowMax", low.nonpriority_fulfil_pct.max(), 1, "%", src,
        "max nonpriority_fulfil_pct [part=c, ratio in {2,3}, AQPS-RT-ET]")
    sv = []
    for (n, ratio), h in g.groupby(["fleet_size", "ratio"]):
        h = h.set_index("method")
        sv.append(100 * (1 - h.loc["AQPS-RT-ET"].unit_cost_per_kwh / h.loc["LLF"].unit_cost_per_kwh))
    put("ScalSaveMin", min(sv), 1, "%", src, "min 100*(1 - $/kWh RT-ET / LLF) [part=c]")
    put("ScalSaveMax", max(sv), 1, "%", src, "max 100*(1 - $/kWh RT-ET / LLF) [part=c]")
    for n in (45, 90, 180):
        for m in ("LLF", "AQPS-RT-ET"):
            r = g[(g.fleet_size == n) & (g.ratio == 4) & (g.method == m)].iloc[0]
            put(f"ScalNonPri{MW[m]}{WORD[n]}RatioFour", r.nonpriority_fulfil_pct, 1, "%", src,
                f"nonpriority_fulfil_pct [part=c, fleet={n}, ratio=4, method={m}]")


def e6():
    d = C.runs("E6")
    src = "results/E6/summary/runs.csv"
    s = d[d.param == "preempt_slack_h"]
    put("DeltaEagerNonPri", s[s.value.isna()].nonpriority_fulfil_pct.iloc[0], 1, "%", src,
        "nonpriority_fulfil_pct [param=preempt_slack_h, value=null (eager)]")
    put("DeltaTwoNonPri", s[s.value == 2.0].nonpriority_fulfil_pct.iloc[0], 1, "%", src,
        "nonpriority_fulfil_pct [param=preempt_slack_h, value=2]")
    put("DeltaThreeNonPri", s[s.value == 3.0].nonpriority_fulfil_pct.iloc[0], 1, "%", src,
        "nonpriority_fulfil_pct [param=preempt_slack_h, value=3]")
    r = d[d.param == "reserve_frac"]
    put("ReserveTwentyMission", r[np.isclose(r.value, 0.2)].mission_sufficiency_pct.iloc[0], 1, "%", src,
        "mission_sufficiency_pct [param=reserve_frac, value=0.2]")


def e7():
    d = C.runs("E7")
    src = "results/E7/summary/runs.csv"
    hi = d[d.priority_share >= 0.65]
    for m in ("AQPS-static", "AQPS-RT-ET", "LLF", "EDF", "MPC-only"):
        x = hi[hi.method == m]
        put(f"OverloadMissionMin{MW[m]}", x.mission_sufficiency_pct.min(), 1, "%", src,
            f"min mission_sufficiency_pct [priority_share>=0.65, method={m}]")
        put(f"OverloadMissionMax{MW[m]}", x.mission_sufficiency_pct.max(), 1, "%", src,
            f"max mission_sufficiency_pct [priority_share>=0.65, method={m}]")
    lo = d[d.priority_share <= 0.5]
    for nm, ms in (("LLFEDF", ["LLF", "EDF"]), ("MPC", ["MPC-only"])):
        x = lo[lo.method.isin(ms)]
        put(f"LowShareMissionMin{nm}", x.mission_sufficiency_pct.min(), 0, "%", src,
            f"min mission_sufficiency_pct [priority_share<=0.5, method in {ms}]")
        put(f"LowShareMissionMax{nm}", x.mission_sufficiency_pct.max(), 0, "%", src,
            f"max mission_sufficiency_pct [priority_share<=0.5, method in {ms}]")
    x = hi[hi.method == "AQPS-static"]
    put("OverloadSoftMinStatic", x.soft_solves.min(), 0, "", src, "min soft_solves [share>=0.65, AQPS-static]")
    put("OverloadSoftMaxStatic", x.soft_solves.max(), 0, "", src, "max soft_solves [share>=0.65, AQPS-static]")


def e4():
    days = C.pv_days()
    src = "data/aemo_vic1_selected_days.json"
    nm = [dd["nmae_pct"] for dd in days["days"]]
    put("NmaeMin", min(nm), 1, "%", src, "min nmae_pct over the 6 selected days")
    put("NmaeMax", max(nm), 1, "%", src, "max nmae_pct over the 6 selected days")
    put("CandidateDays", days["metadata"]["n_candidate_weekdays"], 0, "", src, "metadata.n_candidate_weekdays")
    put("PVNormMW", days["metadata"]["normalisation_max_mw"], 0, "MW", src, "metadata.normalisation_max_mw")
    ref = next(dd for dd in days["days"] if dd["reference"])
    put("PVRefDate", ref["date"], 0, "", src, "date of the reference (50th percentile) day")
    put("PVRefNmae", ref["nmae_pct"], 1, "%", src, "nmae_pct of the reference day")
    hi = max(days["days"], key=lambda dd: dd["percentile"])
    put("PVOverForecastPct", 100 * (hi["energy_forecast_kwh_dayD"] / hi["energy_actual_kwh_dayD"] - 1), 0, "%", src,
        "100*(forecast/actual - 1) of day-D energy on the highest-percentile day")
    e4 = C.runs("E4")
    for n in (45, 90):
        a = e4[(e4.fleet_size == n) & (e4.method == "AQPS-RT-ET")].set_index("pv_date").unit_cost_per_kwh
        l = e4[(e4.fleet_size == n) & (e4.method == "LLF")].set_index("pv_date").unit_cost_per_kwh
        sv = 100 * (1 - a / l)
        put(f"EFourSaveMin{WORD[n]}", sv.min(), 1, "%", "results/E4/summary/runs.csv",
            f"min over days of 100*(1 - $/kWh RT-ET / LLF) [fleet={n}]")
        put(f"EFourSaveMax{WORD[n]}", sv.max(), 1, "%", "results/E4/summary/runs.csv",
            f"max over days of 100*(1 - $/kWh RT-ET / LLF) [fleet={n}]")


def provenance():
    m = json.load(open(os.path.join(C.RES, "E1", "manifest.json"), encoding="utf-8"))
    put("SimCommit", m["git_commit"][:7], 0, "", "results/E1/manifest.json", "git_commit (first 7)")
    put("MosekVersion", m["versions"]["mosek"], 0, "", "results/E1/manifest.json", "versions.mosek")
    put("CvxpyVersion", m["versions"]["cvxpy"], 0, "", "results/E1/manifest.json", "versions.cvxpy")
    put("CPUCores", m["machine"]["cpu_count"], 0, "", "results/E1/manifest.json", "machine.cpu_count")
    d = C.runs("E1")
    r = d[d.method == "AQPS-RT-ET"]
    put("EOneMeanMsMinRTET", r.l2_solve_ms_mean.min(), 0, "ms", "results/E1/summary/runs.csv",
        "min l2_solve_ms_mean [method=AQPS-RT-ET]")
    put("EOneMeanMsMaxRTET", r.l2_solve_ms_mean.max(), 0, "ms", "results/E1/summary/runs.csv",
        "max l2_solve_ms_mean [method=AQPS-RT-ET]")
    put("EOnePNinetyFiveMaxRTET", r.l2_solve_ms_p95.max(), 0, "ms", "results/E1/summary/runs.csv",
        "max l2_solve_ms_p95 [method=AQPS-RT-ET]")


def main():
    for fn in (e1, gantt, e2, e3, e4, e5, e6, e7, provenance):
        fn()
    os.makedirs(C.OUT, exist_ok=True)
    with open(os.path.join(C.OUT, "numbers.tex"), "w", encoding="utf-8", newline="\n") as fh:  # LF on every OS
        fh.write("% generated by postprocess/make_numbers.py -- do not edit by hand; map: numbers_map.csv\n")
        for r in REG:
            fh.write(f"\\newcommand{{{r['macro']}}}{{{r['value']}}}\n")
    with open(os.path.join(C.OUT, "numbers_map.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["macro", "value", "unit", "source", "expression"], lineterminator="\n")
        w.writeheader()
        w.writerows(REG)
    print(f"  numbers: {len(REG)} macros")


if __name__ == "__main__":
    main()
