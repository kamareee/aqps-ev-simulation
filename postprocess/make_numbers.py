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
    c = d[(d.part == "c_scalability") & (d.method == "AQPS-RT-ET")]
    big = c[c.fleet_size == 180]
    put("ScalMeanMsMin", big.l2_solve_ms_mean.min(), 0, "ms", src, "min l2_solve_ms_mean [part=c, 180 EVs, RT-ET]")
    put("ScalMeanMsMax", big.l2_solve_ms_mean.max(), 0, "ms", src, "max l2_solve_ms_mean [part=c, 180 EVs, RT-ET]")
    put("ScalPNinetyFiveMax", big.l2_solve_ms_p95.max(), 0, "ms", src, "max l2_solve_ms_p95 [part=c, 180 EVs, RT-ET]")
    g = d[(d.part == "c_scalability")]
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


def provenance():
    m = json.load(open(os.path.join(C.RES, "E1", "manifest.json"), encoding="utf-8"))
    put("SimCommit", m["git_commit"][:7], 0, "", "results/E1/manifest.json", "git_commit (first 7)")
    put("MosekVersion", m["versions"]["mosek"], 0, "", "results/E1/manifest.json", "versions.mosek")
    put("CvxpyVersion", m["versions"]["cvxpy"], 0, "", "results/E1/manifest.json", "versions.cvxpy")


def main():
    for fn in (e1, e2, e3, e4, e5, e6, e7, provenance):
        fn()
    os.makedirs(C.OUT, exist_ok=True)
    with open(os.path.join(C.OUT, "numbers.tex"), "w", encoding="utf-8") as fh:
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
