# AQPS Fix Implementation Plan (two-step)

**Basis:** `AQPS_revision_review_2026-10-15.md` (issues B1–B9, M1–M15, m1–m12) and `AQPS_review_analysis_feedback.md` (Kamar's resolution plan). This plan covers the simulation side. The manuscript work comes after the results exist (Step 2).

**Default values used below:** the plan assumes the recommended answer to each decision in §0. If you choose differently, the affected PR items change as noted.

---

## 0. Decisions (✱ = recommended default)

| # | Decision | Default |
|---|---|---|
| D1 | PV data source | ✱ Real AEMO NEMWeb (already decided) |
| D2 | Eq. 10 | ✱ Hard constraint for the Layer-0 priority set; Eq. 3 exactly as printed; horizon reaches departure |
| D3 | Tier-3 rotation | **KEEP (Kamar, 2026-10-15)** as the *simulated operator decision*. Guard fixed to include return energy. Counted separately; Proposition 2 covers Tiers 1–2 only |
| D4 | Offline benchmark | ✱ MIQP with exact ≤ M EVSE limit (MOSEK, 900 s time limit, report MIP gap); if it times out, report it as a bound and label it |
| D5 | Window | ✱ 15-min step; 30-h window (06:00 day 1 → 12:00 day 2, 120 steps); horizon to latest departure |
| D6 | AQPS-v1 | ✱ Row in master table only (no figures), with the double-count fix |
| D7 | Eq. 2 net import | **APPROVED.** `max(0, Σx − PV − BESS)` |
| D8 | E_req for priority vehicles | **APPROVED; to be confirmed as two-level.** Eq. 10 (hard) uses E_mission = next trip + return + r − arrival energy. While connected, a priority vehicle keeps charging toward E_session through H^NC, capped by Eq. 9. This keeps Tier-2 safe release meaningful: it releases vehicles that are between E_mission and E_session |
| D9 | PV normalisation | **Simple:** divide AEMO VIC1 MW by the maximum VIC1 actual over the 12-month period, then × 20 kW. Nothing for you to supply. Stated as a limitation |
| D10 | Day selection | **Simple:** rank Jul-2024 → Jun-2025 weekdays by daily nMAE; take the days at the 10th, 25th, 50th, 75th, 90th and 97.5th percentiles. The 50th-percentile day drives E1–E3 and E5–E7 |
| D11 | Time base | **APPROVED.** AEST (NEM market time) throughout |
| D12 *(new, pending)* | Preemption timing | ✱ Just-in-time: preempt only when mission slack ≤ δ = 2 h (E6 sweeps δ ∈ {1, 2, 3 h, eager}). Eager preemption (submitted design) cut non-priority energy by about half in the smoke test with no mission gain in S2 |

---

## 1. Step 1: PR #1 (I build it; you merge, run and push)

Branch: `fix/revision-v3`. One PR, with commits grouped by the items below. The PR description lists every change, the config used and the run commands.

### 1.1 Repo hygiene and traceability (M10, M11, m12)
1. Delete `build_aqps_revision.py`, `fix_solver_and_warnings.py`, `fix_physics_and_remove_v1.py` and `build_epsr_overleaf_zip.py`.
2. Move `all_simulation_studies*.py`, `data_generator_v*.py` and `*_bkp.json` into `legacy/`, unchanged.
3. Add `CLAUDE.md`, `docs/revision_plan.md` (the v2 prompt), `docs/wp0_code_audit.md` and `docs/review_2026-10-15.md`.
4. `.gitignore`: ignore only `results/*/raw/`; track `results/*/summary/*.csv` and `results/*/manifest.json`.
5. `config/experiments.yaml`: a single source for every parameter (α weights, thresholds, horizon, solver, seeds, fleet configs).

### 1.2 Real AEMO PV data (B1, D1, D9–D11)
**New: `scripts/download_aemo_pv.py` (you run it; this workspace can't reach NEMWeb)**
- **Source:** NEMWeb MMSDM monthly archive.
  `https://nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM/{YYYY}/MMSDM_{YYYY}_{MM}/MMSDM_Historical_Data_SQLLoader/DATA/PUBLIC_ARCHIVE%23ROOFTOP_PV_{ACTUAL|FORECAST}%23FILE01%23{YYYYMM}010000.zip`
  - I checked this for 2025-06: ACTUAL is about 0.18 MB per month and FORECAST about 30 MB per month.
  - The script also tries `FILE02…` and the older `PUBLIC_DVD_ROOFTOP_PV_*` naming.
- **Arguments:** `--start 2024-07 --end 2025-06 --out data/aemo_raw/`. Downloads are resumable. Expect roughly 0.4 GB in total.
- Writes `data/aemo_raw/checksums.csv` (URL, file, SHA-256, size, download date). The raw zips are **not** committed; the checksums are.
- **Fails loudly** if a file is missing or a required column is absent. There is no synthetic fallback.

**Rewritten: `scripts/prepare_aemo_pv.py`**
- Parses the MMS CSV format (rows tagged I/D) and filters to `REGIONID == "VIC1"`.
- **Actual:** `ROOFTOP_PV_ACTUAL.POWER`. If several `TYPE` values exist, it prefers `MEASUREMENT`, otherwise `SATELLITE`, and records which it used.
- **Forecast:** `ROOFTOP_PV_FORECAST.POWERMEAN`. For each target day D, it uses the **latest run with `VERSION_DATETIME` ≤ D−1 12:00** (the day-ahead vintage; the cut-off is configurable).
- **Intervals:** `INTERVAL_DATETIME` marks the end of each 30-min interval. The script shifts this to interval start, then **holds** each value for two 15-min steps.
- **Normalisation (D9):** MW ÷ the 12-month maximum of VIC1 actual × 20 kW. No extra input is needed.
- **Day selection (D10):**
  - computes daily nMAE (% of 20 kW, daylight intervals) for every weekday;
  - writes `data/aemo_vic1_daily_errors.csv` (all days);
  - takes the days at the 10th, 25th, 50th, 75th, 90th and 97.5th percentiles;
  - writes `data/aemo_vic1_selected_days.json` (forecast and actual, 120 steps per day, with the source fields recorded).
- The metadata states plainly: regional aggregate, AEMO estimate (not metered), hold interpolation, AEST.
- **No APVI claim** unless you later commit an APVI export.

### 1.3 Scenario data (B3, M7, M12, M14, m5, m7, D5, D8)
`data_generator.py`
- 120-step window starting 06:00 AEST. Tariff and PV arrays have 120 values; PV comes from the selected AEMO day.
- **Roster fields per vehicle:**
  - `evening_return_idx`, `next_trip_kwh`, `return_kwh`, `reserve_kwh` (= 0.10·C), `further_trips_today_kwh`, `next_day_dep_idx`, `next_day_req_kwh`, `w_i`;
  - `E_mission` = trip + return + reserve − arrival energy;
  - `E_session` = the session target.
- Overnight-dwelling vehicles depart next morning, inside the window; no clipping at index 95.
- **Contention is defined before any run**, not by baseline outcome: S2/S3 target ratio = (energy demanded in the contention window) / (P_site × window length), and EVSE demand / M, made equal for the 45-EV and 90-EV configurations. Both ratios are written to the scenario metadata.
- **BESS:** fixed schedule with signed values. It charges from PV over 10:00–14:00 (2.5 kW) and discharges over the full peak 15:00–21:00 (about 1.5 kW), 90% round-trip efficiency.

### 1.4 Layer 2 QP (B4, B5, B6 code side, M3, M9, D2, D7)
`src/aqps/tou_optimization.py`
- **Horizon:** H_k = max over connected vehicles of (d_i − k), capped at the end of the window.
- **Variables:** `x` (vehicles × steps), plus one auxiliary non-negative variable g_t for net import (D7). It is still a convex QP.
- **Constraints**
  - Eq. 6: Σᵢ x_it ≤ P_site.
  - Eq. 7: 0 ≤ x ≤ x̄ inside the dwell window.
  - Eq. 8: x_it ≥ 3.84 kW for priority vehicles while remaining E_req > 0. The only clamp is when E_req < 3.84·Δt (last step). The 0.85/0.95 scaling is removed.
  - Eq. 9: delivered ≤ E_req for every vehicle (also the battery-headroom bound).
  - Eq. 10: delivered ≥ E_req for priority vehicles, **hard**.
  - g_t ≥ Σᵢ x_it − PV_t − BESS_t (D7).
- **Objective**
  - α₁ Σ_t C_t g_t Δt
  - + α₂ (Σᵢ(E_req,i − delivered_i))², i.e. Eq. 3 as printed. It cannot cancel out because of Eq. 9.
  - + α₃ Σ_t P_t².
  - Weights (α₁, α₂, α₃) = (1, 100, 1e-5). Check: 2α₃P_site ≤ 0.003 < 0.0075 = (shoulder − off-peak)·Δt.
- **Solver:** taken from config (default MOSEK). No fallback chain and no greedy heuristic. Each solve returns `status`, `solver`, `solve_ms` and `n_vars`.
- **If Eq. 10 is infeasible:** return `status=infeasible`. The scheduler handles it (§1.7). It is never patched silently.

### 1.5 Layer 0 (B4.4, M8, D8)
`src/aqps/priority_selection.py`
- ATC score exactly as Eq. 12: remove `tardy_boost`; keep only a numeric floor on p_i (1 step) and document it.
- **Candidates:** present vehicles with `day_ahead_priority` or w ≥ w_min (config value, stated in the paper).
- **Feasibility test**
  - Solve the Layer-2 QP with hard Eq. 10 over the candidate prefix that Layer 1 would connect now (at most M free or freeable EVSEs).
  - For candidates beyond the free EVSEs, a documented necessary check: an EVSE must be releasable before dᵢ − pᵢ.
  - The paper will say "exact for the connected set; necessary test for queued vehicles". The plan's "check all S" would overstate feasibility when |S| > M, because the QP has no EVSE-count limit.
- Test the full set first (1 solve), then bisect only if it fails. Each event records its solve count.
- **AQPS-static** = day-ahead flags only; the undocumented `atc ≥ 0.35` rule is removed.

### 1.6 Layer 1 (B7, M5, M6, M7, D3)
`src/aqps/queue_manager.py`
- **Priority-on-priority handling (the novel safe-release approach, plan §2.2) is kept.** When all M chargers are occupied and a priority vehicle needs one:
  1. Disconnect the non-priority vehicle with the highest laxity.
  2. Otherwise, **safe-release** a connected priority vehicle that meets (a)(b)(c) in kWh, choosing the largest surplus first, then the latest next departure. It leaves with at least E_mission; its remaining session energy is deferred to the next charging opportunity.
  3. Otherwise, disconnect nobody. The incoming vehicle waits, and the conflict is escalated to the operator impact preview (keep queue / downgrade / promote).
- Tier 3 guarded FIFO rotation is **kept** as the simulated operator decision (no live operator in simulation). Guard: π_new > 1.02·π_head and (E_curr ≥ trip + **return** + reserve, or laxity > Δt). A rotated EV that has met its mission becomes non-priority; otherwise it re-queues with priority.
- **Just-in-time preemption (added after the smoke test; decision D12):** Tiers 1–3 act only when the waiting priority EV's mission slack is ≤ δ (default 2 h). Otherwise it waits for a naturally freed EVSE. A waiting EV whose slack reaches δ triggers the 'shortfall' event.
- Condition (c) uses the roster's `evening_return_idx` and `next_day_dep_idx`; the hard-coded index is gone.
- **Escalation episodes:** one per vehicle per unserved interval (`episode_id`, start, end, outcome).
- **Separate counters:** preemptions, safe releases, escalation episodes, alert count.

### 1.7 Scheduler: event-triggered MPC and preview (B8, M4, M5, plan §2.3)
`src/aqps/scheduler.py`

| Event | Layer 0 | Layer 1 | Layer 2 | Preview |
|---|---|---|---|---|
| Arrival | yes | yes | re-solve | only on a new escalation episode |
| Departure | no | admit waiting vehicle | re-solve | no |
| Operator injection | yes | yes | re-solve | yes |
| Predicted shortfall / infeasible Eq. 10 | yes (bisection) | yes | re-solve | on a new episode |
| PV deviation > τ_pv | no | no | re-solve | only if cost change > τ_cost |
| Periodic T_max | no | no | re-solve | no |

- **Infeasible-Eq.-10 loop:** Layer 0 re-runs once in the same step. If still infeasible (always the case for `static`), solve with Eq. 10 relaxed into the H^NC penalty, log `priority_infeasible`, and count it. This is the honest difference between the ablations.
- Every step logs `l0_solves`, `l2_solves`, `preview_solves`, `total_solves` and solve times.

### 1.8 Baselines (M1, M2, M15, D4, D6)
`src/aqps/baselines.py`
- **LLF and EDF:** unchanged logic; window becomes 120 steps.
- **MPC-only:** the same `TOUOptimizer` with the same weights, `priority_set = ∅` (so no Eq. 8 or 10), FIFO non-preemptive admission.
- **Offline-Optimal:**
  - Uses the same objective and constraints.
  - Adds binary z_it with Σᵢ z_it ≤ M and x_it ≤ x̄ᵢ z_it.
  - Enforces Eq. 10 hard for day-ahead priority vehicles, with perfect information.
  - MOSEK MIQP; time limit 900 s; logs the MIP gap. If the time limit is hit, it reports the incumbent and the bound.
- **AQPS-v1:** a re-implementation of the submitted logic (binary flags, FIFO, every-step QP) on the new data, with no double count.

### 1.9 Experiment runner and metrics (B2, B3, M4, M9, M10, m1, m2)
`run_all_experiments.py`
- **CLI:** `--exp E1..E7 --fleet --scenario --methods --seeds --solver`. Resumable: one raw file per run, skipped if it already exists.
- **Metrics per run**
  - Mission sufficiency (%); priority fulfilment vs E_req.
  - Non-priority fulfilment: at 22:00 and at end of window.
  - Energy delivered (priority, non-priority, total).
  - Cost ($); **$/kWh delivered**; cost by tier; PV self-consumption.
  - Peak, load factor.
  - Preemptions; safe releases; escalation episodes and alerts.
  - L2 solves and total solves; solve mean and p95; infeasible-Eq.-10 count; solver status counts; MIP gap (offline).
- **Manifest:** git hash, config hash, seeds, date, CPU model, OS, Python / CVXPY / MOSEK versions, AEMO checksums.
- **Experiments**

  | ID | Content | Runs |
  |---|---|---|
  | E1 | 8 methods × S1–S3 × {45, 90} | 48 |
  | E2 | 4 scripted cases (90-EV S2) × {LLF, EDF, static, RT-ET}; preview contents exported as CSV for the table | 16 |
  | E3 | 30 seeds × {LLF, EDF, MPC-only, static, RT-ET}; disruption rates from config | 150 |
  | E4 | 6 real AEMO days × {LLF, EDF, MPC-only, static, RT-ET} × {45, 90} | 60 |
  | E5 | RT vs RT-ET × S1–S3 × {45, 90} (solves, times, outcome Δ); T_max sweep {15, 30, 60, 120} on 90-EV S2; ratio/scalability sweep 45/90/180 × {2:1, 3:1, 4:1} | — |
  | E6 | α₃ ∈ {1e-6…1e-2} (log), α₂ ∈ {10…1000}, k_atc ∈ {0.5…3} **on a configuration with \|candidates\| > M**, r ∈ {2…14 kWh}, τ_pv, τ_cost | — |
  | E7 | Priority share 30→90% assigned **within the mid-day contended cohort** (not the earliest arrivals) × all online methods, including LLF and EDF | — |

### 1.10 Smoke test before the PR (only here; numbers not reported)
One scenario, one fleet, one seed, solver CLARABEL set explicitly in config (MOSEK isn't licensed in my workspace). Offline-MIQP gets a syntax and build check only. AEMO parsing is checked against a small hand-made MMS-format fixture (shape only).

---

## 2. Your run (after merging PR #1)
```bash
git checkout main && git pull
uv sync                                    # MOSEK licence in ~/mosek/mosek.lic
python scripts/download_aemo_pv.py --start 2024-07 --end 2025-06
# optional: add data/vic_rooftop_capacity.csv (month, MW, source)
python scripts/prepare_aemo_pv.py
python data_generator.py
python run_all_experiments.py --exp all    # resumable
git add data/aemo_raw/checksums.csv data/aemo_vic1_* simulation_data_*.json \
        results/*/summary results/*/manifest.json
git commit -m "Results: revision v3 (E1–E7)" && git push
```
The PR will give a runtime estimate for each experiment, based on the smoke-test timings. The longest will be E3 (150 runs) and the Offline MIQP (up to 900 s × 6 runs in E1).

---

## 3. Step 2: PR #2 (after your results are pushed)
1. **Completeness check** against each manifest: methods, scenarios, seeds, git hash = merged PR, solver = MOSEK, `fallback`/`infeasible` counts. I report any gaps before using the data.
2. **Post-processing** (`generate_all_figures.py`, `make_tables.py`): summaries → LaTeX tables and figures (Tol palette, serif, 300 dpi, 3.5" or 7"; fixed `figsize`, no `bbox_inches="tight"` drift).
3. `docs/results_map.md` and a number → source file list, generated by script (fixes the lost `$`).
4. Then the manuscript stages (template fix, §2–§5 rewrite from the new numbers, corrected Propositions 1 and 2, change list).

### Simplified figure set (13 → 8 figures)

| New | Content | Replaces | Change |
|---|---|---|---|
| Fig. 1 | Depot system diagram | old Fig. 1 | keep |
| Fig. 2 | AQPS Layer 0/1/2 + operator loop | fig2_system_architecture | redraw, no overlapping labels, fewer words |
| Fig. 3 | Service: mission sufficiency and non-priority energy (2 rows × 2 fleets, methods as bars) | fig_fulfillment_stacked | drop the "priority fulfilment" row (in table); same palette |
| Fig. 4 | Cost by tier (stacked) with $/kWh marker, S2 and S3 × {45, 90} | cumulative cost 45 / 90 + cost breakdown (3 figs) | 3 → 1 |
| Fig. 5 | Gantt, 90-EV S3 only, 6 vehicles, AQPS-RT-ET vs LLF | Gantt S2 + S3 | 2 → 1; S2 Gantt dropped |
| Fig. 6 | Queue timeline (swimlane): 5–6 vehicles, states connected / waiting / released / escalated, injection marker | fig_queue_trajectory_injection | one panel, no power subplot |
| Fig. 7 | AEMO: (a) forecast vs actual for 3 of the 6 days (one per error tercile); (b) $/kWh vs daily nMAE | fig_aemo_pv_uncertainty | keep 2 panels, simplified |
| Fig. 8 | Sensitivity: (a) α₃ sweep ($/kWh, non-priority %), (b) reserve r sweep (mission %, $/kWh) | 4-panel sensitivity | 4 → 2 panels; α₂ and k_atc go into a table or text |
| — | Monte Carlo results | fig_monte_carlo_disruptions | **figure dropped** → mean ± 95% CI table |
| — | Event-triggered solves | fig_event_triggered_solves | **figure dropped** → E5 table (L2 / total solves, ms, outcome Δ) |

**Tables:** ToU; scenarios (with contention ratios); events; master E1 (+ kWh, $/kWh, total solves); E2 cases + one preview example; E3 Monte Carlo; E5 computation and scalability; E7 overload (all methods).

---

## 4. Points in the feedback file that need correcting
1. **Predicted outcomes stated as facts:**
   - "makes AQPS unit cost lower than LLF" (B2);
   - "AQPS-RT … achieving higher priority fulfillment … than static" (B8);
   - "total QP solves 35–55% lower" (B8).

   These are hypotheses. The plan measures them; the paper will report whatever comes out.
2. **B2/M12 overnight recovery only helps vehicles that stay overnight.** Daytime non-priority vehicles with 4–6 h dwell can't be recovered by extending the window. The main reason for low non-priority fulfilment is that α₃ flattening plus a low non-priority weight keeps EVSEs occupied at low power. The fixes in §1.4 address that, but the outcome must be measured.
3. **B3 "recalibrate 45-EV S2 contention":** define contention by an a-priori ratio (§1.3). Tuning the scenario until LLF fails would be a biased design.
4. **B6 "MOSEK converges to the analytic centre":** don't claim this; it isn't guaranteed (presolve, basis identification). Claim only that the optimal value and the aggregate profile P*_t are unique (true, because the objective is strictly convex in P_agg).
5. **M3 `max(0,·)` in Eq. 2:** this changes a printed equation (D7). It adds an auxiliary variable, so the invariant "only decision variable is x" becomes "x plus an epigraph variable for net import".
6. **M8 "check feasibility over all candidates S":** wrong when |S| > M, because the QP has no EVSE count; see §1.5.
7. **m7 BESS:** charging must also be scheduled and signed (m5), not only discharge.
8. **B1:** "APVI cross-check" must not be claimed unless an APVI export is committed.
