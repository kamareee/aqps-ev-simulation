# AQPS EV depot charging simulator: context for Claude sessions

Paper: "Schedule-Aware Real-time Depot Charging for Electric Fleets" (AQPS). Rejected by J. Energy Storage
(EST-D-26-06112); resubmitting to Electric Power Systems Research (EPSR).
Authors: M. K. Uddin (RMIT), A. Moradi Amani (RMIT), H. Song (Univ. of Newcastle).

Read first: `docs/fix_plan_v3.md` (current plan and decisions), `docs/review_2026-10-15.md` (issue list),
`docs/results_map.md` (experiment → paper mapping).

## Workflow (do not deviate)
- Code changes: feature branch → PR → Kamar merges. Never commit to `main`.
- Kamar runs the simulations locally (MOSEK licence) and pushes `results/<EXP>/summary/*.csv` and
  `results/<EXP>/manifest.json`. Claude only runs short smoke tests (`--smoke --solver CLARABEL`);
  smoke numbers are never reported.
- Figures and tables come from post-processing scripts reading the committed summaries; every number
  in the manuscript must trace to a committed summary file.
- No tests unless asked. Concise communication; state assumptions; flag paper/code mismatches.

## Model invariants (revision v3)
- Layer 2 = convex QP (Eqs. 1–10). Decision variables: EV powers x_it and an epigraph variable g_t for
  net grid import max(0, Σx − PV − BESS) (approved change to Eq. 2). No slack variables.
- Eq. 3 = aggregate (Σ_i (E_sess_i − delivered_i))², exactly as printed (non-negative by Eq. 9).
- Eq. 4 = α3 Σ_t P_t², α3 = 1e-5 (2·α3·P_site < (C_shoulder − C_offpeak)·Δt, so it never overrides the tariff).
- Eq. 6 gross site limit; Eq. 5 with L = 1 reduces to Eq. 6.
- Eq. 8: priority EVs get ≥ 3.84 kW from now until their mission energy is covered.
- Eq. 10 HARD for priority EVs with E_req = E_mission = next trip + return + reserve − arrival energy.
  E_session (≥ E_mission) is the soft target via Eq. 3, capped by Eq. 9.
- Horizon reaches the latest departure of connected EVs. Solver from config, no silent fallback.
- Layer 0: ATC score (Eq. 12), committed priority EVs never demoted, full-set test then bisection.
- Layer 1: just-in-time preemption (slack ≤ δ) → Tier 1 non-priority (largest laxity) → Tier 2 safe
  release (a)(b)(c) in kWh → Tier 3 guarded FIFO rotation (= simulated operator decision) → Tier 4
  escalate. Proposition 2 (no disconnection below mission energy) covers Tiers 1–2.
- Time base AEST (NEM market time); 120 × 15-min steps, 06:00 D → 12:00 D+1.
- PV: real AEMO VIC1 rooftop PV (ROOFTOP_PV_ACTUAL / ROOFTOP_PV_FORECAST day-ahead vintage), scaled to
  20 kW by the period maximum. Never synthetic.
- All methods share M EVSEs and the gross site limit; all baselines are internal (no ACN-Sim).

## Layout
- `config/experiments.yaml`: every parameter.
- `scripts/download_aemo_pv.py`, `scripts/prepare_aemo_pv.py`: real PV data.
- `data_generator.py`: scenarios and E2 injections.
- `src/aqps/`: `tou_optimization.py` (L2), `priority_selection.py` (L0), `queue_manager.py` (L1),
  `scheduler.py` (static / RT / RT-ET), `impact_preview.py`, `baselines.py`, `legacy_v1.py` (submitted method).
- `run_all_experiments.py`: E1–E7.
- `postprocess/`: figures, tables and `paper/numbers.tex` from committed results (`python -m postprocess.build`).
- `paper/`: generated figures/tables/numbers (do not edit by hand).
- `legacy/`: superseded code and documents (kept for traceability, not used).
