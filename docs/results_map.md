# Results map (revision v3)

Each run writes `results/<EXP>/raw/<run_id>.json.gz` (not committed) and `results/<EXP>/summary/runs.csv`
(committed, one row per run). Post-processing scripts (Step 2) aggregate these into paper tables and figures.

| Exp | Content | Summary file(s) | Paper section | Figure / table (planned, simplified set) |
|---|---|---|---|---|
| E1 | 8 methods × S1–S3 × {45, 90} EVs (reference AEMO day) | `E1/summary/runs.csv` (+ traces in raw) | §4.3 | Table: master results (mission %, non-priority %, kWh, cost, $/kWh, solves); Fig: service (mission %, non-priority %); Fig: cost by tier with $/kWh; Fig: Gantt 90-EV S3 |
| E2 | 4 scripted injections, 90-EV S2 | `E2/summary/runs.csv`, `E2/summary/previews.csv` | §4.4 | Table: case studies + one preview example; Fig: queue swimlane (case 1) |
| E3 | 30 random-disruption seeds, 90-EV **S3** (S2 version in `results/E3_S2`) | `E3/summary/runs.csv` | §4.5 | Table: mean ± 95 % CI |
| E4 | 6 real AEMO days (nMAE percentiles) × {45, 90} EVs | `E4/summary/runs.csv` | §4.6 | Fig: (a) forecast vs actual (3 days), (b) $/kWh vs nMAE |
| E5 | (a) RT vs RT-ET, (b) T_max sweep, (c) fleet / EV:EVSE ratio | `E5/summary/runs.csv` | §4.7 | Table: L2 solves, total solves, solve ms, outcome difference; scalability rows |
| E6 | α3, α2, k_atc (congested), reserve, τ_pv, preemption slack δ | `E6/summary/runs.csv` | §4.8 | Fig: (a) α3, (b) reserve; others in a table |
| E7 | Priority share 30–90 % on 45-EV **S3** (cohort centred on the scenario's priority arrivals); S2 version in `results/E7_S2` | `E7/summary/runs.csv` | §4.8 / §3.5 | Table: overload behaviour |

Inputs recorded in each `manifest.json`: git commit, config hash and snapshot, solver and versions,
SHA-256 of `data/aemo_vic1_selected_days.json`, `data/operator_injections.json`,
`data/aemo_raw/checksums.csv` and the three `simulation_data_*EVs.json` files.

## Post-processing (Step 2)

`uv run --frozen python -m postprocess.build` (locked versions; figure bytes depend on the matplotlib version) regenerates everything below from committed files only
(summaries, E1/E2 raw traces, AEMO day file, fleet JSONs, config). AQPS-v1 is excluded from all
outputs (decision 2026-10-16); AQPS-RT appears in tables only.

| Output | Source | Content |
|---|---|---|
| `paper/figures/fig1_depot.pdf`, `fig2_architecture.pdf` | TikZ sources in `paper/figures/src/` (pdflatex) | Depot model; AQPS architecture |
| `paper/figures/fig3_service.pdf` | E1 runs.csv | Mission sufficiency and non-priority fulfilment, S1–S3 × {45, 90} |
| `paper/figures/fig4_cost_tiers.pdf` | E1 runs.csv | Grid cost by tariff tier, $/kWh above bars, S2/S3 × {45, 90} |
| `paper/figures/fig5_gantt.pdf` | E1 raw (90-EV S3, AQPS-RT-ET and LLF) | 3 priority EVs with largest LLF shortfall + 3 most-preempted non-priority EVs |
| `paper/figures/fig6_swimlane.pdf` | E2 raw (case 1, AQPS-RT-ET) | EVSE state per vehicle; Tier 1/2/3 events from the trace |
| `paper/figures/fig7_aemo.pdf` | AEMO day file, E4 runs.csv | Forecast vs actual (p10, p50, p97.5 days); $/kWh vs nMAE |
| `paper/figures/fig8_sensitivity.pdf` | E6 runs.csv | α3 and reserve sweeps |
| `paper/tables/tab_tou.tex` (not used in the manuscript; tariff in text), `tab_scenarios.tex` | config, fleet JSON metadata | Inputs |
| `paper/tables/tab_e1_master.tex` | E1 | Master table (7 methods) |
| `paper/tables/tab_e2_cases.tex`, `tab_e2_preview.tex` | E2 runs.csv, previews.csv | Case studies, preview example |
| `paper/tables/tab_e3_montecarlo.tex` | E3 | Mean ± 95 % t-CI over 30 seeds |
| `paper/tables/tab_e5_et.tex`, `tab_e5_tmax.tex` (values quoted in text), `tab_e5_scal.tex` | E5 | ET vs RT, T_max, scalability |
| `paper/tables/tab_e6.tex` | E6 | α2, k_ATC, τ_pv, δ |
| `paper/tables/tab_e7.tex` | E7 | Priority overload |
| `paper/numbers.tex`, `paper/numbers_map.csv` | all | Every number quoted in the text, as a macro, with its source and expression |
