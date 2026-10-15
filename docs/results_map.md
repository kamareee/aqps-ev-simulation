# Results map (revision v3)

Each run writes `results/<EXP>/raw/<run_id>.json.gz` (not committed) and `results/<EXP>/summary/runs.csv`
(committed, one row per run). Post-processing scripts (Step 2) aggregate these into paper tables and figures.

| Exp | Content | Summary file(s) | Paper section | Figure / table (planned, simplified set) |
|---|---|---|---|---|
| E1 | 7 methods (AQPS-static removed) × S1–S3 × {45, 90} EVs (reference AEMO day) | `E1/summary/runs.csv` (+ traces in raw) | §4.3 | Table: master results (mission %, non-priority %, kWh, cost, $/kWh, solves); Fig: service (mission %, non-priority %); Fig: cost by tier with $/kWh; Fig: Gantt 90-EV S3 |
| E2 | 4 scripted injections, 90-EV S2 | `E2/summary/runs.csv`, `E2/summary/previews.csv` | §4.4 | Table: case studies + one preview example; Fig: queue swimlane (case 1) |
| E3 | 30 random-disruption seeds, 90-EV **S3** (S2 version in `results/E3_S2`) | `E3/summary/runs.csv` | §4.5 | Table: mean ± 95 % CI |
| E4 | 6 real AEMO days (nMAE percentiles) × {45, 90} EVs | `E4/summary/runs.csv` | §4.6 | Fig: (a) forecast vs actual (3 days), (b) $/kWh vs nMAE |
| E5 | (a) RT vs RT-ET, (b) T_max sweep, (c) fleet / EV:EVSE ratio | `E5/summary/runs.csv` | §4.7 | Table: L2 solves, total solves, solve ms, outcome difference; scalability rows |
| E6 | α3, α2, k_atc (congested), reserve, τ_pv, preemption slack δ | `E6/summary/runs.csv` | §4.8 | Fig: (a) α3, (b) reserve; others in a table |
| E7 | Priority share 30–90 % on 45-EV **S3** (cohort centred on the scenario's priority arrivals); S2 version in `results/E7_S2` | `E7/summary/runs.csv` | §4.8 / §3.5 | Table: overload behaviour |

Inputs recorded in each `manifest.json`: git commit, config hash and snapshot, solver and versions,
SHA-256 of `data/aemo_vic1_selected_days.json`, `data/operator_injections.json`,
`data/aemo_raw/checksums.csv` and the three `simulation_data_*EVs.json` files.
