# AQPS Revision Results Map (`docs/results_map.md`)

| Experiment ID | Manuscript Section | Figure / Table | Summary CSV Source | Post-Processing Function |
| :--- | :--- | :--- | :--- | :--- |
| **E1** (Main 8 Methods x S1-S3 x 2 Fleets) | §4.3 Main Results | Table 3 (Master Table), Fig. 3 (Fulfillment), Figs. 4-5 (Cumulative Cost), Figs. 6-7 (Per-EV Gantt), Fig. 8 (Cost Breakdown) | `results/E1_main/summary/e1_master_table.csv` | `generate_all_figures.py` |
| **E2** (Scripted Operator Injections) | §4.4 Real-Time Queue Case Studies | Fig. 9 (Queue Trajectory with Injection Markers), Table 4 (Operator Impact Preview) | `results/E2_case_studies/summary/e2_case_studies.csv` | `generate_all_figures.py` |
| **E3** (Random-Disruption Monte Carlo, 30 runs) | §4.5 Disruption Monte Carlo | Fig. 10 (Monte Carlo Fulfillment & Cost Box/CI Plot) | `results/E3_monte_carlo/summary/e3_mc_summary.csv` | `generate_all_figures.py` |
| **E4** (Real AEMO VIC1 PV Forecast vs Actual) | §4.6 PV Forecast Uncertainty | Fig. 11 (AEMO VIC1 Profiles & Error Impact), Table 5 | `results/E4_aemo_pv/summary/e4_aemo_pv_results.csv` | `generate_all_figures.py` |
| **E5 / E5b** (Event-Triggered vs Every-Step & Scalability) | §4.7 Computational Efficiency & Scalability | Fig. 12 (Solve Count & Time Reduction), Table 6 (Scalability 45/90/180 EVs & Ratios 2:1-4:1) | `results/E5_scalability/summary/e5a_event_trigger_comparison.csv`, `e5b_scalability_ratios.csv` | `generate_all_figures.py` |
| **E6** (Hyperparameter Sensitivity) | §4.8 Sensitivity Analysis | Fig. 13 (4-Panel Sensitivity Plateau: $lpha_2, lpha_3, k, r$) | `results/E6_sensitivity/summary/e6_sensitivity.csv` | `generate_all_figures.py` |
| **E7** (Overloaded Priority & Escalation) | §3.5 & §4.8 Feasibility Condition | Table 7 / Fig. 14 (Overloaded Priority Fulfilment, Safe Releases & Escalations) | `results/E7_overloaded/summary/e7_overloaded_priority.csv` | `generate_all_figures.py` |
