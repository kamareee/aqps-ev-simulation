# AQPS — EPSR manuscript: change list and traceability

**Manuscript:** "Schedule-Aware Real-time Depot Charging for Electric Fleets" (EPSR, Full Length Article).
**Results:** simulation commit `070d9e2` (MOSEK 11.2.6, CVXPY 1.8.1); results commit `b3ce86e`.
**Build:** `uv run --frozen python -m postprocess.build`, then `uv run --frozen python manuscript/assemble.py`.

## 1. Compliance with the EPSR Guide for Authors and the Editorial Manager LaTeX guide

| Requirement | Status |
|---|---|
| Full Length Article ≤ 7000 words, counting abstract, text, tables and figures, excluding references | About 6000 words (abstract, text, tables, captions, end declarations); about 6600 including text inside figures. Count method: detex for text and tables; pdftotext for text inside figures. |
| Article length stated in the cover letter (compulsory) | Added. |
| Abstract ≤ 250 words, no citations | About 215 words; no citations. |
| 1–7 keywords | 6. |
| Highlights: 3–5 bullets, ≤ 85 characters each, separate file with "highlights" in its name | 5 bullets, 67–83 characters; file `highlights.tex`. |
| Layout | Two-column (`\reviewlayoutfalse`) is the default; the guide allows double-column for LaTeX submissions. Setting `\reviewlayouttrue` gives the EPSR template layout: `preprint,12pt` + `epsr.sty`, single column, double spacing, 29 pages. Both compile. |
| Editorial Manager: no subfolders; no two files with the same name; one period per file name | Flat folder. The TikZ sources stay in the repo (`paper/figures/src`) and are not uploaded. |
| Style file uploaded | `elsarticle.cls`, `elsarticle-num-names.bst` and `epsr.sty` are included. |
| Vector figures with embedded fonts | All figures are PDFs with embedded fonts. |
| Numbered sections | Yes. |
| Equations numbered in the order they are referred to | Fixed: Eq. 11 is no longer cited before Eqs. 1–10. |
| Tables: no vertical rules, notes below the table | Booktabs; notes below the table. |
| References numbered in order of citation, every reference real | Checked; one unverified reference removed (see §4). DOIs are recommended by the guide but have not been added yet. |
| CRediT statement | Present. |
| Competing interests | Present. You still need to complete the declarations tool and upload its .docx. |
| Funding statement (required) | Added, using the guide's no-funding sentence. **Please confirm.** |
| Data statement (Option C: deposit, cite and link) | Cites the public GitHub repository as a [dataset] reference. **Recommended:** archive a release on Zenodo and add its DOI. |
| Generative-AI declaration (new section before the references) | Added. Figs. 1–2 captions disclose AI assistance. **Please confirm the wording.** |

## 2. Main changes for the word limit (previously about 9400 words)

**Text**
- Every section was tightened. Content, numbers and claims are unchanged.

**Tables**
- **Table 3 (E1):** the 45- and 90-vehicle depots now sit side by side. Columns kept: mission, non-priority, energy, unit cost, solves. AQPS-RT is dropped here because its outcomes equal AQPS-RT-ET (Table 7).
- **Table 4 (E2):** mission, non-priority, safe releases and escalation episodes only.
- **Table 7 (E5a):** solve counts and solve times only. The deltas between event-triggered and every-step runs are given in the text.
- **Table 8 (E5c):** LLF vs AQPS-RT-ET only. The caption states that both meet every mission.
- **Sensitivity table (E6):** removed. Fig. 8 and the text keep the results.
- **ToU tariff table and T_max table:** replaced by text.

**Figures**
- Figs. 3, 4 and 7 are slightly shorter so that each two-figure page fits.

## 3. Reviewer comments (J. Energy Storage, EST-D-26-06112) → where addressed

| Comment | Change | Where |
|---|---|---|
| R5.1, R2.1: novelty and coupling | Layer 0 admits through a QP feasibility test with hard Eq. 10; a Layer-2 shortfall triggers Layer 0. | §1, §3, Fig. 2 |
| R5.2: priority assignment; PV known | Roster-derived mission energy with ATC ranking; real AEMO forecast and actual PV. | §3.1, §4.2 |
| R5.3, R1.1, R2.2, R3.1, R4.3: single LLF baseline | 7 methods: LLF, EDF, MPC-only, Offline-Opt. (MIQP), AQPS-static, AQPS-RT, AQPS-RT-ET. | §4.3, Tables 3–9 |
| R5.4, R2.5: convexity, solver, real time | Proposition 1 corrected; problem size stated; MOSEK, no fallback; solve times reported. | §3.4, §5.5 |
| R5.5, R3.2: non-priority service | Reported, compared with the perfect-information bound; the paper states where LLF is better. | §5.1, §5.5, §5.8 |
| R3.3: guarantee conditions | Proposition 2: sufficient conditions (A1)–(A3) and admission conditions (C1)–(C3). | §3.4 |
| R1.2, R2.3: weights | Marginal-value argument, confirmed by E6. | §2.2, §5.6 |
| R2.4, R2.6, R4.2: uncertainty | E2 scripted disruptions, E3 Monte Carlo, E4 real AEMO days. | §5.2–5.4 |
| R2.7, R4.1: scalability | E5c: 45–180 EVs, EV:EVSE ratios 2:1–4:1. | §5.5 |
| R2.8: FMS integration | Discussion paragraph. | §3.3, §5.8 |
| R1.3: PV/BESS sizing | Rationale given; BESS charging now counted as load. | §4.1 |
| R1.5: Fig. 3 bug | All figures regenerated by scripts. | Figs. 3–8 |
| R3.5, R5.6: typos | Text rewritten. | — |
| R2.9: recent references | Only verified references kept. | References |
| R4.4: degradation | Stated as an assumption. | §2.1 |

## 4. Number → source

- Every result quoted in the text is a `\R…` macro from `numbers.tex`.
- `numbers_used.csv` lists the 111 macros used, each with its value, source file and expression.
- Tables are generated by `postprocess/make_tables.py`; Figs. 3–8 by `postprocess/make_figures.py`; Figs. 1–2 from `paper/figures/src/*.tex`.
- Literal numbers in the text are inputs only: configuration, scenario generator, injections and tariff.

**References**
- Removed: `skugor2025hierarchical` (could not be verified).
- Added:
  - Heemels et al. 2012;
  - Kellerer et al. 2004;
  - Boyd & Vandenberghe 2004;
  - CVXPY;
  - MOSEK;
  - AEMO MMS data model;
  - the code and data repository (as a [dataset] reference).

## 5. Please confirm before submission

1. **Funding statement:** the no-funding sentence, or your actual funder and grant number.
2. **Generative-AI declaration:** the wording.
3. **Data statement:** whether to archive the repository on Zenodo and cite its DOI.
4. **Cover letter:** keep or delete the optional paragraph disclosing the earlier submission.
5. **Hui Song's affiliation:** "School of Information and Physical Sciences" is used.
6. **Layout:** two-column (default) or the EPSR single-column review layout (one switch in `main.tex`).
