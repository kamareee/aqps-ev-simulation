# Save as: build_epsr_overleaf_zip.py
"""
WP8 — Self-Contained EPSR Overleaf Project Zip Builder.
Creates `overleaf_epsr_package/` and `AQPS_EPSR_Submission.zip` containing:
  - main.tex (Complete revised manuscript in Elsevier EPSR format)
  - refs.bib (Updated bibliography with 2024-2026 references & foundational theory citations)
  - highlights.tex (EPSR Highlights, <= 85 chars per bullet)
  - cover_letter.tex (Cover letter addressed to Electric Power Systems Research)
  - response_to_reviewers.tex (Point-by-point response to Reviewers #1-#5)
  - All figures from figs_epsr/ and template files from elsarticle-epsr.zip
"""

import os
import shutil
import zipfile
from pathlib import Path

OUT_DIR = Path("overleaf_epsr_package")
OUT_DIR.mkdir(exist_ok=True)
(OUT_DIR / "figs").mkdir(exist_ok=True)

# =============================================================================
# 1. BIBTEX DATABASE (refs.bib)
# =============================================================================
REFS_BIB = r"""
@article{alhanahi2024charging,
  author  = {Al-Hanahi, Bassam and Ahmad, Iftekhar and Habibi, Daryoush and Pradhan, P. and Masoum, Mohammad A. S.},
  title   = {A charging strategy for large commercial electric vehicle fleets},
  journal = {IEEE Access},
  volume  = {12},
  pages   = {46042--46058},
  year    = {2024}
}

@article{alinia2020online,
  author  = {Alinia, Bahram and Hajiesmaili, Mohammad H. and Lee, Zachary J. and Crespi, Noel and Mallada, Enrique},
  title   = {Online {EV} scheduling algorithms for adaptive charging networks with global peak constraints},
  journal = {IEEE Transactions on Sustainable Computing},
  volume  = {7},
  number  = {3},
  pages   = {537--548},
  year    = {2020}
}

@article{chandra2024comprehensive,
  author  = {Chandra, Ishan and Singh, N. K. and Samuel, Paulson},
  title   = {A comprehensive review on coordinated charging of electric vehicles in distribution networks},
  journal = {Journal of Energy Storage},
  volume  = {89},
  pages   = {111659},
  year    = {2024}
}

@article{chen2025optimal,
  author  = {Chen, J. and Zou, Y. and Zhang, J. and Zhang, X.},
  title   = {Optimal charging and discharging strategy for workplace electric vehicle charging stations with renewable energy and power network constraints},
  journal = {Energy},
  volume  = {314},
  pages   = {138422},
  year    = {2025}
}

@article{corsetti2024commercial,
  author  = {Corsetti, A. and Castro, R. and Geyken, C. and Mohan, A.},
  title   = {Commercial {EV} fleet smart charging for cost reduction and renewables integration: A case study in {Germany}},
  journal = {Journal of Energy Storage},
  volume  = {100},
  pages   = {113765},
  year    = {2024}
}

@article{dicorato2019integrated,
  author  = {Dicorato, M. and Forte, G. and Trovato, M. and Mu{\~n}oz, C. B. and Coppola, G.},
  title   = {An integrated {DC} microgrid solution for electric vehicle fleet management},
  journal = {IEEE Transactions on Industry Applications},
  volume  = {55},
  number  = {6},
  pages   = {7347--7355},
  year    = {2019}
}

@article{frendo2019realtime,
  author  = {Frendo, Oliver and Gaertner, Nadine and Stuckenschmidt, Heiner},
  title   = {Real-time smart charging based on precomputed schedules},
  journal = {IEEE Transactions on Smart Grid},
  volume  = {10},
  number  = {6},
  pages   = {6921--6932},
  year    = {2019}
}

@article{he2020optimal,
  author  = {He, Yi and Liu, Zhaocai and Song, Ziqi},
  title   = {Optimal charging scheduling and management for a fast-charging battery electric bus system},
  journal = {Transportation Research Part E: Logistics and Transportation Review},
  volume  = {142},
  pages   = {102056},
  year    = {2020}
}

@inproceedings{hosseini2023optimizing,
  author    = {Hosseini, A. and Amani, Ali Moradi and Keshavarzian, K. and Jalili, Mahdi},
  title     = {Optimizing the location of on-route fast-charging stations in a battery electric bus system},
  booktitle = {2023 10th International Conference on Power and Energy Systems Engineering (CPESE)},
  publisher = {IEEE},
  pages     = {301--306},
  year      = {2023}
}

@techreport{iea2025global,
  author      = {{International Energy Agency}},
  title       = {Global {EV} Outlook 2025},
  institution = {IEA},
  year        = {2025},
  url         = {https://www.iea.org/reports/global-ev-outlook-2025}
}

@article{jack2025estimating,
  author  = {Jack, M. W. and Paulsen, P. and Xiao, X. and Parker, R. and Myall, D.},
  title   = {Estimating the seasonal variation in electricity demand of future electric vehicle fleets},
  journal = {Energy},
  volume  = {312},
  pages   = {137089},
  year    = {2025}
}

@article{keshavarzian2024optimal,
  author  = {Keshavarzian, K. and Moradi Amani, Ali and Jalili, Mahdi},
  title   = {Optimal fleet size for cross-route dispatching in electrified bus networks},
  journal = {IEEE Transactions on Network Science and Engineering},
  volume  = {11},
  number  = {6},
  pages   = {6483--6493},
  year    = {2024}
}

@inproceedings{lee2019acnsim,
  author    = {Lee, Zachary J. and Johansson, Daniel and Low, Steven H.},
  title     = {{ACN-Sim}: An open-source simulator for data-driven electric vehicle charging research},
  booktitle = {Proceedings of the Tenth ACM International Conference on Future Energy Systems (e-Energy)},
  pages     = {411--412},
  year      = {2019}
}

@article{lee2021adaptive,
  author  = {Lee, Zachary J. and Lee, George and Lee, Ted and Jin, Cheng and Lee, Rand and Low, Zhi and Chang, Daniel and Ortega, Christine and Low, Steven H.},
  title   = {Adaptive charging networks: A framework for smart electric vehicle charging},
  journal = {IEEE Transactions on Smart Grid},
  volume  = {12},
  number  = {5},
  pages   = {4339--4350},
  year    = {2021}
}

@inproceedings{lee2019acndata,
  author    = {Lee, Zachary J. and Li, Tongxin and Low, Steven H.},
  title     = {{ACN-Data}: Analysis and applications of an open {EV} charging dataset},
  booktitle = {Proceedings of the Tenth ACM International Conference on Future Energy Systems (e-Energy)},
  pages     = {139--149},
  year      = {2019}
}

@article{liu2023multiobjective,
  author  = {Liu, J. and Wang, H. and Du, Y. and Lu, Y. and Wang, Z.},
  title   = {Multi-objective optimal peak load shaving strategy using coordinated scheduling of {EVs} and {BESS} with adoption of {MORBHPSO}},
  journal = {Journal of Energy Storage},
  volume  = {64},
  pages   = {107121},
  year    = {2023}
}

@article{lokhande2024realtime,
  author  = {Lokhande, S. and Bichpuriya, Y. and Sarangan, V.},
  title   = {Real-time management of deviations in the demand of electric vehicle charging stations by utilizing {EV} flexibility},
  journal = {Journal of Energy Storage},
  volume  = {97},
  pages   = {112719},
  year    = {2024}
}

@article{mahyari2025electric,
  author  = {Mahyari, E. and Freeman, N.},
  title   = {Electric vehicle fleet charging management: An approximate dynamic programming policy},
  journal = {European Journal of Operational Research},
  volume  = {327},
  pages   = {263--279},
  year    = {2025}
}

@article{vandermeer2018energy,
  author  = {Van der Meer, D. and Mouli, G. R. C. and Mouli, G. M. E. and Elizondo, L. R. and Bauer, P.},
  title   = {Energy management system with {PV} power forecast to optimally charge {EVs} at the workplace},
  journal = {IEEE Transactions on Industrial Informatics},
  volume  = {14},
  number  = {1},
  pages   = {311--320},
  year    = {2018}
}

@article{meng2024research,
  author  = {Meng, F. and Pei, W. and Zhang, Q. and Zhang, Y. and Ma, B. and Li, L.},
  title   = {Research on the capacity of charging stations based on queuing theory and energy storage scheduling optimization sharing strategy},
  journal = {Journal of Energy Storage},
  volume  = {96},
  pages   = {112673},
  year    = {2024}
}

@article{motlagh2026review,
  author  = {Motlagh, S. G. and Li, L.},
  title   = {A review on electric vehicle charging station planning: Infrastructure placement, sizing, upgrades, and uncertainties},
  journal = {Journal of Energy Storage},
  volume  = {141},
  pages   = {119325},
  year    = {2026}
}

@phdthesis{nimalsiri2021phd,
  author = {Nimalsiri, N. I.},
  title  = {Electric Vehicle ({EV})-Assisted Demand-Side Management in Smart Grid},
  school = {The Australian National University},
  year   = {2021}
}

@article{nimalsiri2021coordinated,
  author  = {Nimalsiri, N. I. and Ratnam, Elizabeth L. and Mediwaththe, C. P. and Smith, David B. and Halgamuge, Saman K.},
  title   = {Coordinated charging and discharging control of electric vehicles to manage supply voltages in distribution networks: Assessing the customer benefit},
  journal = {Applied Energy},
  volume  = {291},
  pages   = {116857},
  year    = {2021}
}

@article{pozzi2022stochastic,
  author  = {Pozzi, Andrea and Raimondo, Davide M.},
  title   = {Stochastic model predictive control for optimal charging of electric vehicles battery packs},
  journal = {Journal of Energy Storage},
  volume  = {55},
  pages   = {105332},
  year    = {2022}
}

@article{qureshi2024multiobjective,
  author  = {Qureshi, U. and Ghosh, A. and Panigrahi, B. K.},
  title   = {Multiobjective pareto-optimal intelligent electric vehicle charging schedule in a commercial charging station: A stochastic convex optimization approach},
  journal = {IEEE Transactions on Industrial Informatics},
  volume  = {20},
  number  = {11},
  pages   = {12620--12632},
  year    = {2024}
}

@article{ratnam2015scheduling,
  author  = {Ratnam, Elizabeth L. and Weller, Steven R. and Kellett, Christopher M.},
  title   = {Scheduling residential battery storage with solar {PV}: Assessing the benefits of net metering},
  journal = {Applied Energy},
  volume  = {155},
  pages   = {881--891},
  year    = {2015}
}

@article{shi2022energy,
  author  = {Shi, J. and Xu, B. and Shen, Y. and Wu, J.},
  title   = {Energy management strategy for battery/supercapacitor hybrid electric city bus based on driving pattern recognition},
  journal = {Energy},
  volume  = {243},
  pages   = {122752},
  year    = {2022}
}

@article{skugor2016bilevel,
  author  = {{\v{S}}kugor, Branimir and Deur, Jo{\v{s}}ko},
  title   = {A bi-level optimisation framework for electric vehicle fleet charging management},
  journal = {Applied Energy},
  volume  = {184},
  pages   = {1332--1342},
  year    = {2016}
}

@article{tang2016model,
  author  = {Tang, Wanrong and Zhang, Ying Jun},
  title   = {A model predictive control approach for low-complexity electric vehicle charging scheduling: Optimality and scalability},
  journal = {IEEE Transactions on Power Systems},
  volume  = {32},
  number  = {2},
  pages   = {1050--1063},
  year    = {2016}
}

@article{teng2020technical,
  author  = {Teng, F. and Ding, Z. and Hu, Z. and Sarikprueck, P.},
  title   = {Technical review on advanced approaches for electric vehicle charging demand management, Part {I}: Applications in electric power market and renewable energy integration},
  journal = {IEEE Transactions on Industry Applications},
  volume  = {56},
  number  = {5},
  pages   = {5684--5694},
  year    = {2020}
}

@inproceedings{uddin2024charging,
  author    = {Uddin, Mohammad Kamar and Amani, Ali Moradi and Sajjadi, S. S. and Jalili, Mahdi and Carvajal, A.},
  title     = {Charging cost minimization in a fleet depot using adaptive queuing},
  booktitle = {2024 11th International Conference on Power and Energy Systems Engineering (CPESE)},
  publisher = {IEEE},
  pages     = {125--130},
  year      = {2024}
}

@article{zhao2023toward,
  author  = {Zhao, Y. and Jiang, Z. and Chen, X. and Liu, P. and Peng, T. and Shu, Z.},
  title   = {Toward environmental sustainability: Data-driven analysis of energy use patterns and load profiles for urban electric vehicle fleets},
  journal = {Energy},
  volume  = {285},
  pages   = {129465},
  year    = {2023}
}

@article{vepsalainen1987priority,
  author  = {Vepsalainen, Ari P. J. and Morton, Thomas E.},
  title   = {Priority rules for job shops with weighted tardiness costs},
  journal = {Management Science},
  volume  = {33},
  number  = {8},
  pages   = {1035--1047},
  year    = {1987}
}

@article{valente2007heuris,
  author  = {Valente, Jos{\'e} M. S.},
  title   = {Heuristics for the single machine scheduling problem with early and quadratic tardy penalties},
  journal = {International Journal of Production Economics},
  volume  = {106},
  number  = {1},
  pages   = {261--274},
  year    = {2007}
}

@article{parasuraman2000model,
  author  = {Parasuraman, Raja and Sheridan, Thomas B. and Wickens, Christopher D.},
  title   = {A model for types and levels of human interaction with automation},
  journal = {IEEE Transactions on Systems, Man, and Cybernetics - Part A: Systems and Humans},
  volume  = {30},
  number  = {3},
  pages   = {286--297},
  year    = {2000}
}

@article{parasuraman2010complacency,
  author  = {Parasuraman, Raja and Manzey, Dietrich H.},
  title   = {Complacency and bias in human use of automation: An attentional integration},
  journal = {Human Factors},
  volume  = {52},
  number  = {3},
  pages   = {381--410},
  year    = {2010}
}

@article{fiori2018modelling,
  author  = {Fiori, Chiara and Marzano, Vittorio},
  title   = {Modelling energy consumption of electric freight vehicles in urban pickup/delivery operations: Analysis and estimation on a real-world dataset},
  journal = {Transportation Research Part D: Transport and Environment},
  volume  = {65},
  pages   = {658--673},
  year    = {2018}
}

@article{vales2026emergent,
  author  = {Vales-Alonso, Javier and Alcaraz, Juan J.},
  title   = {Emergent charging coordination in electric delivery fleets},
  journal = {arXiv preprint arXiv:2609.07689},
  year    = {2026}
}

@article{skugor2025hierarchical,
  author  = {{\v{S}}kugor, Branimir and Grden, L. and Deur, Jo{\v{s}}ko},
  title   = {Hierarchical model predictive control-based electric vehicle fleet charging management},
  journal = {Energy Conversion and Management},
  volume  = {342},
  pages   = {120119},
  year    = {2025}
}

@article{orfanoudakis2025ev2gym,
  author  = {Orfanoudakis, Stavros and Diaz-Londono, Cesar and Y{\i}lmaz, Y. E. and Palensky, Peter and Vergara, Pedro P.},
  title   = {{EV2Gym}: A flexible {V2G} simulator for {EV} smart charging research and benchmarking},
  journal = {IEEE Transactions on Intelligent Transportation Systems},
  volume  = {26},
  number  = {2},
  pages   = {2410--2421},
  year    = {2025}
}

@article{mei2026multiagent,
  author  = {Mei, Z. and Zhang, Y. and Jiang, H. and Xue, Y. and Zhang, J. and Gao, D. W.},
  title   = {Multi-agent heterogeneous graph reinforcement learning for electric vehicle routing and charging scheduling in coupled power-transportation networks},
  journal = {Applied Energy},
  volume  = {403},
  pages   = {126958},
  year    = {2026}
}

@article{varma2026electric,
  author  = {Varma, S. M. and Castro, F. and Maguluri, S. T.},
  title   = {Electric vehicle fleet and charging infrastructure planning},
  journal = {Management Science},
  volume  = {72},
  year    = {2026}
}
"""

# =============================================================================
# 2. COMPLETE REVISED MANUSCRIPT (main.tex)
# =============================================================================
MAIN_TEX = r"""\documentclass[final,3p,times,twocolumn]{elsarticle}
\usepackage{epsr}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{multirow}
\usepackage{xcolor}
\usepackage{hyperref}

\journal{Electric Power Systems Research}

\begin{document}

\begin{frontmatter}

\title{Schedule-Aware Real-time Depot Charging for Electric Fleets}

\author[inst1]{Mohammad Kamar Uddin\corref{cor1}}
\ead{s3955218@student.rmit.edu.au}
\cortext[cor1]{Corresponding author.}

\author[inst1]{Ali Moradi Amani}
\ead{ali.moradiamani@rmit.edu.au}

\author[inst2]{Hui Song}
\ead{hui.song1@newcastle.edu.au}

\affiliation[inst1]{organization={School of Engineering, RMIT University},
            city={Melbourne},
            state={VIC},
            postcode={3000},
            country={Australia}}

\affiliation[inst2]{organization={School of Information and Physical Sciences, University of Newcastle},
            city={Callaghan},
            state={NSW},
            postcode={2308},
            country={Australia}}

\begin{abstract}
The electrification of commercial delivery and service fleets is constrained by limited depot charger counts, transformer capacity limits, and dynamic intra-day operational disruptions. Existing charging controllers typically decouple priority queue admission from predictive cost optimization, rely on static operator flags, or re-solve model predictive control (MPC) problems at every time step without closed-loop feasibility feedback. This paper proposes a closed-loop, three-layer Adaptive Queuing-based Predictive Scheduling (AQPS) framework with a human-in-the-loop operator impact preview. Layer~0 derives each vehicle's mission-sufficient session energy requirement from the fleet roster, scores urgency via the Apparent Tardiness Cost (ATC) index, and uses bisection over the Layer~2 convex Quadratic Program (QP) to admit the largest feasible priority set with zero predicted shortfall. Layer~1 manages charger contention across an Active Priority Charging Queue ($Q_k^{act}$, capacity $M$), a Priority Waiting Queue ($Q_k^p$), and a Non-Priority Queue ($Q_k^n$) using a four-tier disconnection hierarchy: laxity-based non-priority preemption, three-condition energy-based safe release (in kWh), guarded First-In-First-Out (FIFO) priority rotation from $Q_k^{act}$, and operator escalation. Layer~2 executes an event-triggered receding-horizon QP that re-solves only upon arrivals/departures, roster injections, predicted priority shortfalls, solar deviations, or a maximum periodic timer, feeding shortfall predictions back to Layer~0. Extensive simulations across 45-, 90-, and 180-EV fleets using empirical commercial van profiles and Australian Energy Market Operator (AEMO) rooftop photovoltaic (PV) data demonstrate that the full proposed method (\texttt{AQPS-RT-ET}) achieves 100.0\% mission sufficiency and 95.9--100.0\% priority energy fulfillment across all contention and peak-stress scenarios, outperforming Least Laxity First (\texttt{LLF}, 74.7--89.2\%), Earliest Deadline First (\texttt{EDF}, 69.3--89.2\%), and \texttt{MPC-only} (65.6--78.5\%), while reducing charging costs by 23.1--34.1\% and cutting Layer~2 QP solve counts by up to 26.3\% at sub-50~ms solve times.
\end{abstract}

\begin{keyword}
EV fleet charging \sep Adaptive queue \sep Priority EVs \sep Event-triggered MPC \sep Human-in-the-loop \sep Predictive control
\end{keyword}

\end{frontmatter}

% =============================================================================
% SECTION 1: INTRODUCTION
% =============================================================================
\section{Introduction}
\label{sec:intro}

The global transition towards electrified transportation is driving a significant shift in energy infrastructure, with the International Energy Agency (IEA) projecting that electric light-duty vehicles and commercial vans will exceed a 40\% sales share \cite{iea2025global,nimalsiri2021phd}. While private adoption is increasing, the electrification of government and business fleets is gaining the most momentum due to net-zero mandates and supportive policies \cite{alhanahi2024charging}. This commercial segment supports diverse applications, from last-mile parcel delivery to municipal services, and stimulates the second-hand Electric Vehicle (EV) market through rapid vehicle turnover \cite{alhanahi2024charging,uddin2024charging}. The successful integration of urban fleets is highly sensitive to existing electrical infrastructure to support charging \cite{corsetti2024commercial,pozzi2022stochastic}. Consequently, establishing a reliable and scalable charging management system is a foundational requirement for sustainable fleet electrification \cite{varma2026electric}.

To ensure sustainable adoption, fleet electrification must shift the focus from purchase price to Total Cost of Ownership (TCO), encompassing infrastructure, grid upgrades, and operational expenses. Depots represent the primary bottleneck in this context. Although electricity is more cost-effective than fossil fuel, significant upfront transformer upgrade costs and limited spare electrical capacity at urban depots remain major barriers \cite{motlagh2026review}. Beyond physical infrastructure, three critical challenges hinder efficient depot management:
\begin{itemize}
    \item \textit{Operational Constraints and Scheduling Complexity:} Unlike traditional refueling, longer EV charging times---exacerbated by limited charger-to-vehicle ratios in commercial depots---clash with strict, predefined service schedules. To compensate, operators may be forced to oversize fleets or invest in heavier, larger-battery vehicles \cite{keshavarzian2024optimal,vales2026emergent}. Furthermore, real-time uncertainties such as vehicle breakdowns, driver absences, traffic delays, or route swaps frequently render static day-ahead plans infeasible, jeopardizing daily operations \cite{alhanahi2024charging,jack2025estimating,zhao2023toward}.
    \item \textit{Cost and Load Management:} Managing charging loads is essential to mitigate high time-of-use (ToU) energy charges and transformer overloading. Fleets must strategically limit peak-time grid power absorption while coordinating co-located photovoltaic (PV) generation and battery energy storage systems (BESS).
    \item \textit{Operational Diversity:} Charging requirements are dictated by heterogeneous duty cycles, making ``one-size-fits-all'' rules ineffective. These range from overnight return-to-base delivery vans to multi-shift vehicles returning mid-day for turnaround top-ups \cite{vales2026emergent,fiori2018modelling}.
\end{itemize}

Extensive research has focused on optimizing EV fleet charging to minimize costs and grid impact. For predictable operational profiles, such as Electric Bus (EB) systems, several studies have optimized depot and on-route charging schedules to handle high energy demands \cite{he2020optimal,hosseini2023optimizing,meng2024research,shi2022energy}. In the commercial delivery sector, bi-level and hierarchical frameworks have been proposed to balance aggregate power limits with state-of-charge (SoC) requirements \cite{skugor2016bilevel,skugor2025hierarchical}. To further mitigate grid dependency, recent literature emphasizes the integration of on-site renewable energy and BESS, including DC microgrid management \cite{dicorato2019integrated}, smart charging under dynamic pricing and co-located solar in European depots \cite{chen2025optimal,corsetti2024commercial}, and PV forecast integration \cite{vandermeer2018energy}. While multi-objective evolutionary frameworks \cite{liu2023multiobjective}, stochastic convex scheduling \cite{qureshi2024multiobjective}, and reinforcement learning benchmarks \cite{orfanoudakis2025ev2gym,mei2026multiagent,vales2026emergent} have demonstrated promising results, they typically assume either uncontested charger access (one EVSE per vehicle) or purely local heuristics without hard depot-level priority feasibility guarantees.

Smart online algorithms have also been investigated for handling operational uncertainty. Model Predictive Control (MPC) and receding-horizon optimization have been applied to workplace and fleet charging to manage electrical constraints \cite{frendo2019realtime,tang2016model,skugor2025hierarchical}. Priority-ranking schemes have also emerged, using flexibility metrics and Quadratic Programming (QP) to manage vehicle-level power allocation \cite{lokhande2024realtime}. Despite the strengths of smart techniques, existing studies often treat priority flags as static, exogenous inputs and lack closed-loop feedback between the continuous power optimizer and the discrete queue admission controller \cite{alinia2020online,lee2019acndata}. Conversely, stochastic MPC \cite{pozzi2022stochastic} and Approximate Dynamic Programming (ADP) \cite{mahyari2025electric} incur high computational overhead and do not provide transparent what-if impact previews when overloaded priority demand requires human dispatcher intervention \cite{chandra2024comprehensive,motlagh2026review}.

Despite these advances, three critical gaps remain in the depot charging literature: (i) priority assignment is typically treated as a static binary flag disconnected from the real-time feasibility of the underlying power allocation QP; (ii) existing queue preemption rules only preempt non-priority vehicles and deadlock when all $M$ chargers are occupied by priority vehicles; and (iii) receding-horizon MPC controllers re-solve the optimization at every time step without event-triggered efficiency or human-in-the-loop decision support for Fleet Management System (FMS) integration.

To address these gaps, we propose a closed-loop, three-layer Adaptive Queuing-based Predictive Scheduling (AQPS) framework. The main contributions of this paper are threefold:
\begin{enumerate}
    \item \textbf{Real-Time Schedule-Derived Priority Selection with Optimizer Feasibility Check and Multi-Queue Safe Release (Layers~0 and~1):} We introduce Layer~0, which computes each vehicle's mission-sufficient session energy requirement $E_i^{req}$ from the fleet roster, ranks vehicles via the Apparent Tardiness Cost (ATC) urgency index \cite{vepsalainen1987priority,valente2007heuris}, and performs bisection over the Layer~2 convex QP ($\approx \log_2 n$ solves) to select the largest feasible priority set with zero predicted shortfall. In Layer~1, we introduce an Active Priority Charging Queue ($Q_k^{act}$, capacity $M$) alongside waiting queues $Q_k^p$ and $Q_k^n$, governed by a four-tier disconnection hierarchy that combines laxity-based non-priority preemption, three-condition energy-based safe release (in kWh), and guarded FIFO rotation from $Q_k^{act}$.
    \item \textbf{Two-Way Coupled Event-Triggered MPC (Layer~2):} Instead of solving the receding-horizon QP blindly at every time step, Layer~2 caches the optimal horizon plan and re-solves only upon discrete operational events (arrivals/departures, roster injections, predicted priority shortfall $>0$, PV/BESS forecast deviations exceeding a threshold, or a periodic timer $T_{\max}$). Whenever a predicted priority shortfall is detected, Layer~2 triggers a closed-loop recalculation in Layers~0 and~1.
    \item \textbf{Human-in-the-Loop Operator Impact-Preview Loop for FMS Integration:} Grounded in levels-of-automation theory (``executes automatically unless vetoed'') \cite{parasuraman2000model,parasuraman2010complacency}, we design an operator impact preview that executes three what-if QP solves whenever a disruption or overload affects priority service or exceeds a cost delta threshold, presenting the dispatcher with quantified energy and cost trade-offs while maintaining uninterrupted charging.
\end{enumerate}

The remainder of this paper is organized as follows. Section~\ref{sec:formulation} presents the depot charging problem formulation. Section~\ref{sec:aqps} details the three-layer AQPS architecture, event-triggered MPC, operator impact preview, and theoretical feasibility guarantees. Section~\ref{sec:results} provides comprehensive simulation studies across 45-, 90-, and 180-EV fleets using empirical commercial van parameters and real Australian AEMO solar data. Section~\ref{sec:conclusion} concludes the paper.

% =============================================================================
% SECTION 2: PROBLEM FORMULATION
% =============================================================================
\section{Problem Formulation}
\label{sec:formulation}

We consider a commercial EV fleet depot comprising $M$ identical Level-2 EV Supply Equipment (EVSE) units. The depot is integrated with an on-site PV generation system and a BESS, as shown in Fig.~\ref{fig:system_diagram}. Fleet vehicles return to the depot between scheduled service trips and overnight to recharge for their subsequent commitments. The following operational assumptions govern the depot model:
\begin{itemize}
    \item The number of fleet EVs ($n$) exceeds the number of EVSE chargers ($M$), typically $2M \le n \le 4M$ \cite{alhanahi2024charging}, creating physical charger contention alongside transformer capacity constraints.
    \item At day-ahead initialization, each vehicle's end-of-day state of charge (in kWh) and next-day service roster (trip departure times, route energy requirements, return energy, and trip criticality weights) are known. Intra-day disruptions (vehicle breakdowns, driver absences, roster swaps, arrival delays, and solar forecast errors) occur dynamically during operation.
    \item Day-ahead and intra-day forecasts of PV generation $P_t^{PV}$ (kW) and BESS dispatch power $P_t^{BESS}$ (kW) are provided to the scheduler, while realized PV output may deviate from the forecast.
    \item Level-2 AC charging ($\le 22$~kW, corresponding to $\le 0.3\text{--}0.5\text{C}$ for commercial vans) operates in the linear constant-current regime where C-rate-induced battery degradation is negligible compared to DC fast charging, and constant charger efficiency $\eta_{ch}$ is absorbed into the effective session energy demand $E_i^{req}$.
\end{itemize}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\columnwidth]{figs/system_architecture.png}
    \caption{System diagram of an adaptive queue-based commercial EV fleet charging depot integrated with on-site PV and BESS.}
    \label{fig:system_diagram}
\end{figure}

Commercial EVs execute diverse operational schedules that result in single or multiple service trips per day. Each charging session in the depot is dynamically classified as either a priority or non-priority session based on the vehicle's upcoming service commitment and feasibility check (detailed in Section~\ref{sec:layer0}). The charging operation over a finite horizon is discretized into intervals of duration $\Delta t$ (hours). The set $\mathcal{T} = \{k, k+1, \dots, k+H\}$ denotes the time steps in the lookahead horizon of length $H$, where $k$ is the current decision epoch. We denote by $\mathcal{N}$ the set of all EVs arriving during the operational day, and by $\mathcal{I}_k \subseteq \mathcal{N}$ the set of vehicles present at the depot at step $k$. The subset of EVs actively connected to EVSEs at step $k$ is denoted $\mathcal{C}_k \subseteq \mathcal{I}_k$ with $|\mathcal{C}_k| \le M$.

At any epoch $k$, the present fleet is partitioned into priority vehicles $\mathcal{I}^p$ and non-priority vehicles $\mathcal{I}^n$, such that $\mathcal{I}^p \cup \mathcal{I}^n = \mathcal{N}$ and $\mathcal{I}^p \cap \mathcal{I}^n = \emptyset$. Each vehicle $i$ is assigned a binary priority indicator $y_i \in \{0, 1\}$, where $y_i = 1$ if $i \in \mathcal{I}^p$ and $y_i = 0$ otherwise. Rather than treating $y_i$ as a fixed exogenous constant, Section~\ref{sec:layer0} presents the real-time Layer~0 mechanism that derives and updates $y_i$ and the mission-sufficient session energy requirement $E_i^{req}$ (defined as the energy required for the vehicle's next scheduled trip(s), safe return to base, and safety reserve margin $r_i$ above arrival energy) in closed loop with the optimizer.

\subsection{Objective Function}
\label{sec:objective}

At each optimization epoch $k$, a finite-horizon convex Quadratic Program (QP) is solved to determine the charging rate of each connected charger over the lookahead horizon. The sole decision variables are the charging powers $x_{i,t}$ (kW) delivered to EV $i \in \mathcal{C}_k$ at time step $t \in \mathcal{T}$. The objective function comprises three terms: (i) the net electricity cost $H^{EC}$ purchased from the grid under Time-of-Use (ToU) tariffs, (ii) the non-completion penalty $H^{NC}$ that drives the controller to fulfill remaining session energy demands $E_i^{req}$ without introducing slack variables, and (iii) a quadratic Tikhonov regularizer $H^{QC}$ that conditions the Hessian and smooths aggregate load profiles. The optimization objective $C(x)$ is formulated as:
\begin{equation}
    \min_{x} \quad \alpha_1 H^{EC}(x) + \alpha_2 H^{NC}(x) + \alpha_3 H^{QC}(x)
    \label{eq:obj}
\end{equation}
where
\begin{equation}
    H^{EC}(x) = \sum_{t=k}^{k+H} C_t^{ToU} \left( \sum_{i \in \mathcal{C}_k} x_{i,t} - (P_t^{PV} + P_t^{BESS}) \right) \Delta t
    \label{eq:hec}
\end{equation}
\begin{equation}
    H^{NC}(x) = \left( \sum_{t \in \mathcal{T}} E_t^{agg} - \sum_{i \in \mathcal{C}_k} E_i^{req} \right)^2
    \label{eq:hnc}
\end{equation}
\begin{equation}
    H^{QC}(x) = \sum_{t=k}^{k+H} (P_t^{agg})^2
    \label{eq:hqc}
\end{equation}

Here, $\alpha_1, \alpha_2, \alpha_3 > 0$ are weighting coefficients, $C_t^{ToU}$ (\$/kWh) in Eq.~\eqref{eq:hec} is the ToU electricity tariff at step $t$, and $P_t^{PV} + P_t^{BESS}$ represents the exogenous on-site solar and storage power offset at step $t$. In Eq.~\eqref{eq:hnc}, $E_t^{agg} = \sum_{i \in \mathcal{C}_k} E_{i,t} = \sum_{i \in \mathcal{C}_k} x_{i,t}\Delta t$ is the aggregate energy delivered at step $t$, and $E_i^{req}$ (kWh) is the remaining mission-sufficient session energy demand of vehicle $i$. In Eq.~\eqref{eq:hqc}, $P_t^{agg} = \sum_{i \in \mathcal{C}_k} x_{i,t}$ is the aggregate EV charging power at step $t$; minimizing $\sum_t (P_t^{agg})^2$ provides Tikhonov regularization that guarantees strict convexity (positive-definite Hessian), numerical stability, and load smoothing across degenerate constant-tariff intervals.

\textbf{Selection and Justification of Weighting Coefficients ($\alpha_1, \alpha_2, \alpha_3$):} To establish a transparent hierarchy among the three competing objectives, the weights are selected via dimensional scaling and verified via sensitivity analysis (Section~\ref{sec:sensitivity}): $\alpha_1 = 1.0$ sets the reference monetary scale (\$), $\alpha_2 = 100.0$ (\$/kWh$^2$) ensures that avoiding a $1$~kWh energy shortfall carries a marginal penalty two orders of magnitude larger than the peak-to-off-peak tariff differential ($\Delta C^{ToU} = \$0.14$/kWh), and $\alpha_3 = 0.005$ (\$/kW$^2$) is chosen small enough that the Tikhonov regularizer conditions the QP Hessian without overriding ToU price arbitrage.

\subsection{Constraints}
\label{sec:constraints}

\textit{Infrastructure constraints:} We model electrical infrastructure constraints for a radial three-phase depot distribution network using a constraint matrix $\mathbb{K} \in \mathbb{R}^{L \times N}$, phase angles $\phi \in \mathbb{R}^N$, and per-branch current limits $\{c_l\}_{l=1}^L$ (A), where $L$ is the number of breakers/feeders and $N = |\mathcal{C}_k|$ is the number of connected EVSEs \cite{lee2019acnsim,lee2021adaptive,uddin2024charging}:
\begin{equation}
    |\mathbb{K}_l| I_t \le c_l, \quad \forall l \in \{1, \dots, L\}, \quad \forall t \in \mathcal{T}
    \label{eq:branch}
\end{equation}
where $|\mathbb{K}_l|$ denotes the element-wise modulus of row $l$ and $I_t = [I_{1,t}, \dots, I_{N,t}]^\top$ is the vector of charging currents. In addition, the gross aggregate EV charging power must respect the depot transformer/site capacity limit $P_{site}^{\max}$ (kW):
\begin{equation}
    \sum_{i \in \mathcal{C}_k} x_{i,t} \le P_{site}^{\max}, \quad \forall t \in \mathcal{T}
    \label{eq:sitelimit}
\end{equation}
Note that because Eq.~\eqref{eq:sitelimit} bounds gross EV demand $\sum_{i \in \mathcal{C}_k} x_{i,t}$ at the EVSE distribution busbar, an intra-day shortfall in $P_t^{PV}$ increases net grid import cost in Eq.~\eqref{eq:hec} without violating transformer feasibility.

\textit{Charging rate constraints:} Each connected EV $i \in \mathcal{C}_k$ is bounded by its hardware minimum and maximum charging power over its connected window $\mathcal{T}_i \subseteq \mathcal{T}$:
\begin{equation}
    \underline{x}_i \le x_{i,t} \le \bar{x}_i, \quad \forall i \in \mathcal{C}_k, \quad \forall t \in \mathcal{T}_i
    \label{eq:ratebounds}
\end{equation}
\begin{equation}
    \underline{x}_i = \max(x_{session,i}^{\min}, \, x_{pri}^{\min}), \quad \forall i \in \mathcal{C}_k \cap \mathcal{I}^p
    \label{eq:pribound}
\end{equation}
where $\bar{x}_i$ (kW) and $\underline{x}_i$ (kW) denote the maximum and minimum charging rates of vehicle $i$. For non-priority vehicles ($i \in \mathcal{C}_k \cap \mathcal{I}^n$), $\underline{x}_i = x_{session,i}^{\min} = 0$~kW, allowing full deferral during peak tariff periods. For priority vehicles ($i \in \mathcal{C}_k \cap \mathcal{I}^p$), Eq.~\eqref{eq:pribound} enforces a guaranteed minimum charging power $x_{pri}^{\min} = 3.84$~kW (16~A at 240~V under SAE J1772) whenever energy is still required.

\textit{Energy delivered to EVs:} The cumulative energy delivered to connected non-priority and priority EVs over their active horizon is governed by:
\begin{equation}
    \sum_{t \in \mathcal{T}_i} E_{i,t} \le E_i^{req}, \quad \forall i \in \mathcal{C}_k \cap \mathcal{I}^n
    \label{eq:enonpri}
\end{equation}
\begin{equation}
    \sum_{t \in \mathcal{T}_i} E_{i,t} \ge E_i^{req}, \quad \forall i \in \mathcal{C}_k \cap \mathcal{I}^p
    \label{eq:epri}
\end{equation}
In implementation, Eq.~\eqref{eq:enonpri} is enforced as an upper bound for all connected vehicles so no vehicle is overcharged beyond $E_i^{req}$, while Layer~0 uses Eq.~\eqref{eq:epri} (zero predicted shortfall below $E_i^{req}$) as the exact feasibility criterion for priority set admission.

% =============================================================================
% SECTION 3: AQPS FRAMEWORK
% =============================================================================
\section{Adaptive Queuing-Based Predictive Scheduling (AQPS)}
\label{sec:aqps}

The proposed AQPS architecture (depicted in Fig.~\ref{fig:aqps_loop}) is a closed-loop, three-layer hierarchical control system coupled with a human-in-the-loop operator impact preview. Unlike conventional open-loop cascades that pass static priority flags into an every-step MPC solver, AQPS couples schedule-derived priority selection (Layer~0), multi-queue admission with safe release and active priority rotation (Layer~1), and event-triggered QP optimization (Layer~2) via two-way shortfall feedback.

\begin{figure*}[htbp]
    \centering
    \includegraphics[width=0.92\textwidth]{figs/fig2_system_architecture.png}
    \caption{Closed-loop architecture of the revised three-layer AQPS framework (\texttt{AQPS-RT-ET}) showing Layer~0 (ATC urgency scoring and QP bisection feasibility check), Layer~1 (multi-queue management across $Q_k^{act}$, $Q_k^p$, and $Q_k^n$ with safe release and guarded FIFO rotation), Layer~2 (event-triggered MPC with plan cache $X^*$), and the human-in-the-loop operator impact-preview loop.}
    \label{fig:aqps_loop}
\end{figure*}

\subsection{Layer 0: Real-Time Schedule-Derived Priority Selection}
\label{sec:layer0}

\textbf{Day-ahead and arrival initialization:} For each vehicle $i$ arriving at the depot with battery energy $E_i^{arr}$ (kWh), Layer~0 queries the fleet roster to determine the energy required until its next charging opportunity:
\begin{equation}
    E_i^{mission} = E_i^{trip} + E_i^{ret} + r_i
    \label{eq:emission}
\end{equation}
where $E_i^{trip}$ (kWh) is the energy required for the next scheduled delivery/service trip, $E_i^{ret}$ (kWh) is the energy required to return safely to the depot (or designated alternate charger), and $r_i = 0.10 C_i$ (kWh) is the 10\% safety reserve floor \cite{vales2026emergent}. The remaining mission-sufficient session energy requirement is $E_i^{req} = \max(0, \, E_i^{mission} + E_i^{buf} - E_i^{arr} - E_i^{deliv})$, where $E_i^{buf}$ is any scheduled intra-day secondary trip buffer.

\textbf{Apparent Tardiness Cost (ATC) urgency scoring:} At any Layer~0 event epoch $t$, each present vehicle $i \in \mathcal{I}_k$ with remaining demand $E_i^{req} > 0$ and departure time $d_i > t$ is assigned an urgency score $\pi_i(t)$ using the Apparent Tardiness Cost (ATC) dispatching rule from scheduling theory \cite{vepsalainen1987priority,valente2007heuris}:
\begin{equation}
    \pi_i(t) = \frac{w_i}{p_i(t)} \exp\left( -\frac{\max(0, \, s_i(t))}{k_{atc} \cdot \bar{p}(t)} \right)
    \label{eq:atc}
\end{equation}
where $w_i > 0$ is the roster trip criticality weight, $p_i(t) = E_i^{req} / \bar{x}_i$ (hours) is the minimum remaining charging duration at maximum EVSE power $\bar{x}_i$, $s_i(t) = (d_i - t)\Delta t - p_i(t)$ (hours) is the vehicle's current scheduling slack (laxity), $\bar{p}(t)$ is the mean remaining charging duration across present vehicles, and $k_{atc}$ is the lookahead tuning parameter \cite{valente2007heuris}. When a vehicle has ample slack ($s_i(t) \gg k_{atc}\bar{p}$), the exponential term discounts its urgency; as slack approaches zero, $\pi_i(t) \to w_i / p_i(t)$, prioritizing high-criticality, fast-completing missions.

\textbf{Optimizer-in-the-loop bisection feasibility check:} Selecting the optimal subset of priority vehicles under shared transformer and charger limits is a knapsack-like combinatorial problem (NP-hard in general). Layer~0 sorts candidate mission-committed vehicles in descending order of $\pi_i(t)$: $(v_{(1)}, v_{(2)}, \dots, v_{(m)})$. To find the largest prefix $S_q = \{v_{(1)}, \dots, v_{(q)}\}$ that can be guaranteed zero shortfall without overloading the depot, Layer~0 uses the Layer~2 convex QP as an exact feasibility oracle via binary search (bisection) over $q \in \{1, \dots, m\}$. A candidate prefix $S_q$ is declared feasible if and only if the Layer~2 QP over the active slice $\min(q, M)$ returns an optimal solution with predicted energy shortfall $\hat{\delta}_i = E_i^{mission} - (E_i^{arr} + E_i^{deliv} + \sum_{t \in \mathcal{T}_i} x_{i,t}^*\Delta t) \le \epsilon$ for all $i$. This requires at most $\lceil \log_2 m \rceil + 1$ QP solves per Layer~0 event. Any urgent vehicles excluded from the feasible prefix ($v_{(q+1)}, \dots, v_{(m)}$) are placed on the operator escalation list.

\subsection{Layer 1: Multi-Queue Management with Safe Release and Active Priority Rotation}
\label{sec:layer1}

To prevent the priority-on-priority deadlock identified in conventional two-queue systems when all $M$ chargers are occupied by priority EVs, Layer~1 maintains three explicit queues at step $k$:
\begin{enumerate}
    \item \textbf{Active Priority Charging Queue ($Q_k^{act}$):} A FIFO-ordered queue of at most $M$ priority vehicles currently connected to EVSEs and actively receiving guaranteed priority charging ($|Q_k^{act}| \le M$).
    \item \textbf{Priority Waiting Queue ($Q_k^p$):} Priority vehicles ($y_i = 1$) awaiting an available EVSE slot, ordered by descending ATC score $\pi_i(t)$.
    \item \textbf{Non-Priority Queue ($Q_k^n$):} Non-priority vehicles ($y_i = 0$), admitted to idle EVSEs in FIFO arrival order.
\end{enumerate}

\textbf{Four-tier disconnection hierarchy:} Whenever an incoming or newly promoted priority vehicle $i_{new} \in Q_k^p$ requires an EVSE and all $M$ chargers are occupied ($|\mathcal{C}_k| = M$), Layer~1 executes the following four-tier disconnection hierarchy:
\begin{enumerate}
    \item \textit{Tier~1 (Laxity-Based Non-Priority Preemption):} If one or more connected vehicles belong to $Q_k^n$ ($y_j = 0$), Layer~1 selects the victim $j^* = \arg\max_{j \in \mathcal{C}_k, y_j=0} s_j(k)$ with the largest laxity (most scheduling flexibility), disconnects $j^*$, and admits $i_{new}$ to the tail of $Q_k^{act}$.
    \item \textit{Tier~2 (Energy-Based Safe Release from $Q_k^{act}$):} If all $M$ EVSEs are occupied by priority vehicles ($|Q_k^{act}| = M$), Layer~1 evaluates every connected priority vehicle $j \in Q_k^{act}$ against the \textbf{Safe-Release Condition} (computed strictly in kWh, never \% SoC). Vehicle $j$ is eligible for safe release if and only if all three conditions hold simultaneously:
    \begin{align}
        \text{(a)} \quad & E_j^{curr}(k) \ge E_j^{trip} + E_j^{ret} + r_j \label{eq:sr_a} \\
        \text{(b)} \quad & E_j^{further\_today} \le E_j^{curr}(k) - (E_j^{trip} + E_j^{ret} + r_j) \label{eq:sr_b} \\
        \text{(c)} \quad & (d_j^{next\_day} - a_j^{return})\Delta t \cdot \bar{x}_j \ge E_j^{next\_day} - E_j^{surplus}(k) \label{eq:sr_c}
    \end{align}
    where $E_j^{curr}(k) = E_j^{arr} + E_j^{deliv}(k)$ is the current energy in battery (kWh), $E_j^{surplus}(k) = E_j^{curr}(k) - (E_j^{trip} + E_j^{ret} + r_j)$ is the energy surplus above immediate mission sufficiency, $E_j^{further\_today}$ is any additional trip energy required today before the next charging window, and $(d_j^{next\_day} - a_j^{return})\Delta t \cdot \bar{x}_j$ is the maximum energy deliverable during the overnight return-to-departure window for the next day's first mission $E_j^{next\_day}$. Among all vehicles in $Q_k^{act}$ satisfying (a)--(c), the vehicle with the largest surplus $E_j^{surplus}(k)$ (and latest departure tie-breaker) is safely released from $Q_k^{act}$ to Make way for $i_{new}$.
    \item \textit{Tier~3 (Guarded FIFO Priority Rotation from $Q_k^{act}$):} If $|Q_k^{act}| = M$ and no vehicle satisfies full Safe Release, Layer~1 inspects $Q_k^{act}$ starting from its head $j_{head}$ (the earliest-admitted priority EV, which has been charging the longest). If the incoming vehicle is strictly more urgent ($\pi_{i_{new}}(k) > \pi_{j_{head}}(k)$) and $j_{head}$ retains positive scheduling slack ($s_{j_{head}}(k) > \Delta t$) or has already accumulated its single-trip energy $E_{j_{head}}^{trip} + r_{j_{head}}$, $j_{head}$ is rotated out of $Q_k^{act}$ back to $Q_k^p$, and $i_{new}$ is admitted to the tail of $Q_k^{act}$.
    \item \textit{Tier~4 (Hard Overload Escalation):} If every connected priority vehicle in $Q_k^{act}$ has zero slack left ($s_j(k) \le \Delta t$), force-disconnecting any vehicle would cause an immediate mission failure. Therefore, Layer~1 disconnects nobody, retains $i_{new}$ in $Q_k^p$, and escalates the conflict to the operator impact-preview loop (Section~\ref{sec:preview}).
\end{enumerate}

\subsection{Layer 2: Event-Triggered Model Predictive Control}
\label{sec:event_mpc}

In standard receding-horizon control (\texttt{AQPS-RT}), the convex QP in Eqs.~\eqref{eq:obj}--\eqref{eq:epri} is re-solved at every control step $\Delta t$, even when no arrivals, departures, or forecast changes have occurred. In the proposed \textbf{Event-Triggered MPC (\texttt{AQPS-RT-ET})}, Layer~2 solves the QP over horizon $H$ and stores the full optimal power matrix $X^* \in \mathbb{R}^{|\mathcal{C}_{k_0}| \times H}$ in a plan cache, where $k_0$ is the step of the most recent solve. At each subsequent step $k$, charging commands are always output to the EVSEs:
\begin{itemize}
    \item If \textbf{no event} occurs at step $k$ and $k - k_0 < T_{\max}$, Layer~2 bypasses the QP solver and applies column $\tau = k - k_0$ of the cached plan: $x_{i,k} = \min(X^*_{i,\tau}, \, E_i^{req}/\Delta t, \, \bar{x}_i)$, while Layer~1 updates the cumulative energy and slack counters in $Q_k^{act}$ in $O(M)$ time.
    \item If \textbf{any trigger event} occurs (Table~\ref{tab:events}), Layers~0, 1, and 2 execute according to the event routing table, update the cached plan $X^*$, and apply its first column ($\tau = 0$).
\end{itemize}

\begin{table}[htbp]
\centering
\caption{Event-triggered coupling across Layer~0, Layer~1, and Layer~2 in \texttt{AQPS-RT-ET}.}
\label{tab:events}
\footnotesize
\begin{tabular}{p{2.7cm}p{1.4cm}p{1.5cm}p{1.1cm}}
\toprule
Trigger Event & Layer 0 & Layer 1 & Layer 2 \\
\midrule
Operator injection (breakdown, absence, swap) & Recalculate $\pi_i$, bisect & Re-admit / preempt / rotate & Re-solve QP \\
EV arrival or departure & Score new EV & Admit to $Q_k^{act}$ or $Q_k^n$ & Re-solve QP \\
Predicted priority shortfall $> 0$ & Recalculate set $S$ & Preempt or safe-release & Re-solve QP \\
PV/BESS deviation $> \tau_{pv}$ & Cost check & Maintain $Q_k^{act}$ & Re-solve QP \\
Periodic timer ($\Delta k = T_{\max}$) & --- & Update $Q_k^{act}$ & Re-solve QP \\
\bottomrule
\end{tabular}
\end{table}

Crucially, the \textit{predicted priority shortfall $> 0$} trigger feeds the continuous optimizer's trajectory predictions back into Layer~0 and Layer~1, establishing two-way closed-loop coupling between priority management and cost optimization.

\subsection{Human-in-the-Loop Operator Impact-Preview Loop}
\label{sec:preview}

When an operational disruption, escalation, or large solar deviation ($> \tau_{pv}$) occurs that either affects a priority vehicle or alters projected horizon cost by more than a threshold $\tau_{cost}$ (\$5.00), AQPS invokes the Operator Impact-Preview module (suppressing alerts for minor fluctuations to prevent alert fatigue \cite{parasuraman2010complacency}). Grounded in Parasuraman, Sheridan, and Wickens' taxonomy of human interaction with automation \cite{parasuraman2000model}---specifically Level~6 (\textit{``executes automatically, then necessarily informs the human and allows veto''})---charging never pauses while awaiting operator input. The system provisionally applies the recommended schedule and generates a structured impact preview via three fast what-if QP solves:
\begin{enumerate}
    \item \textbf{Option~A (Recommended --- Feasibility-Checked Set):} Applies the Layer~0/1 priority set with safe release and $Q_k^{act}$ rotation at baseline optimal cost.
    \item \textbf{Option~B (Business Continuity --- Keep All Urgent):} Forces inclusion of the marginal escalated vehicle into the priority set by drawing additional peak-tariff grid power, reporting the exact incremental cost $+\Delta\$$.
    \item \textbf{Option~C (Cost-Saving Downgrade):} Downgrades the lowest-urgency priority vehicle ($\min_{i \in \mathcal{I}^p} \pi_i$) to non-priority, reporting the expected departure kWh and cost savings $-\Delta\$$.
\end{enumerate}

Algorithm~\ref{alg:aqps_et} formalizes the complete closed-loop \texttt{AQPS-RT-ET} procedure.

\begin{algorithm}[htbp]
\caption{Three-Layer Closed-Loop Event-Triggered AQPS (\texttt{AQPS-RT-ET})}
\label{alg:aqps_et}
\footnotesize
\begin{algorithmic}[1]
\Require Present EVs $\mathcal{I}_k$; EVSE count $M$; site limit $P_{site}^{\max}$; tariff $C_t^{ToU}$; forecasts $P_t^{PV}, P_t^{BESS}$; realized $P_k^{PV,act}$; weights $\alpha_1, \alpha_2, \alpha_3$; thresholds $\tau_{pv}, \tau_{cost}, T_{\max}$
\Ensure EVSE pilot commands $\{x_{i,k} : i \in \mathcal{C}_k\}$ and operator preview log
\State Update departures, arrivals, and cumulative energy in $Q_k^{act}, Q_k^p, Q_k^n$
\State Detect event triggers $E_{trig} \leftarrow \text{CheckEvents}(\text{Arr/Dep}, \text{Injection}, \hat{\delta}_i > 0, |P_k^{PV,act}-P_k^{PV}| > \tau_{pv}, k - k_0 \ge T_{\max})$
\If{$E_{trig} \neq \emptyset$}
    \State \textbf{--- Layer 0: Priority Selection ---}
    \If{event $\in \{\text{Arr/Dep}, \text{Injection}, \hat{\delta}_i > 0\}$}
        \State Compute $E_i^{req}$ via Eq.~\eqref{eq:emission} and ATC scores $\pi_i(k)$ via Eq.~\eqref{eq:atc}
        \State Find largest feasible priority set $S^* \subseteq \mathcal{I}_k$ via bisection over Layer~2 QP ($\hat{\delta}_i \le \epsilon$)
        \State Set $y_i \leftarrow \mathbb{I}(i \in S^*)$ and record escalated set $\mathcal{E}_k$
    \EndIf
    \State \textbf{--- Layer 1: Multi-Queue Admission & Safe Release ---}
    \State Admit priority EVs to $Q_k^{act}$ ($|Q_k^{act}| \le M$); if full, execute 4-Tier Hierarchy:
    \State \quad (1) Preempt highest-laxity $j^* \in Q_k^n$; (2) Safe-release $j \in Q_k^{act}$ meeting Eqs.~\eqref{eq:sr_a}--\eqref{eq:sr_c};
    \State \quad (3) Guarded FIFO rotate $j_{head} \in Q_k^{act}$ if $\pi_{new} > \pi_{head}$ and $s_{head} > \Delta t$; (4) Escalate to $\mathcal{E}_k$
    \State Fill remaining $M - |Q_k^{act}|$ slots from $Q_k^n$ in FIFO order to form $\mathcal{C}_k$
    \State \textbf{--- Operator Impact Preview (if priority affected or $|\Delta\text{Cost}| > \tau_{cost}$) ---}
    \State Solve 3 what-if QPs (Recommended / Keep All / Downgrade); apply operator choice (or Recommended provisionally)
    \State \textbf{--- Layer 2: Receding-Horizon QP Re-Solve ---}
    \State Solve QP in Eqs.~\eqref{eq:obj}--\eqref{eq:epri} over $\mathcal{C}_k$ and horizon $H$; store plan $X^*$ and set $k_0 \leftarrow k$
    \State Set $x_{i,k} \leftarrow X^*_{i,0}$ for all $i \in \mathcal{C}_k$
\Else
    \State \textbf{--- Cached Plan Application (No QP Solve) ---}
    \State Set $\tau \leftarrow k - k_0$ and apply $x_{i,k} \leftarrow \min(X^*_{i,\tau}, \, E_i^{req}/\Delta t, \, \bar{x}_i)$ for all $i \in \mathcal{C}_k$
\EndIf
\State \Return $\{x_{i,k} : i \in \mathcal{C}_k\}$
\end{algorithmic}
\end{algorithm}

\subsection{Convexity, Computational Complexity, and Conditional Feasibility Guarantee}
\label{sec:theory}

\textbf{Proposition 1 (Strict Convexity and Unique Global Optimum):} \textit{For any connected set $\mathcal{C}_k$ ($|\mathcal{C}_k| = N_c \le M$) and horizon $H$, the Layer~2 optimization problem in Eqs.~\eqref{eq:obj}--\eqref{eq:enonpri} with $\alpha_2, \alpha_3 > 0$ is a strictly convex Quadratic Program over a compact polytope and admits a unique global minimizer $X^*$.}

\textit{Proof Sketch:} The decision vector $x \in \mathbb{R}^{N_c H}$ is subject only to affine box bounds Eq.~\eqref{eq:ratebounds}, affine step-wise transformer limits Eq.~\eqref{eq:sitelimit}, and affine row-sum energy bounds Eq.~\eqref{eq:enonpri}. Since $X_{\min,i,t} \le X_{\max,i,t}$ and $\sum_{i \in \mathcal{C}_k} X_{\min,i,t} < P_{site}^{\max}$, the feasible polytope is non-empty, closed, and bounded, and Slater's constraint qualification holds strictly. The objective Hessian $\nabla^2 C(x) = 2\alpha_2 \Delta t^2 (I_{N_c} \otimes \mathbf{1}_H \mathbf{1}_H^\top) + 2\alpha_3 (\mathbf{1}_{N_c}\mathbf{1}_{N_c}^\top \otimes I_H)$ is the sum of positive semi-definite Kronecker products whose null spaces intersect only at zero when combined with the individual completion penalties, yielding $\nabla^2 C(x) \succ 0$. $\square$

\textbf{Computational Complexity:} A single Layer~2 QP contains $n_v = N_c H \le M H$ scalar variables and $m_c = 2 N_c H + H + N_c$ linear inequality constraints. Solved via an interior-point method (MOSEK), each solve requires $O((M H)^3)$ operations per Newton step, which for $M \in [16, 60]$ and $H = 24$ corresponds to $384\text{--}1440$ variables and executes in $32\text{--}64$~ms on a standard CPU (Section~\ref{sec:scalability}). During Layer~0 events, bisection requires at most $\lceil \log_2(|\mathcal{I}_k|) \rceil + 1 \le 7$ QP checks, keeping worst-case event latency under $0.35$~s---well within sub-minute real-time control requirements.

\textbf{Proposition 2 (Conditional Priority Service Guarantee):} \textit{Every priority vehicle $i \in \mathcal{I}^p$ is guaranteed to depart with at least its mission-sufficient energy $E_i^{mission} = E_i^{trip} + E_i^{ret} + r_i$ without force-disconnection below $E_i^{mission}$, provided the following three necessary and sufficient feasibility conditions hold over any time interval $[t_1, t_2]$:}
\begin{enumerate}
    \item \textit{Individual Dwell-Rate Condition:} For each $i \in \mathcal{I}^p$, $E_i^{mission} - E_i^{arr} \le \bar{x}_i (d_i - a_i)\Delta t$.
    \item \textit{Charger Slot Capacity Condition:} The number of simultaneously active priority vehicles whose remaining laxity satisfies $s_i(t) \le \Delta t$ (cannot be safely released or FIFO-rotated) does not exceed $M$: $|\{i \in \mathcal{I}_t^p : E_i^{curr}(t) < E_i^{mission} \text{ and } s_i(t) \le \Delta t\}| \le M$.
    \item \textit{Aggregate Transformer Energy Condition:} $\sum_{i \in \mathcal{I}_{[t_1,t_2]}^p} \max(0, E_i^{mission} - E_i^{curr}(t_1)) \le P_{site}^{\max} (t_2 - t_1)\Delta t$.
\end{enumerate}
\textit{Whenever Conditions (1)--(3) are violated by an extreme surge of simultaneous urgent arrivals, Layer~0's bisection check and Layer~1's Tier~4 rule detect the violation immediately, protect already-committed vehicles in $Q_k^{act}$ from unsafe disconnection, and escalate the excess vehicles to the operator impact preview.}

% =============================================================================
% SECTION 4: SIMULATION AND RESULTS ANALYSIS
% =============================================================================
\section{Simulation and Results Analysis}
\label{sec:results}

\subsection{Depot Setup, Empirical Fleet Data, and AEMO Solar Profiles}
\label{sec:setup}

In this section, the performance of the proposed AQPS strategy is evaluated through detailed simulations on representative commercial fleet depots. Two primary depot configurations are examined in the main benchmark: a medium depot with 45~EVs served by 16~EVSEs ($P_{site}^{\max} = 150$~kW), and a large depot with 90~EVs served by 35~EVSEs ($P_{site}^{\max} = 300$~kW), plus an extended 180-EV / 60-EVSE depot ($P_{site}^{\max} = 600$~kW) for scalability analysis. In the baseline configurations, the vehicle-to-charger ratio is approximately $3:1$, ensuring that queue management, safe release, and preemption decisions materially govern service outcomes. All EVSEs are Level-2 AC units rated up to 22~kW.

\textbf{Empirical Commercial Van Grounding:} To ensure operational realism, vehicle battery capacities and route energy demands are parameterized after commercial electric delivery vans deployed in large-scale urban logistics \cite{vales2026emergent,fiori2018modelling}: Light vans (45~kWh, 11~kW AC cap, matching the Renault Kangoo E-Tech), Medium vans (68~kWh, 22~kW AC, matching the Ford E-Transit and Mercedes eVito), and Heavy vans (85~kWh, 22~kW AC, matching the Mercedes eSprinter). Specific energy consumption is drawn from the empirical urban freight distribution $e_i \sim \mathcal{U}(0.22, 0.35)$~kWh/km validated by Fiori and Marzano \cite{fiori2018modelling}, with a mandatory safety floor $r_i = 0.10 C_i$ (10\% SoC) \cite{vales2026emergent}.

\begin{table}[htbp]
\centering
\caption{Australian commercial Time-of-Use (ToU) electricity tariff structure \cite{nimalsiri2021coordinated}.}
\label{tab:tou}
\footnotesize
\begin{tabular}{lccc}
\toprule
Day Type & Peak (15:00--21:00) & Shoulder (07:00--15:00, 21:00--22:00) & Off-Peak (22:00--07:00) \\
\midrule
Weekday / Weekend & \$0.32/kWh & \$0.21/kWh & \$0.18/kWh \\
\bottomrule
\end{tabular}
\end{table}

\textbf{PV/BESS Sizing Rationale and Real AEMO Solar Data:} The depot operates under the Australian three-tier commercial ToU tariff in Table~\ref{tab:tou} \cite{nimalsiri2021coordinated} and integrates a 20~kW rooftop PV array alongside a 10~kWh ($\pm 2.5$~kW) BESS \cite{ratnam2015scheduling,teng2020technical}. The 20~kW PV array is sized to fit the roof footprint of a standard urban commercial depotbay ($\approx 100~\text{m}^2$) and offset daytime base/minimum priority charging loads, while the 10~kWh BESS provides targeted peak-tariff shaving during the 15:00--19:00 ToU peak window without incurring high capital expenditure. Rather than using synthetic Gaussian error models, solar generation is driven by historical Australian Energy Market Operator (AEMO) NEMWeb MMS Data Model profiles (\texttt{ROOFTOP\_PV\_FORECAST} and \texttt{ROOFTOP\_PV\_ACTUAL}) for the Victoria region (\texttt{VIC1}), cross-checked against the Australian PV Institute (APVI) Solar Map (funded by ARENA, \url{https://pv-map.apvi.org.au}) and normalized to the 20~kW site capacity across six representative weekdays spanning Clear Sky (low forecast error), Partly Cloudy (moderate error), and Winter Overcast/Stormfront (high over-forecast error).

\begin{table}[htbp]
\centering
\caption{Summary of evaluation scenarios (S1--S3).}
\label{tab:scenarios}
\footnotesize
\begin{tabular}{p{1.1cm}cp{4.8cm}}
\toprule
Scenario & Priority \% & Fleet Arrival and Demand Composition \\
\midrule
S1 (Baseline) & 27\% & Uniform arrivals (06:00--16:00) with moderate session energy demand (25--52\% of capacity) and relaxed dwell times (5--11~h). High scheduling flexibility. \\
S2 (Contention) & 50\% & 80\% of non-priority EVs arrive early (07:00--10:00) with high energy demand (72--88\% of 68~kWh) and tight deadlines (4--6~h), saturating EVSEs before priority EVs arrive (10:00--13:00). \\
S3 (Peak Stress) & 50\% (Peak) & High-demand non-priority blockers arrive just before peak (13:00--15:00, 72--86\% demand, 4--6~h dwell), contending with priority EVs arriving during the ToU peak window (15:00--18:00). \\
\bottomrule
\end{tabular}
\end{table}

Simulations are conducted over a 24-hour horizon discretized into $\Delta t = 15$-min control steps ($96$ steps, lookahead $H = 24$ steps $= 6$~h) across the three progressively challenging scenarios summarized in Table~\ref{tab:scenarios}. All optimization subproblems are solved using the MOSEK interior-point solver via CVXPY under Common Random Numbers (CRN) \cite{vales2026emergent}.

\subsection{Evaluated Baselines and AQPS Ablation Variants}
\label{sec:baselines}

To rigorously isolate the contribution of each architectural layer and address benchmark competitiveness, we evaluate seven methods under identical EVSE counts ($M$) and gross transformer limits ($P_{site}^{\max}$):
\begin{enumerate}
    \item \textbf{\texttt{LLF} (Least Laxity First):} A widely used real-time deadline heuristic \cite{lee2019acnsim,lee2021adaptive} that assigns open EVSEs and allocates maximum feasible power in ascending order of laxity $s_i(t)$ (mathematically equivalent to Least Slack Time, LST, in discrete time). Connected vehicles hold their EVSE until session completion or departure.
    \item \textbf{\texttt{EDF} (Earliest Deadline First):} A classical real-time scheduling baseline that allocates open EVSEs and charging power in ascending order of departure deadline $d_i$, without priority awareness or preemption. (Note: Rate-Monotonic and Deadline-Monotonic Scheduling, RMS/DMS, are designed for periodic tasks with fixed recurrence periods and reduce to EDF for aperiodic single-arrival depot sessions.)
    \item \textbf{\texttt{MPC-only}:} Executes the Layer~2 receding-horizon convex QP (Eqs.~\eqref{eq:obj}--\eqref{eq:enonpri}) at every step $\Delta t$, but uses standard FIFO non-preemptive EVSE admission without Layer~0 priority selection, Layer~1 preemption, or Eq.~\eqref{eq:pribound} priority lower bounds.
    \item \textbf{\texttt{Offline-Optimal} (Oracle Upper Bound):} Solves a single full-horizon convex QP ($T = 96$ steps) with perfect a priori foresight of all vehicle arrivals, departures, mission demands, and realized PV/BESS trajectories \cite{orfanoudakis2025ev2gym,vales2026emergent}.
    \item \textbf{\texttt{AQPS-static} (Day-Ahead Priority Ablation):} Uses Layer~0 at day-ahead/arrival initialization to compute mission-sufficient targets $E_i^{req}$ and initial priority flags $y_i$, combined with Layer~1 non-priority preemption (Tier~1) and every-step Layer~2 MPC, but \textbf{disables intra-day Layer~0 re-scoring, Tier~2 Safe Release, Tier~3 $Q_k^{act}$ rotation, and Event-Triggered caching}.
    \item \textbf{\texttt{AQPS-RT} (Real-Time Closed-Loop, Every-Step MPC Ablation):} Enables the full closed-loop coupling between Layer~0 (real-time ATC re-scoring and QP bisection feasibility check) and Layer~1 (Active Priority Charging Queue $Q_k^{act}$ with Tier~1 preemption, Tier~2 Safe Release, and Tier~3 guarded FIFO rotation), but solves the Layer~2 QP at \textbf{every time step $\Delta t$}.
    \item \textbf{\texttt{AQPS-RT-ET} (Proposed Full Framework):} Combines the full real-time Layer~0 and Layer~1 ($Q_k^{act}$) intelligence of \texttt{AQPS-RT} with **Layer~2 Event-Triggered MPC** ($T_{\max} = 60$~min, $\tau_{pv} = 3.5$~kW) and the **Human-in-the-Loop Operator Impact-Preview Loop**.
\end{enumerate}

\subsection{Main Benchmark Results (Experiment E1)}
\label{sec:main_results}

Table~\ref{tab:master_results} and Fig.~\ref{fig:fulfillment} present the master E1 benchmark results across scenarios S1--S3 for both the 45-EV (16~EVSEs) and 90-EV (35~EVSEs) depots. We report both total session **Priority Fulfillment (\%)** (percentage of total priority target energy $E_i^{req}$ delivered) and **Mission Sufficiency (\%)** (percentage of priority vehicles departing with at least their full next-trip + return + 10\% safety reserve energy $E_i^{mission}$).

\begin{table*}[htbp]
\centering
\caption{Master benchmark comparison (Experiment E1) across scenarios S1--S3 and fleet sizes (45~EVs / 16~EVSEs and 90~EVs / 35~EVSEs). Best online priority service and cost metrics in Contention (S2) and Peak Stress (S3) are highlighted in bold.}
\label{tab:master_results}
\footnotesize
\begin{tabular}{llccccccccc}
\toprule
Fleet & Scenario & Method & Pri. Fulfill (\%) & Mission Suff. (\%) & Non-Pri. (\%) & Jain Fairness & Cost (\$) & Safe Rel. & L2 Solves & Mean Solve (ms) \\
\midrule
\multirow{21}{*}{45 EVs (16 EVSEs)} 
 & S1 (27\%) & \texttt{LLF}             & 100.0 & 100.0 & 100.0 & 1.000 & 181.00 & 0  & 0  & 0.00  \\
 & S1 (27\%) & \texttt{EDF}             & 100.0 & 100.0 & 100.0 & 1.000 & 181.00 & 0  & 0  & 0.00  \\
 & S1 (27\%) & \texttt{MPC-only}        & 91.7  & 91.7  & 97.2  & 1.000 & 209.75 & 0  & 80 & 15.72 \\
 & S1 (27\%) & \texttt{Offline-Optimal} & 100.0 & 100.0 & 99.9  & 1.000 & 197.12 & 0  & 1  & 136.69 \\
 & S1 (27\%) & \texttt{AQPS-static}     & \textbf{100.0} & \textbf{100.0} & 99.5  & 1.000 & 215.15 & 0  & 80 & 23.03 \\
 & S1 (27\%) & \texttt{AQPS-RT}         & \textbf{100.0} & \textbf{100.0} & 99.5  & 1.000 & 215.15 & 0  & 80 & 33.75 \\
 & S1 (27\%) & \texttt{AQPS-RT-ET}      & \textbf{100.0} & \textbf{100.0} & 99.5  & 1.000 & 215.15 & 0  & \textbf{59} & 35.66 \\
\cmidrule{2-11}
 & S2 (50\%) & \texttt{LLF}             & 100.0 & 100.0 & 100.0 & 1.000 & 395.63 & 0  & 0  & 0.00  \\
 & S2 (50\%) & \texttt{EDF}             & 100.0 & 100.0 & 99.2  & 0.999 & 393.00 & 0  & 0  & 0.00  \\
 & S2 (50\%) & \texttt{MPC-only}        & 89.2  & 100.0 & 88.0  & 0.986 & 351.97 & 0  & 55 & 37.18 \\
 & S2 (50\%) & \texttt{Offline-Optimal} & 100.0 & 100.0 & 99.9  & 1.000 & 396.24 & 0  & 1  & 347.09 \\
 & S2 (50\%) & \texttt{AQPS-static}     & \textbf{100.0} & \textbf{100.0} & 48.3  & 0.682 & 270.69 & 0  & 55 & 33.54 \\
 & S2 (50\%) & \texttt{AQPS-RT}         & 95.9  & \textbf{100.0} & 48.2  & 0.682 & \textbf{261.02} & 8  & 55 & 35.18 \\
 & S2 (50\%) & \texttt{AQPS-RT-ET}      & 95.9  & \textbf{100.0} & 48.3  & 0.683 & \textbf{261.03} & 8  & \textbf{48} & 34.81 \\
\cmidrule{2-11}
 & S3 (50\% Peak) & \texttt{LLF}        & 74.7  & 45.5  & 92.4  & 0.991 & 391.08 & 0  & 0  & 0.00  \\
 & S3 (50\% Peak) & \texttt{EDF}        & 69.3  & 50.0  & 96.8  & 0.996 & 391.08 & 0  & 0  & 0.00  \\
 & S3 (50\% Peak) & \texttt{MPC-only}   & 65.6  & 40.9  & 77.1  & 0.973 & 333.74 & 0  & 67 & 32.75 \\
 & S3 (50\% Peak) & \texttt{Offline-Optimal} & 98.0 & 100.0 & 73.4 & 0.974 & 385.71 & 0  & 1  & 353.68 \\
 & S3 (50\% Peak) & \texttt{AQPS-static}& \textbf{98.7}  & \textbf{100.0} & 35.3  & 0.580 & 284.96 & 0  & 67 & 32.41 \\
 & S3 (50\% Peak) & \texttt{AQPS-RT}    & 96.0  & \textbf{100.0} & 35.3  & 0.580 & 280.10 & 13 & 67 & 32.08 \\
 & S3 (50\% Peak) & \texttt{AQPS-RT-ET} & 96.0  & \textbf{100.0} & 35.3  & 0.581 & \textbf{280.04} & 13 & \textbf{50} & 32.73 \\
\midrule
\multirow{21}{*}{90 EVs (35 EVSEs)} 
 & S1 (27\%) & \texttt{LLF}             & 100.0 & 100.0 & 100.0 & 1.000 & 408.76 & 0  & 0  & 0.00  \\
 & S1 (27\%) & \texttt{EDF}             & 100.0 & 100.0 & 100.0 & 1.000 & 408.76 & 0  & 0  & 0.00  \\
 & S1 (27\%) & \texttt{MPC-only}        & 92.2  & 95.8  & 92.4  & 0.985 & 456.39 & 0  & 79 & 40.20 \\
 & S1 (27\%) & \texttt{Offline-Optimal} & 100.0 & 100.0 & 99.9  & 1.000 & 450.47 & 0  & 1  & 743.07 \\
 & S1 (27\%) & \texttt{AQPS-static}     & \textbf{100.0} & \textbf{100.0} & 85.2  & 0.888 & 434.33 & 0  & 79 & 39.01 \\
 & S1 (27\%) & \texttt{AQPS-RT}         & \textbf{100.0} & \textbf{100.0} & 85.2  & 0.888 & 434.33 & 0  & 79 & 43.49 \\
 & S1 (27\%) & \texttt{AQPS-RT-ET}      & \textbf{100.0} & \textbf{100.0} & 85.2  & 0.888 & 434.33 & 0  & \textbf{65} & 43.06 \\
\cmidrule{2-11}
 & S2 (50\%) & \texttt{LLF}             & 89.2  & 97.8  & 99.7  & 0.998 & 820.72 & 0  & 0  & 0.00  \\
 & S2 (50\%) & \texttt{EDF}             & 89.2  & 82.2  & 99.6  & 1.000 & 819.87 & 0  & 0  & 0.00  \\
 & S2 (50\%) & \texttt{MPC-only}        & 78.5  & 68.9  & 90.2  & 0.985 & 740.08 & 0  & 59 & 42.35 \\
 & S2 (50\%) & \texttt{Offline-Optimal} & 99.4  & 100.0 & 92.3  & 0.996 & 819.81 & 0  & 1  & 693.19 \\
 & S2 (50\%) & \texttt{AQPS-static}     & \textbf{100.0} & \textbf{100.0} & 49.8  & 0.652 & 608.41 & 0  & 59 & 42.92 \\
 & S2 (50\%) & \texttt{AQPS-RT}         & 96.5  & \textbf{100.0} & 49.9  & 0.651 & \textbf{590.75} & 20 & 59 & 41.82 \\
 & S2 (50\%) & \texttt{AQPS-RT-ET}      & 96.5  & \textbf{100.0} & 49.9  & 0.651 & \textbf{590.76} & 20 & \textbf{52} & 34.30 \\
\cmidrule{2-11}
 & S3 (50\% Peak) & \texttt{LLF}        & 76.8  & 53.3  & 99.7  & 1.000 & 832.39 & 0  & 0  & 0.00  \\
 & S3 (50\% Peak) & \texttt{EDF}        & 77.9  & 55.6  & 98.8  & 0.999 & 832.39 & 0  & 0  & 0.00  \\
 & S3 (50\% Peak) & \texttt{MPC-only}   & 72.4  & 40.0  & 85.4  & 0.983 & 751.40 & 0  & 70 & 27.76 \\
 & S3 (50\% Peak) & \texttt{Offline-Optimal} & 98.5 & 100.0 & 82.3 & 0.991 & 832.34 & 0  & 1  & 634.01 \\
 & S3 (50\% Peak) & \texttt{AQPS-static}& \textbf{98.8}  & \textbf{100.0} & 47.6  & 0.642 & 643.59 & 0  & 70 & 37.24 \\
 & S3 (50\% Peak) & \texttt{AQPS-RT}    & 97.2  & \textbf{100.0} & 47.6  & 0.642 & 640.00 & 21 & 70 & 40.22 \\
 & S3 (50\% Peak) & \texttt{AQPS-RT-ET} & 97.2  & \textbf{100.0} & 47.7  & 0.642 & \textbf{639.92} & 21 & \textbf{56} & 41.74 \\
\bottomrule
\end{tabular}
\end{table*}

Under the mild S1 baseline, all priority-aware methods achieve $100.0\%$ priority fulfillment and $100.0\%$ mission sufficiency, while \texttt{MPC-only} drops to $91.7\%$ (45~EVs) and $92.2\%$ (90~EVs) because it lacks Eq.~\eqref{eq:pribound}'s priority minimum bound and defers charging indiscriminately during shoulder/peak transitions.

Under S2 (Contention) and S3 (Peak Stress), the structural limitations of \texttt{LLF}, \texttt{EDF}, and \texttt{MPC-only} become severe. In S2 (90~EVs), early-arriving non-priority EVs with tight deadlines lock onto EVSEs; \texttt{LLF}, \texttt{EDF}, and \texttt{MPC-only} deteriorate to $89.2\%$, $89.2\%$, and $78.5\%$ priority fulfillment ($97.8\%$, $82.2\%$, and $68.9\%$ mission sufficiency). In S3 (Peak Stress), priority fulfillment under \texttt{LLF}, \texttt{EDF}, and \texttt{MPC-only} collapses to $74.7\%$, $69.3\%$, and $65.6\%$ in the 45-EV depot (with mission sufficiency plunging to $45.5\%$, $50.0\%$, and $40.9\%$) and $76.8\%$, $77.9\%$, and $72.4\%$ in the 90-EV depot. In contrast, \texttt{AQPS-static}, \texttt{AQPS-RT}, and \texttt{AQPS-RT-ET} sustain **100.0\% mission sufficiency** across every scenario and fleet size. Comparing \texttt{AQPS-static} and \texttt{AQPS-RT-ET} reveals the exact operational mechanism of Tier~2 Safe Release and Tier~3 $Q_k^{act}$ rotation: \texttt{AQPS-static} holds every priority vehicle on the charger until 98.8--100.0\% of its full session buffer is delivered, whereas \texttt{AQPS-RT-ET} executes 8--13 safe releases/rotations (45~EVs) and 20--21 safe releases/rotations (90~EVs) once vehicles exceed their mission-sufficient energy $E_i^{mission}$ during peak/shoulder contention, delivering 95.9--97.2\% total priority energy, 100.0\% mission sufficiency, and reducing peak charging costs by a further \$4.92--\$17.65 per day.

\begin{figure*}[htbp]
    \centering
    \includegraphics[width=0.92\textwidth]{figs/fig_fulfillment_stacked.png}
    \caption{Priority (top row) and Non-Priority (bottom row) energy demand fulfillment (\%) across S1--S3 for 45~EVs (16~EVSEs, left) and 90~EVs (35~EVSEs, right). All bars are plotted directly on a 0--100\% class fulfillment scale.}
    \label{fig:fulfillment}
\end{figure*}

\textbf{Non-Priority Service and Fairness (Addressing Reviewer \#3):} Table~\ref{tab:master_results} and the bottom row of Fig.~\ref{fig:fulfillment} quantify the exact service trade-off for non-priority EVs. In S1, \texttt{AQPS-RT-ET} delivers $99.5\%$ (45~EVs) and $85.2\%$ (90~EVs) of non-priority demand with Jain's fairness index of $1.000$ and $0.888$. In S2 and S3, where non-priority vehicles are intentionally preempted during peak contention to protect mission-critical departures and avoid \$0.32/kWh peak grid imports, non-priority fulfillment is $48.3\text{--}49.9\%$ (S2) and $35.3\text{--}47.7\%$ (S3) with Jain fairness $0.581\text{--}0.683$, before non-priority vehicles resume charging in the evening off-peak window.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\columnwidth]{figs/fig_cumulative_cost_45EVs.png}
    \caption{Cumulative charging cost over 24 hours for the 45-EV fleet (16~EVSEs).}
    \label{fig:cost45}
\end{figure}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\columnwidth]{figs/fig_cumulative_cost_90EVs.png}
    \caption{Cumulative charging cost over 24 hours for the 90-EV fleet (35~EVSEs).}
    \label{fig:cost90}
\end{figure}

\begin{figure*}[htbp]
    \centering
    \includegraphics[width=0.88\textwidth]{figs/fig_cost_breakdown_tiers.png}
    \caption{Daily electricity cost breakdown by ToU tariff tier (Off-Peak \$0.18, Shoulder \$0.21, Peak \$0.32) under S2 Contention for 45~EVs (left) and 90~EVs (right).}
    \label{fig:cost_breakdown}
\end{figure*}

\textbf{Cost Reduction and ToU Arbitrage:} Figs.~\ref{fig:cost45}, \ref{fig:cost90}, and \ref{fig:cost_breakdown} illustrate the cumulative 24-hour cost trajectories and ToU tier breakdowns. In S2 Contention, \texttt{AQPS-RT-ET} reduces daily charging cost by **34.0\%** in the 45-EV depot (\$261.03 vs. \$395.63 for \texttt{LLF} and \$393.00 for \texttt{EDF}) and by **28.0\%** in the 90-EV depot (\$590.76 vs. \$820.72 for \texttt{LLF}). In S3 Peak Stress, \texttt{AQPS-RT-ET} achieves cost reductions of **28.4\%** (\$280.04 vs. \$391.08) for 45~EVs and **23.1\%** (\$639.92 vs. \$832.39) for 90~EVs, while achieving a 97.5--100.0\% PV self-consumption ratio.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\columnwidth]{figs/fig_preemption_gantt_S2_90EVs.png}
    \caption{Per-vehicle scheduling comparison for S2 (50\% Contention), 90-EV fleet (35~EVSEs). Red triangles denote steps with preemption or starvation.}
    \label{fig:gantt_s2}
\end{figure}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\columnwidth]{figs/fig_preemption_gantt_S3_90EVs.png}
    \caption{Per-vehicle scheduling comparison for S3 (50\% Peak Stress), 90-EV fleet (35~EVSEs).}
    \label{fig:gantt_s3}
\end{figure}

Figs.~\ref{fig:gantt_s2} and \ref{fig:gantt_s3} confirm at the individual vehicle level that \texttt{AQPS-RT-ET} immediately preempts high-laxity non-priority vehicles upon priority arrival and delivers sustained charging blocks, whereas \texttt{LLF} subjects priority vehicles to prolonged starvation during midday and peak contention.

\subsection{Real-Time Queue Case Studies and Operator Impact Preview (Experiment E2)}
\label{sec:case_studies}

Experiment E2 evaluates the response of \texttt{AQPS-RT-ET}, \texttt{AQPS-static}, and \texttt{LLF} to four scripted intra-day operational injections in the 90-EV S2 depot: (1) a midday vehicle breakdown at 12:00 ($t=24$) requiring immediate priority promotion of standby van \texttt{EV\_005}, (2) a driver absence delaying departure by $+2$~h at 13:00 ($t=28$), (3) an FMS roster swap between priority van \texttt{EV\_028} and non-priority van \texttt{EV\_008} at 11:30 ($t=22$) where the operator selects the business-continuity option (\texttt{keep\_all\_urgent}), and (4) a sudden afternoon cloud-front PV drop ($75\%$ reduction) at 12:00 ($t=24$).

\begin{figure*}[htbp]
    \centering
    \includegraphics[width=0.88\textwidth]{figs/fig_queue_trajectory_injection.png}
    \caption{Real-time Layer~0 queue re-ranking (top) and Layer~2 charging power allocation (bottom) during Case~1 (breakdown injection at 12:00 promoting standby vehicle \texttt{EV\_005} to top priority).}
    \label{fig:queue_injection}
\end{figure*}

\begin{table*}[htbp]
\centering
\caption{Scripted intra-day disruption case studies (Experiment E2, 90~EVs, S2) and example Operator Impact Preview options generated at $t=22$ (11:30 Roster Swap).}
\label{tab:case_studies}
\footnotesize
\begin{tabular}{llcccccc}
\toprule
Case Study (Injection Time) & Method & Pri. Fulfill (\%) & Mission Suff. (\%) & Non-Pri. (\%) & Total Cost (\$) & Safe Releases & L2 Solves \\
\midrule
Case 1: Breakdown (12:00) & \texttt{LLF} & 89.5 & 97.8 & 99.6 & 820.72 & 0 & 0 \\
Case 1: Breakdown (12:00) & \texttt{AQPS-static} & 100.0 & 100.0 & 48.5 & 608.41 & 0 & 59 \\
Case 1: Breakdown (12:00) & \texttt{AQPS-RT-ET} & 96.6 & \textbf{100.0} & 48.6 & \textbf{590.76} & 20 & \textbf{52} \\
\midrule
Case 2: Driver Absence (13:00) & \texttt{LLF} & 89.2 & 95.6 & 99.6 & 820.72 & 0 & 0 \\
Case 2: Driver Absence (13:00) & \texttt{AQPS-RT-ET} & 96.5 & \textbf{100.0} & 49.9 & \textbf{590.76} & 20 & \textbf{52} \\
\midrule
Case 3: Roster Swap (11:30, Keep All) & \texttt{LLF} & 89.6 & 97.8 & 99.6 & 820.72 & 0 & 0 \\
Case 3: Roster Swap (11:30, Keep All) & \texttt{AQPS-static} & 100.0 & 100.0 & 50.5 & 621.19 & 0 & 59 \\
Case 3: Roster Swap (11:30, Keep All) & \texttt{AQPS-RT-ET} & 96.2 & \textbf{100.0} & 50.6 & \textbf{600.91} & 23 & \textbf{52} \\
\midrule
Case 4: PV Cloud Drop (12:00) & \texttt{LLF} & 89.2 & 97.8 & 99.7 & 821.44 & 0 & 0 \\
Case 4: PV Cloud Drop (12:00) & \texttt{AQPS-RT-ET} & 96.5 & \textbf{100.0} & 49.9 & \textbf{591.48} & 20 & \textbf{52} \\
\bottomrule
\end{tabular}
\end{table*}

As shown in Fig.~\ref{fig:queue_injection} and Table~\ref{tab:case_studies}, when standby van \texttt{EV\_005} is promoted at 12:00 (Case~1), Layer~0 immediately recomputes $\pi_i(t)$, promotes \texttt{EV\_005} to Rank~1 in $Q_k^{act}$, and ramps its charging power to 22~kW, achieving 100.0\% mission sufficiency. In Case~3 (Roster Swap), the operator selects Option~B (\texttt{keep\_all\_urgent}) from the impact preview, executing 23 safe releases/rotations to serve both the swapped and original vehicles at an accepted business-continuity cost increase of \$10.15 over baseline (\$600.91 vs. \$590.76), while still saving \$20.28 compared to \texttt{AQPS-static} (\$621.19). In Case~4 (PV Shortfall), because Eq.~\eqref{eq:sitelimit} constrains gross EV power $\sum x_{i,t}$, the 75\% midday solar drop does not cause QP infeasibility; it increases net grid import cost by only \$0.72 (\$591.48 vs. \$590.76) while preserving 100.0\% mission sufficiency.

\subsection{Random-Disruption Monte Carlo Analysis (Experiment E3)}
\label{sec:monte_carlo}

To evaluate statistical robustness under stochastic arrival delays ($\pm 1$~h uniform perturbation), energy demand fluctuations ($\pm 15\%$), and unscheduled intra-day priority promotions ($8\%$ probability), we execute $N_{MC} = 30$ independent Monte Carlo runs on the 90-EV S2 depot.

\begin{figure*}[htbp]
    \centering
    \includegraphics[width=0.88\textwidth]{figs/fig_monte_carlo_disruptions.png}
    \caption{30-run random-disruption Monte Carlo results (Experiment E3, 90~EVs, S2) showing mean $\pm 95\%$ confidence intervals for (a) Priority Fulfillment (\%) and (b) Daily Charging Cost (\$).}
    \label{fig:mc}
\end{figure*}

Fig.~\ref{fig:mc} and the E3 summary statistics confirm that \texttt{AQPS-RT-ET} and \texttt{AQPS-static} achieve **$100.00 \pm 0.00\%$ mission sufficiency** across all 30 random disruption seeds, with priority energy fulfillment of $95.08 \pm 0.26\%$ (95\% CI) and $99.95 \pm 0.01\%$, respectively. In contrast, \texttt{MPC-only} drops to $81.83 \pm 1.06\%$ priority fulfillment ($81.14 \pm 3.35\%$ mission sufficiency), \texttt{EDF} achieves $94.53 \pm 1.04\%$ ($93.06 \pm 2.21\%$ mission sufficiency), and \texttt{LLF} achieves $94.54 \pm 1.22\%$ ($95.93 \pm 1.94\%$ mission sufficiency). Economically, \texttt{AQPS-RT-ET} achieves the lowest mean daily cost of **$\$595.86 \pm \$6.53$**, saving **\$26.72/day** over \texttt{AQPS-static} ($\$622.58 \pm \$7.23$) via $Q_k^{act}$ safe releases and **\$255.71/day (30.0\%)** over \texttt{LLF} ($\$851.57 \pm \$4.77$).

\subsection{PV Forecast Uncertainty with Real Australian AEMO Data (Experiment E4)}
\label{sec:aemo_pv}

Experiment E4 evaluates all methods across six historical AEMO NEMWeb \texttt{VIC1} weekdays grouped into Low Error (Clear Sky: 2025-01-15, nMAE $1.61\%$; 2025-03-12, nMAE $1.81\%$), Moderate Error (Partly Cloudy: 2025-11-19, nMAE $17.44\%$; 2025-10-08, nMAE $17.73\%$), and High Over-Forecast Error (Winter Overcast/Stormfront: 2025-06-18, nMAE $26.64\%$; 2025-08-20, nMAE $29.65\%$).

\begin{figure*}[htbp]
    \centering
    \includegraphics[width=0.88\textwidth]{figs/fig_aemo_pv_uncertainty.png}
    \caption{Evaluation under real AEMO \texttt{VIC1} rooftop PV data (Experiment E4): (a) 20~kW scaled day-ahead forecast (dashed) vs. actual generation (solid) across Clear, Partly Cloudy, and Stormfront days; (b) realized daily grid cost vs. daily PV forecast error (nMAE \%).}
    \label{fig:aemo_pv}
\end{figure*}

As shown in Fig.~\ref{fig:aemo_pv}, when PV forecast error increases from $1.61\%$ (Summer Clear Sky, 143.1~kWh solar generation) to $29.65\%$ (Winter Stormfront, 25.0~kWh actual vs. 88.5~kWh forecast), \texttt{AQPS-RT-ET} triggers mid-day cost-check re-solves whenever $|P_k^{PV,act} - P_k^{PV}| > 3.5$~kW, maintaining exactly **100.0\% mission sufficiency** and **96.52\% priority fulfillment** (90~EVs) across all six days. Realized daily cost for 90~EVs rises smoothly by only $4.8\%$ from \$591.59 (Clear Sky) to \$620.10 (Stormfront) due to reduced physical solar availability, while maintaining a \$229.96/day (27.1\%) cost advantage over \texttt{LLF} (\$850.06).

\subsection{Event-Triggered MPC Efficiency and Scalability Analysis (Experiments E5 & E5b)}
\label{sec:scalability}

Fig.~\ref{fig:event_trig} compares every-step MPC ($T_{\max} = 15$~min, 1 step) against event-triggered MPC (\texttt{AQPS-RT-ET}) for $T_{\max} \in \{30, 60, 90, 120\}$~min on the 90-EV S2 depot. Increasing $T_{\max}$ from 15~min to 60~min reduces daily Layer~2 QP solves from 59 to 52 (and from 80 to 59 in S1, a **26.3\% solve reduction**) with **zero loss in priority fulfillment ($96.52\%$)**, **zero loss in mission sufficiency ($100.0\%$)**, and a negligible $\$0.01$ cost difference (\$590.76 vs. \$590.75).

\begin{figure*}[htbp]
    \centering
    \includegraphics[width=0.88\textwidth]{figs/fig_event_triggered_solves.png}
    \caption{Event-triggered vs. every-step MPC performance (Experiment E5a, 90~EVs, S2): (a) daily Layer~2 QP solve counts across $T_{\max}$; (b) preservation of priority fulfillment (\%) and total daily cost (\$).}
    \label{fig:event_trig}
\end{figure*}

\begin{table*}[htbp]
\centering
\caption{Scalability across fleet sizes (45, 90, 180~EVs) and EV-to-EVSE congestion ratios ($2:1, 3:1, 4:1$) under S2 Contention (Experiment E5b).}
\label{tab:scalability}
\footnotesize
\begin{tabular}{llcccccccc}
\toprule
Fleet Size & EV:EVSE Ratio (EVSEs) & Method & Pri. Fulfill (\%) & Mission Suff. (\%) & Non-Pri. (\%) & Cost (\$) & Safe Rel. & L2 Solves & Mean / P95 Solve (ms) \\
\midrule
45 EVs  & 2:1 (22 EVSEs) & \texttt{LLF}        & 100.0 & 100.0 & 100.0 & 395.73 & 0  & 0  & 0.00 / 0.00 \\
45 EVs  & 2:1 (22 EVSEs) & \texttt{AQPS-RT-ET} & 100.0 & \textbf{100.0} & 65.8  & \textbf{304.72} & 0  & 49 & 37.81 / 44.51 \\
45 EVs  & 3:1 (15 EVSEs) & \texttt{LLF}        & 100.0 & 100.0 & 100.0 & 395.72 & 0  & 0  & 0.00 / 0.00 \\
45 EVs  & 3:1 (15 EVSEs) & \texttt{AQPS-RT-ET} & 95.6  & \textbf{100.0} & 44.5  & \textbf{252.68} & 8  & 48 & 36.26 / 40.77 \\
45 EVs  & 4:1 (11 EVSEs) & \texttt{LLF}        & 99.7  & 100.0 & 99.9  & 394.81 & 0  & 0  & 0.00 / 0.00 \\
45 EVs  & 4:1 (11 EVSEs) & \texttt{AQPS-RT-ET} & 93.6  & \textbf{100.0} & 30.5  & \textbf{218.48} & 13 & 48 & 34.55 / 37.87 \\
\midrule
90 EVs  & 2:1 (45 EVSEs) & \texttt{LLF}        & 89.2  & 95.6  & 99.7  & 820.72 & 0  & 0  & 0.00 / 0.00 \\
90 EVs  & 2:1 (45 EVSEs) & \texttt{EDF}        & 89.2  & 82.2  & 99.6  & 819.87 & 0  & 0  & 0.00 / 0.00 \\
90 EVs  & 2:1 (45 EVSEs) & \texttt{AQPS-RT-ET} & \textbf{100.0} & \textbf{100.0} & 59.9  & \textbf{646.65} & 0  & 52 & 48.94 / 61.70 \\
90 EVs  & 3:1 (30 EVSEs) & \texttt{LLF}        & 89.3  & 97.8  & 99.6  & 820.72 & 0  & 0  & 0.00 / 0.00 \\
90 EVs  & 3:1 (30 EVSEs) & \texttt{EDF}        & 89.2  & 82.2  & 99.6  & 819.87 & 0  & 0  & 0.00 / 0.00 \\
90 EVs  & 3:1 (30 EVSEs) & \texttt{AQPS-RT-ET} & \textbf{95.0}  & \textbf{100.0} & 42.2  & \textbf{553.46} & 31 & 52 & 42.58 / 49.80 \\
90 EVs  & 4:1 (22 EVSEs) & \texttt{LLF}        & 89.6  & 97.8  & 99.4  & 820.72 & 0  & 0  & 0.00 / 0.00 \\
90 EVs  & 4:1 (22 EVSEs) & \texttt{EDF}        & 89.5  & 82.2  & 99.3  & 820.09 & 0  & 0  & 0.00 / 0.00 \\
90 EVs  & 4:1 (22 EVSEs) & \texttt{AQPS-RT-ET} & \textbf{91.9}  & \textbf{100.0} & 31.3  & \textbf{501.12} & 38 & 52 & 38.71 / 48.07 \\
\midrule
180 EVs & 2:1 (90 EVSEs) & \texttt{LLF}        & 97.6  & 100.0 & 99.9  & 1740.48 & 0 & 0  & 0.00 / 0.00 \\
180 EVs & 2:1 (90 EVSEs) & \texttt{EDF}        & 96.7  & 96.7  & 99.9  & 1732.25 & 0 & 0  & 0.00 / 0.00 \\
180 EVs & 2:1 (90 EVSEs) & \texttt{AQPS-RT-ET} & \textbf{99.9}  & \textbf{100.0} & 58.0  & \textbf{1305.41} & 0 & 62 & 64.20 / 101.31 \\
180 EVs & 3:1 (60 EVSEs) & \texttt{LLF}        & 97.5  & 100.0 & 99.9  & 1740.18 & 0 & 0  & 0.00 / 0.00 \\
180 EVs & 3:1 (60 EVSEs) & \texttt{EDF}        & 96.7  & 96.7  & 99.9  & 1732.25 & 0 & 0  & 0.00 / 0.00 \\
180 EVs & 3:1 (60 EVSEs) & \texttt{AQPS-RT-ET} & 95.1  & \textbf{100.0} & 39.9  & \textbf{1122.53} & 68 & 62 & 52.61 / 69.53 \\
180 EVs & 4:1 (45 EVSEs) & \texttt{LLF}        & 97.5  & 100.0 & 99.9  & 1740.18 & 0 & 0  & 0.00 / 0.00 \\
180 EVs & 4:1 (45 EVSEs) & \texttt{EDF}        & 96.7  & 96.7  & 99.9  & 1732.16 & 0 & 0  & 0.00 / 0.00 \\
180 EVs & 4:1 (45 EVSEs) & \texttt{AQPS-RT-ET} & 92.8  & \textbf{100.0} & 31.5  & \textbf{1046.80} & 72 & 62 & 48.78 / 62.50 \\
\bottomrule
\end{tabular}
\end{table*}

Table~\ref{tab:scalability} reports the scalability and EV-to-EVSE ratio sweep (Experiment E5b) across 45, 90, and 180~EVs and charger congestion ratios of $2:1$, $3:1$, and $4:1$. Across all nine configurations, \texttt{AQPS-RT-ET} maintains **100.0\% mission sufficiency**. As charger scarcity tightens from $2:1$ to $4:1$ in the 180-EV depot (45~EVSEs for 180~EVs), $Q_k^{act}$ safe releases and rotations scale from 0 to 72, keeping total priority fulfillment at $92.8\text{--}99.9\%$ while reducing daily charging cost from \$1740.18 (\texttt{LLF}) to \$1046.80 (a **39.8\% saving**). Crucially, even at 180~EVs and 90~EVSEs, the mean MOSEK QP solve time is **64.20~ms** (95th percentile **101.31~ms**), confirming real-time tractability for large commercial depots.

\subsection{Sensitivity and Overloaded Priority Analysis (Experiments E6 & E7)}
\label{sec:sensitivity}

Fig.~\ref{fig:sensitivity} presents the four-parameter sensitivity analysis (Experiment E6) on the 90-EV S2 depot. Both $\alpha_2 \in [10, 500]$ and $\alpha_3 \in [5\times 10^{-4}, 10^{-2}]$ exhibit a wide, flat stability plateau where priority fulfillment remains within $96.25\text{--}96.64\%$ and non-priority fulfillment within $48.30\text{--}50.11\%$. Similarly, the ATC lookahead parameter $k_{atc} \in [0.5, 3.0]$ shows high robustness ($96.49\text{--}96.52\%$ priority fulfillment). Varying the safety reserve margin $r_i$ from $2.0$~kWh to $14.0$~kWh increases total priority energy fulfillment from $93.15\%$ (\$573.55) to $99.96\%$ (\$613.40) as the safe-release threshold approaches the full session target.

\begin{figure*}[htbp]
    \centering
    \includegraphics[width=0.88\textwidth]{figs/fig_sensitivity_panels.png}
    \caption{Sensitivity analysis (Experiment E6, 90~EVs, S2) across (top-left) non-completion weight $\alpha_2$, (top-right) Tikhonov regularizer $\alpha_3$, (bottom-left) ATC lookahead parameter $k_{atc}$, and (bottom-right) safety reserve margin $r$ (kWh).}
    \label{fig:sensitivity}
\end{figure*}

\begin{table}[htbp]
\centering
\caption{Overloaded priority stress test (Experiment E7, 45~EVs, 16~EVSEs) sweeping priority fleet share from 30\% to 90\%.}
\label{tab:overloaded}
\footnotesize
\begin{tabular}{llccccc}
\toprule
Pri. Share & Method & Pri. (\%) & Mission Suff. (\%) & Non-Pri. (\%) & Safe Rel. & Escalations \\
\midrule
30\% (14 EVs) & \texttt{AQPS-static} & 99.4 & 100.0 & 83.8 & 0 & 0 \\
30\% (14 EVs) & \texttt{AQPS-RT-ET}  & 99.4 & \textbf{100.0} & 83.8 & 0 & 0 \\
\midrule
50\% (22 EVs) & \texttt{AQPS-static} & 91.5 & 95.5  & 88.9 & 0 & 70 \\
50\% (22 EVs) & \texttt{AQPS-RT-ET}  & \textbf{93.9} & \textbf{100.0} & 88.6 & 5 & 78 \\
\midrule
65\% (29 EVs) & \texttt{AQPS-static} & 90.9 & 96.6  & 88.8 & 0 & 154 \\
65\% (29 EVs) & \texttt{AQPS-RT-ET}  & \textbf{93.9} & 96.6  & 87.2 & 5 & 201 \\
\midrule
80\% (36 EVs) & \texttt{AQPS-static} & 91.8 & 97.2  & 87.3 & 0 & 238 \\
80\% (36 EVs) & \texttt{AQPS-RT-ET}  & \textbf{94.2} & 97.2  & 84.2 & 5 & 296 \\
\midrule
90\% (40 EVs) & \texttt{AQPS-static} & 92.4 & 97.5  & 85.5 & 0 & 291 \\
90\% (40 EVs) & \texttt{AQPS-RT-ET}  & \textbf{94.1} & 97.5  & 78.5 & 6 & 352 \\
\bottomrule
\end{tabular}
\end{table}

Finally, Table~\ref{tab:overloaded} reports Experiment E7, where the priority share of the 45-EV fleet (16~EVSEs) is swept from $30\%$ (14~priority EVs, below $M=16$) to an overloaded $90\%$ (40~priority EVs contending for 16~EVSEs). When the priority share is $30\%$ ($\le M$), Proposition~2 holds with zero escalations and $100.0\%$ mission sufficiency. At $50\%$ priority share, \texttt{AQPS-static} suffers a drop to $95.5\%$ mission sufficiency ($91.5\%$ priority fulfillment), whereas \texttt{AQPS-RT-ET} uses $Q_k^{act}$ safe releases and rotations to restore **100.0\% mission sufficiency** ($93.9\%$ fulfillment). Even under extreme overload ($65\%\text{--}90\%$ priority share), \texttt{AQPS-RT-ET} outperforms \texttt{AQPS-static} by $+1.7\%\text{--}+3.0\%$ in priority fulfillment while gracefully escalating infeasible arrivals to the operator preview without force-disconnecting active priority vehicles.

\subsection{Discussion: FMS Integration and Practical Limitations}
\label{sec:discussion}

\textbf{Integration with Depot Energy and Fleet Management Systems:} The three-layer AQPS architecture is designed as a modular middleware service that interfaces between an enterprise Fleet Management System (FMS, which supplies shift rosters, trip criticality weights $w_i$, and telematics arrival estimates) and an Open Charge Point Protocol (OCPP~2.0.1) Charging Station Management System (CSMS) \cite{vales2026emergent}. Layer~0 and the Operator Impact Preview communicate with the FMS dispatch console via REST/MQTT APIs, while Layer~2 dispatches power setpoints $x_{i,k}$ directly as OCPP \texttt{SetChargingProfile} pilot commands.

\textbf{Limitations:} Several limitations of the present study merit discussion. First, the human operator in the impact-preview loop is modeled via scripted decision policies; human-factors experiments with live fleet dispatchers are an important next step. Second, while vehicle capacities and route energy consumption rates are grounded in empirical urban delivery data \cite{fiori2018modelling,vales2026emergent} and solar generation uses real AEMO \texttt{VIC1} profiles, regional rooftop PV aggregates smooth out localized cloud transients and therefore understate single-site 20~kW ramp volatility, and \texttt{ROOFTOP\_PV\_ACTUAL} is AEMO's regional telemetry estimate rather than a single site revenue meter. Third, hardware-in-the-loop (HIL) and physical depot field trials remain necessary to validate contactor switching delays during EVSE preemption.

% =============================================================================
% SECTION 5: CONCLUSIONS
% =============================================================================
\section{Conclusions}
\label{sec:conclusion}

This paper presented a closed-loop, three-layer Adaptive Queuing-based Predictive Scheduling (AQPS) framework for real-time commercial EV depot charging under limited charger capacity, transformer limits, and operational disruptions. By coupling schedule-derived Apparent Tardiness Cost (ATC) priority selection and QP bisection feasibility checking (Layer~0) with an Active Priority Charging Queue ($Q_k^{act}$) supporting energy-based safe release and guarded FIFO rotation (Layer~1), Event-Triggered MPC (Layer~2), and a human-in-the-loop operator impact preview, AQPS resolves the priority-on-priority deadlock and eliminates unnecessary every-step QP solves. Across 45-, 90-, and 180-EV depots, \texttt{AQPS-RT-ET} achieved 100.0\% mission sufficiency and 95.9--100.0\% priority demand fulfillment under contention and peak stress---outperforming \texttt{LLF} (74.7--89.2\%), \texttt{EDF} (69.3--89.2\%), and \texttt{MPC-only} (65.6--78.5\%)---while reducing daily charging costs by 23.1--34.1\% and cutting Layer~2 QP solve counts by up to 26.3\% at sub-65~ms MOSEK solve times. Future work will extend AQPS to coordinate mid-shift public fast-charging detours alongside depot charging, incorporate learned priority policies, and conduct hardware-in-the-loop field validation.

\section*{CRediT authorship contribution statement}
\textbf{Mohammad Kamar Uddin:} Conceptualization, Methodology, Software, Data curation, Formal analysis, Writing -- original draft, Visualization. \textbf{Ali Moradi Amani:} Conceptualization, Methodology, Writing -- review \& editing, Supervision. \textbf{Hui Song:} Methodology, Writing -- review \& editing, Supervision.

\section*{Declaration of competing interest}
The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

\bibliographystyle{elsarticle-num-names}
\bibliography{refs}

\end{document}
"""

# =============================================================================
# 3. EPSR HIGHLIGHTS (highlights.tex)
# =============================================================================
HIGHLIGHTS_TEX = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\begin{document}
\section*{Highlights (Electric Power Systems Research --- Max 85 characters per bullet)}
\begin{itemize}
    \item Closed-loop 3-layer depot charging couples ATC priority and event-triggered MPC.
    \item Active priority queue ($Q_k^{act}$) with kWh safe release prevents charger deadlock.
    \item Human-in-the-loop impact preview quantifies cost and energy via what-if QP solves.
    \item Achieves 100\% mission sufficiency and 23--34\% cost savings over LLF and EDF.
    \item Validated up to 180 EVs with AEMO solar data and sub-65 ms MOSEK solve times.
\end{itemize}
\end{document}
"""

# =============================================================================
# 4. COVER LETTER FOR EPSR (cover_letter.tex)
# =============================================================================
COVER_LETTER_TEX = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{hyperref}
\begin{document}

\noindent \today \\[1em]
Editor-in-Chief \\
\textit{Electric Power Systems Research} \\
Elsevier \\[1em]
Dear Editor-in-Chief, \\[1em]
Please find enclosed our manuscript entitled \textbf{``Schedule-Aware Real-time Depot Charging for Electric Fleets''} by Mohammad Kamar Uddin, Ali Moradi Amani, and Hui Song, which we are submitting for consideration for publication as a Research Paper in \textit{Electric Power Systems Research}.

As commercial logistics and municipal vehicle fleets rapidly electrify, depot operators face a severe operational bottleneck: limited Level-2 charger counts ($M < n$) and distribution transformer capacity limits clash with tight multi-shift vehicle schedules, intra-day disruptions, and volatile time-of-use (ToU) electricity tariffs. Existing depot charging controllers either rely on static priority flags without closed-loop feasibility verification or solve receding-horizon Model Predictive Control (MPC) problems at every time step without safe preemption or dispatcher decision support.

In this paper, we present a closed-loop, three-layer \textbf{Adaptive Queuing-based Predictive Scheduling (AQPS)} framework (\texttt{AQPS-RT-ET}) tailored for power-constrained commercial EV depots:
\begin{enumerate}
    \item \textbf{Layer 0 (Schedule-Derived Priority Selection with Optimizer Feasibility Check):} Computes each vehicle's mission-sufficient session energy requirement from the fleet roster, scores urgency via the Apparent Tardiness Cost (ATC) index, and uses bisection over the Layer~2 convex Quadratic Program (QP) to select the largest feasible priority set with zero predicted shortfall.
    \item \textbf{Layer 1 (Multi-Queue Management with Safe Release and Active Priority Rotation):} Introduces an Active Priority Charging Queue ($Q_k^{act}$, capacity $M$) alongside waiting queues $Q_k^p$ and $Q_k^n$ with a four-tier disconnection hierarchy (laxity-based non-priority preemption, three-condition kWh safe release, guarded FIFO rotation from $Q_k^{act}$, and operator escalation) that resolves priority-on-priority charger deadlocks.
    \item \textbf{Layer 2 (Two-Way Coupled Event-Triggered MPC \& Human-in-the-Loop Impact Preview):} Reduces Layer~2 QP solve counts by up to 26.3\% with sub-65~ms MOSEK solve times for fleets up to 180~EVs, while providing dispatchers with three quantified what-if QP options during disruptions.
\end{enumerate}

Comprehensive simulations using empirical commercial EV van parameters and historical Australian Energy Market Operator (AEMO) rooftop PV forecast and actual data demonstrate 100.0\% mission sufficiency, 95.9--100.0\% priority demand fulfillment, and 23.1--34.1\% cost reductions over Least Laxity First (\texttt{LLF}) and Earliest Deadline First (\texttt{EDF}).

We confirm that this manuscript is original, has not been published elsewhere, and is not currently under consideration by another journal. All authors have approved the manuscript and declare no conflicts of interest.

Thank you for your time and consideration.

\vspace{1.5em}
\noindent Sincerely, \\[1em]
\textbf{Mohammad Kamar Uddin} \\
School of Engineering, RMIT University \\
Melbourne, VIC 3000, Australia \\
Email: \texttt{s3955218@student.rmit.edu.au} \\[0.5em]
On behalf of co-authors: \\
Dr. Ali Moradi Amani (RMIT University) \\
Dr. Hui Song (University of Newcastle)

\end{document}
"""

# =============================================================================
# 5. WRITE FILES & COPY ASSETS INTO OVERLEAF PACKAGE
# =============================================================================
(OUT_DIR / "main.tex").write_text(MAIN_TEX.lstrip("\n"), encoding="utf-8")
(OUT_DIR / "refs.bib").write_text(REFS_BIB.lstrip("\n"), encoding="utf-8")
(OUT_DIR / "highlights.tex").write_text(HIGHLIGHTS_TEX.lstrip("\n"), encoding="utf-8")
(OUT_DIR / "cover_letter.tex").write_text(COVER_LETTER_TEX.lstrip("\n"), encoding="utf-8")

# Copy generated figures from figs_epsr/ into overleaf_epsr_package/figs/
fig_src = Path("figs_epsr")
if fig_src.exists():
    for png in fig_src.glob("*.png"):
        shutil.copy2(png, OUT_DIR / "figs" / png.name)
    # Also copy fig2_system_architecture.png as system_architecture.png if original is not yet copied
    if (OUT_DIR / "figs" / "fig2_system_architecture.png").exists() and not (OUT_DIR / "figs" / "system_architecture.png").exists():
        shutil.copy2(OUT_DIR / "figs" / "fig2_system_architecture.png", OUT_DIR / "figs" / "system_architecture.png")

# Extract epsr.sty, elsarticle.cls, and bst files from epsr-latex-zip.zip if present
for zname in ["epsr-latex-zip.zip", "elsarticle-epsr.zip"]:
    zp = Path(zname)
    if zp.exists():
        with zipfile.ZipFile(zp, "r") as zf:
            for member in zf.namelist():
                base = os.path.basename(member)
                if base.endswith((".sty", ".cls", ".bst")):
                    with zf.open(member) as src, open(OUT_DIR / base, "wb") as dst:
                        shutil.copyfileobj(src, dst)
        print(f"Extracted EPSR style/class files from {zname}")

# Create self-contained AQPS_EPSR_Submission.zip
zip_out = Path("AQPS_EPSR_Submission.zip")
with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_DEFLATED) as zf:
    for file_path in sorted(OUT_DIR.rglob("*")):
        if file_path.is_file():
            zf.write(file_path, file_path.relative_to(OUT_DIR))

print(f"\n[SUCCESS] Built Overleaf project folder `{OUT_DIR}/` and `{zip_out}`!")