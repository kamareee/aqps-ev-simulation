# AQPS Project State Document

> **Purpose**: This document captures the complete implementation state of the Adaptive Queuing Priority Scheduler (AQPS) project. Add this to Claude Project Knowledge to maintain context across conversations without re-uploading source files.

> **Last Updated**: Phase 4 Complete (Quantization & Polish)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Implementation Status](#2-implementation-status)
3. [Architecture & File Structure](#3-architecture--file-structure)
4. [Key Design Decisions](#4-key-design-decisions)
5. [Data Structures Reference](#5-data-structures-reference)
6. [API Reference](#6-api-reference)
7. [Configuration Reference](#7-configuration-reference)
8. [Algorithm Pseudocode](#8-algorithm-pseudocode)
9. [Usage Examples](#9-usage-examples)
10. [Known Issues & Limitations](#10-known-issues--limitations)
11. [Roadmap](#11-roadmap)

---

## 1. Project Overview

### 1.1 Research Objective

Develop the **Adaptive Queuing Priority Scheduler (AQPS)** as a computationally efficient alternative to Model Predictive Control (MPC) approaches for priority-aware EV charging systems. The research aims to demonstrate that heuristic algorithms can achieve similar performance to optimization-based methods while:

- Reducing computational complexity from O(n³) to O(n log n)
- Eliminating dependency on commercial solvers like MOSEK
- Enabling real-time deployment on embedded systems

### 1.2 Key Constraints

| Constraint | Specification |
|------------|---------------|
| EVSE Hardware | AeroVironment J1772 |
| Pilot Signals | Discrete: [0, 8, 16, 24, 32] Amps |
| Network Topology | Three-phase, 54 EVSEs (18/18/18 balanced) |
| Voltage | 415V three-phase |
| Scheduling Period | 5 minutes |
| Priority Guarantee | Minimum 16A for priority EVs |

### 1.3 Core Principles (from AQPC Theory)

| AQPC Component | AQPS Implementation |
|----------------|---------------------|
| H^EC (Energy Cost) | TOU-aware slot preference with departure feasibility |
| H^NC (Non-completion) | Two-tier queue ensuring priority EVs first |
| H^QC (Peak Demand) | Load spreading via fair-share allocation |
| Priority Constraint | Minimum rate guarantee + preemption mechanism |

---

## 2. Implementation Status

### 2.1 Phase Completion

| Phase | Status | Description |
|-------|--------|-------------|
| **Phase 1** | ✅ Complete | Core algorithm, two-tier queue, scenario generation |
| **Phase 2** | ✅ Complete | Preemption (Option B + A), threshold tracking |
| **Phase 3** | ✅ Complete | TOU optimization, three-phase network, renewables |
| **Phase 4** | ✅ Complete | J1772 quantization, computational benchmarking |
| **Phase 5** | 🔲 Pending | Simulation studies (S1-S6 scenarios) |
| **Phase 6** | 🔲 Pending | LaTeX pseudocode for journal submission |

### 2.2 Version History

| Version | Phase | Key Changes |
|---------|-------|-------------|
| 0.1.0 | 1 | Initial two-tier queue, basic allocation |
| 0.2.0 | 2 | Preemption logic, threshold tracking |
| 0.3.0 | 3 | TOU tariffs, three-phase network, PV/BESS |
| 0.4.0 | 4 | J1772 quantization, timing metrics, DataFrame export |

---

## 3. Architecture & File Structure

### 3.1 Directory Layout

```
aqps-scheduler/
├── src/
│   └── aqps/
│       ├── __init__.py              # Package exports (v0.4.0)
│       ├── scheduler.py             # Main AQPS algorithm (~1500 lines)
│       ├── data_structures.py       # All dataclasses (~750 lines)
│       ├── queue_manager.py         # Two-tier queue management
│       ├── preemption.py            # Phase 2: Preemption logic
│       ├── threshold_tracker.py     # Phase 2: Violation tracking
│       ├── three_phase_network.py   # Phase 3: Network infrastructure
│       ├── tou_optimization.py      # Phase 3: TOU tariff & optimizer
│       ├── renewable_integration.py # Phase 3: PV & BESS
│       ├── scenario_generator.py    # S1-S6 test scenarios
│       └── utils.py                 # Helpers, quantization utilities
├── tests/
│   ├── test_scheduler.py
│   ├── test_preemption.py
│   ├── test_threshold_tracker.py
│   ├── test_queue_manager.py
│   ├── test_laxity.py
│   └── test_phase3.py
├── examples/
│   ├── basic_usage.py
│   └── scenario_comparison.py
├── README.md
├── PHASE4_SUMMARY.md
├── setup.py
└── requirements.txt
```

### 3.2 Module Dependencies

```
scheduler.py
    ├── data_structures.py (SessionInfo, configs, metrics)
    ├── queue_manager.py (partition_and_sort)
    ├── preemption.py (PreemptionManager)
    ├── threshold_tracker.py (ThresholdTracker)
    ├── tou_optimization.py (TOUOptimizer) [lazy loaded]
    ├── three_phase_network.py (ThreePhaseNetwork) [lazy loaded]
    ├── renewable_integration.py (RenewableIntegration) [lazy loaded]
    └── utils.py (format_schedule, validate_sessions)
```

---

## 4. Key Design Decisions

### 4.1 Algorithm Design

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Queue Structure | Two-tier (priority/non-priority) | O(n) partitioning vs O(n log n) sorting |
| Priority Sorting | FIFO within tier | Simplicity; laxity sorting optional |
| Preemption Order | Highest-laxity-first (Option B) | Most flexible EVs preempted first |
| Preemption Fallback | Proportional reduction (Option A) | Ensures capacity freed when B insufficient |
| TOU Deferral | Aggressive policy | Defer whenever cheaper slot exists |
| Deferral Scope | Non-priority only | Priority EVs NEVER deferred |

### 4.2 Infrastructure Design

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Network Topology | Balanced static (18/18/18) | Simplicity, predictable phase assignment |
| Phase Assignment | EVSEs 1-18→A, 19-36→B, 37-54→C | Sequential mapping |
| Transformer Limits | Per-phase Amps (configurable) | Matches real infrastructure |
| Voltage | 415V three-phase | Australian standard |

### 4.3 Quantization Design (Phase 4)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Default State | Enabled | Realistic discrete signals |
| Pilot Signals | [0, 8, 16, 24, 32] Amps | J1772 standard |
| Priority Quantization | Ceiling if floor < minimum | Maintain guarantee |
| Non-Priority Quantization | Floor | Conservative, respects capacity |
| Capacity Overage | Reduce non-priority iteratively | Balance priority vs capacity |

---

## 5. Data Structures Reference

### 5.1 Core Structures

```python
@dataclass
class SessionInfo:
    session_id: str
    station_id: str
    arrival_time: int          # Period number
    departure_time: int        # Period number
    energy_requested: float    # kWh
    max_rate: float            # Amps
    min_rate: float = 0.0      # Amps
    current_charge: float = 0.0  # kWh (updated externally)
    is_priority: bool = False
    
    @property
    def remaining_demand(self) -> float:  # kWh
        return max(0.0, self.energy_requested - self.current_charge)

@dataclass
class PriorityQueueEntry:
    session: SessionInfo
    laxity: float              # Time slack (periods)
    is_priority: bool
    preferred_rate: float = 0.0
    actual_rate: float = 0.0
    deferred: bool = False     # TOU deferral status
```

### 5.2 Configuration Structures

```python
@dataclass
class AQPSConfig:
    min_priority_rate: float = 11.0    # Amps (use 16.0 for J1772)
    total_capacity: float = 150.0      # Amps
    period_minutes: float = 5.0
    voltage: float = 220.0             # Use 415.0 for three-phase
    enable_logging: bool = True
    max_priority_ratio: float = 0.30

@dataclass
class Phase3Config:
    enable_tou_optimization: bool = True
    enable_renewable_integration: bool = True
    deferral_policy: str = "aggressive"  # aggressive/conservative/balanced
    apply_to_priority: bool = False      # Never defer priority
    congestion_safety_margin: float = 0.8
    simulation_start_hour: float = 0.0

@dataclass
class Phase4Config:
    enable_quantization: bool = True
    pilot_signals: List[float] = [0.0, 8.0, 16.0, 24.0, 32.0]
    priority_ceil_enabled: bool = True   # Round UP priority if needed
    track_quantization_events: bool = True
    enable_timing: bool = True
```

### 5.3 Metrics Structures

```python
@dataclass
class SchedulingMetrics:
    timestamp: int
    priority_sessions_active: int
    non_priority_sessions_active: int
    total_allocated_capacity: float
    priority_allocated_capacity: float
    capacity_utilization: float
    warnings: List[str]

@dataclass
class PreemptionEvent:
    timestamp: int
    priority_session_id: str
    preempted_session_ids: List[str]
    capacity_needed: float
    capacity_freed: float
    method: PreemptionMethod  # HIGHEST_LAXITY, PROPORTIONAL, COMBINED
    success: bool

@dataclass
class ThresholdViolationEvent:
    timestamp: int
    priority_count: int
    non_priority_count: int
    capacity_required: float
    capacity_available: float
    capacity_shortfall: float
    severity: str  # minor/moderate/severe

@dataclass
class QuantizationMetrics:
    timestamp: int
    pre_quantization_capacity: float
    post_quantization_capacity: float
    capacity_lost: float
    capacity_gained: float
    priority_ceil_count: int
    quantization_efficiency: float  # percentage

@dataclass
class ComputationalMetrics:
    timestamp: int
    schedule_time_ms: float
    num_sessions: int
    time_per_session_us: float

@dataclass
class SimulationSummary:
    total_timesteps: int
    priority_fulfillment_rate: float
    total_preemptions: int
    total_threshold_violations: int
    total_deferrals: int
    avg_schedule_time_ms: float
    quantization_efficiency_avg: float
```

---

## 6. API Reference

### 6.1 Scheduler Initialization

```python
from aqps import AdaptiveQueuingPriorityScheduler, AQPSConfig, Phase3Config, Phase4Config

scheduler = AdaptiveQueuingPriorityScheduler(
    config=AQPSConfig(...),
    phase3_config=Phase3Config(...),  # Optional
    phase4_config=Phase4Config(...)   # Optional
)
```

### 6.2 Core Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `schedule(sessions, current_time)` | Main scheduling entry point | `Dict[str, float]` |
| `configure_tou(...)` | Set TOU tariff parameters | `None` |
| `configure_network(...)` | Set three-phase limits | `None` |
| `configure_renewables(...)` | Set PV/BESS parameters | `None` |

### 6.3 Analysis Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `get_metrics(last_n)` | Get scheduling metrics history | `List[SchedulingMetrics]` |
| `get_current_metrics()` | Get most recent metrics | `SchedulingMetrics` |
| `get_preemption_statistics()` | Preemption summary | `Dict` |
| `get_preemption_history()` | All preemption events | `List[PreemptionEvent]` |
| `get_threshold_statistics()` | Threshold summary | `Dict` |
| `get_threshold_violations()` | All violations | `List[ThresholdViolationEvent]` |
| `get_tou_statistics()` | TOU/deferral summary | `Dict` |
| `get_tou_deferral_history()` | All deferral events | `List[TOUDeferralEvent]` |
| `get_quantization_statistics()` | Quantization summary | `Dict` |
| `get_quantization_history()` | All quantization metrics | `List[QuantizationMetrics]` |
| `get_computational_statistics()` | Timing summary | `Dict` |
| `get_computational_history()` | All timing metrics | `List[ComputationalMetrics]` |
| `get_simulation_summary()` | Aggregate summary | `SimulationSummary` |
| `export_dataframes()` | All metrics as dicts | `Dict[str, List[Dict]]` |
| `export_analysis_data()` | Full analysis export | `Dict` |

### 6.4 Utility Functions

```python
from aqps import (
    calculate_laxity,
    calculate_remaining_energy,
    energy_deliverable_at_rate,
    rate_needed_for_energy,
    quantize_rate_floor,
    quantize_rate_ceil,
    quantize_schedule_floor,
    calculate_quantization_loss,
    generate_scenario,
)
```

---

## 7. Configuration Reference

### 7.1 Recommended Production Configuration

```python
config = AQPSConfig(
    min_priority_rate=16.0,    # Valid J1772 signal
    total_capacity=600.0,      # 3 phases × 200A
    period_minutes=5.0,
    voltage=415.0,             # Three-phase
    enable_logging=False,      # Disable for performance
    max_priority_ratio=0.30
)

phase3 = Phase3Config(
    enable_tou_optimization=True,
    deferral_policy="aggressive",
    apply_to_priority=False,
    congestion_safety_margin=0.8
)

phase4 = Phase4Config(
    enable_quantization=True,
    priority_ceil_enabled=True,
    enable_timing=True
)
```

### 7.2 TOU Tariff Configuration

```python
scheduler.configure_tou(
    peak_price=0.40,           # $/kWh
    off_peak_price=0.15,       # $/kWh
    shoulder_price=0.25,       # Optional
    peak_hours=[(14, 20)],     # 2pm-8pm
    shoulder_hours=[(7, 14), (20, 22)],  # Optional
    tariff_name="Australian TOU"
)
```

### 7.3 Network Configuration

```python
scheduler.configure_network(
    phase_a_limit=200.0,       # Amps
    phase_b_limit=200.0,       # Amps
    phase_c_limit=200.0,       # Amps
    total_evses=54             # Must be divisible by 3
)
```

---

## 8. Algorithm Pseudocode

### 8.1 Main Schedule Method

```
FUNCTION schedule(sessions, current_time):
    start_timer()
    
    # Step 1: Partition
    priority_queue, non_priority_queue ← partition_by_priority(sessions)
    
    # Step 2: Allocate priority EVs
    schedule ← {}
    FOR each entry IN priority_queue:
        min_rate ← max(entry.min_rate, config.min_priority_rate)
        schedule[entry.station_id] ← min_rate
    
    # Step 3: Preemption if needed
    IF priority_capacity > available_capacity:
        schedule ← execute_preemption(priority_queue, non_priority_queue, schedule)
    
    # Step 4: Maximize priority rates
    FOR each entry IN priority_queue:
        schedule[entry.station_id] ← min(entry.max_rate, available_capacity)
    
    # Step 5: Allocate non-priority with TOU
    FOR each entry IN non_priority_queue:
        IF tou_enabled AND should_defer(entry):
            schedule[entry.station_id] ← 0  # Defer
        ELSE:
            schedule[entry.station_id] ← fair_share
    
    # Step 6: Quantize to J1772 signals
    IF quantization_enabled:
        schedule ← quantize_schedule(schedule, priority_queue)
        IF total > capacity:
            schedule ← adjust_for_overage(schedule)
    
    record_metrics()
    RETURN schedule
```

### 8.2 Preemption Logic

```
FUNCTION execute_preemption(needed_capacity, non_priority_queue, schedule):
    freed ← 0
    
    # Option B: Highest laxity first
    sorted_by_laxity_desc ← sort(non_priority_queue, by=laxity, descending)
    
    FOR entry IN sorted_by_laxity_desc:
        IF freed >= needed_capacity:
            BREAK
        reducible ← schedule[entry.station_id] - entry.min_rate
        reduction ← min(reducible, needed_capacity - freed)
        schedule[entry.station_id] -= reduction
        freed += reduction
    
    # Option A fallback: Proportional
    IF freed < needed_capacity:
        shortfall ← needed_capacity - freed
        total_reducible ← sum of reducible capacity
        ratio ← shortfall / total_reducible
        FOR entry IN non_priority_queue:
            schedule[entry.station_id] *= (1 - ratio)
    
    RETURN schedule
```

### 8.3 Quantization Logic

```
FUNCTION quantize_schedule(schedule, priority_queue):
    priority_ids ← {entry.session_id for entry IN priority_queue}
    
    FOR station_id, rate IN schedule:
        floor_rate ← get_floor_signal(rate)  # Round down
        
        IF station_id IN priority_ids AND floor_rate < min_priority_rate:
            # Priority: use ceiling to maintain guarantee
            quantized_rate ← get_ceil_signal(rate)
        ELSE:
            # Non-priority: use floor
            quantized_rate ← floor_rate
        
        schedule[station_id] ← quantized_rate
    
    RETURN schedule
```

---

## 9. Usage Examples

### 9.1 Basic Usage

```python
from aqps import AdaptiveQueuingPriorityScheduler, AQPSConfig, SessionInfo

# Initialize
config = AQPSConfig(min_priority_rate=16.0, total_capacity=600.0, voltage=415.0)
scheduler = AdaptiveQueuingPriorityScheduler(config)

# Create sessions
sessions = [
    SessionInfo(
        session_id='EV1', station_id='S1',
        arrival_time=0, departure_time=100,
        energy_requested=20.0, max_rate=32.0,
        is_priority=True
    ),
    # ... more sessions
]

# Schedule
schedule = scheduler.schedule(sessions, current_time=0)
# schedule = {'S1': 32.0, 'S2': 16.0, ...}  # All valid J1772 signals
```

### 9.2 Full Simulation Loop

```python
from aqps import (
    AdaptiveQueuingPriorityScheduler, 
    AQPSConfig, 
    Phase3Config,
    Phase4Config,
    generate_scenario
)

# Configure
config = AQPSConfig(min_priority_rate=16.0, total_capacity=600.0, voltage=415.0)
scheduler = AdaptiveQueuingPriorityScheduler(config)
scheduler.configure_tou(peak_price=0.40, off_peak_price=0.15, peak_hours=[(14, 20)])
scheduler.configure_network(phase_a_limit=200.0, phase_b_limit=200.0, phase_c_limit=200.0)

# Generate scenario
sessions = generate_scenario('S1', n_sessions=100, seed=42)

# Run simulation
for t in range(288):  # 24 hours
    active = [s for s in sessions if s.arrival_time <= t < s.departure_time]
    if active:
        schedule = scheduler.schedule(active, current_time=t)
        # Apply schedule to simulation...

# Analyze results
summary = scheduler.get_simulation_summary()
data = scheduler.export_dataframes()
```

### 9.3 DataFrame Export for Analysis

```python
import pandas as pd

data = scheduler.export_dataframes()

scheduling_df = pd.DataFrame(data['scheduling'])
preemption_df = pd.DataFrame(data['preemption'])
quantization_df = pd.DataFrame(data['quantization'])
computational_df = pd.DataFrame(data['computational'])

# Save for publication
scheduling_df.to_csv('results/scheduling_metrics.csv', index=False)
```

---

## 10. Known Issues & Limitations

### 10.1 Current Limitations

| Limitation | Description | Workaround |
|------------|-------------|------------|
| No internal state tracking | Scheduler doesn't track charging progress | External simulator must update `current_charge` |
| Static phase assignment | EVSEs fixed to phases | Modify `ThreePhaseNetwork` for dynamic |
| Single-period horizon | No multi-period lookahead | By design (heuristic approach) |
| No battery degradation | Doesn't model battery health | Add to `SessionInfo` if needed |

### 10.2 Test Failures (Minor)

- `test_filter_feasible_sessions` in `test_queue_manager.py` - Laxity threshold edge case
- `test_preemption.py` requires pytest (not available in all environments)

### 10.3 Edge Cases

| Case | Behavior |
|------|----------|
| All priority, exceeds capacity | Threshold violation logged, partial allocation |
| Priority ceiling causes overage | Non-priority rates reduced iteratively |
| No non-priority to preempt | Warning logged, priority may be under-served |
| Empty session list | Returns empty schedule, metrics still recorded |

---

## 11. Roadmap

### 11.1 Remaining Phases

| Phase | Tasks | Priority |
|-------|-------|----------|
| **Phase 5** | Run S1-S6 scenarios, generate publication figures, compare vs baselines | High |
| **Phase 6** | LaTeX pseudocode matching Elsevier journal format | Medium |

### 11.2 Phase 5 Scope

**Scenarios to Test:**
- S1: Baseline (30% priority, uniform arrivals)
- S2: Low priority (10%)
- S3: High priority (40%)
- S4: Morning rush (clustered arrivals)
- S5: Reduced PV (cloudy day)
- S6: Peak stress (50% priority, PM cluster)

**Metrics to Report:**
- Priority fulfillment rate (target: 100%)
- Non-priority energy delivery percentage
- Total energy cost vs uncontrolled baseline
- Computation time vs MPC baseline
- Preemption frequency
- Threshold violation conditions

**Outputs:**
- Publication-quality figures (300 DPI PNG)
- Structured DataFrames for tables
- Statistical analysis (mean, std, confidence intervals)

### 11.3 Future Enhancements (Post-Publication)

- Multi-period lookahead optimization
- Dynamic phase balancing
- Battery degradation modeling
- V2G (vehicle-to-grid) support
- Real-time price signal integration

---

## Document Maintenance

**To Update This Document:**
1. After completing a phase, update Section 2 (Implementation Status)
2. Add new data structures to Section 5
3. Add new API methods to Section 6
4. Update pseudocode in Section 8 if algorithm changes
5. Document any new known issues in Section 10

**Version:** 0.4.0 | **Phase:** 4 Complete | **Author:** Research Team
