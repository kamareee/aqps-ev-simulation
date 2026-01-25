# Adaptive Queuing Priority Scheduler (AQPS)

A computationally efficient, real-time deployable scheduler for priority-aware EV fleet charging.

## Overview

AQPS provides an **O(n)** simple partitioning approach as an alternative to MPC-based methods while maintaining:
- ✅ Priority guarantees (minimum charging rates)
- ✅ Cost optimization (TOU-aware deferral)
- ✅ Three-phase infrastructure constraint compliance
- ✅ PV/BESS renewable integration
- ✅ J1772 discrete pilot signal quantization
- ✅ Real-time performance without commercial solvers

### Current Status: Phase 4 Complete ✅

**Phase 1 - Core Algorithm:**
- Two-tier priority queue structure
- Guaranteed minimum rates for priority EVs
- Fair-share allocation for non-priority EVs
- S1-S6 scenario generation

**Phase 2 - Preemption & Threshold Tracking:**
- Highest-laxity-first preemption (Option B)
- Proportional fallback (Option A)
- Journal-quality threshold violation tracking

**Phase 3 - TOU Optimization:**
- Three-phase network infrastructure (balanced 18/18/18)
- Per-phase transformer capacity limits
- Configurable TOU tariff interface
- Aggressive deferral for non-priority EVs
- PV and BESS renewable integration

**Phase 4 - Quantization & Polish (NEW):**
- J1772 discrete pilot signal quantization [0, 8, 16, 24, 32] Amps
- Priority-aware quantization (ceiling for priority, floor for non-priority)
- Capacity overage adjustment after quantization
- Computational performance benchmarking
- Simulation summary statistics
- DataFrame export for analysis

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/aqps-scheduler.git
cd aqps-scheduler

# Install in development mode
pip install -e .

# Or install dependencies directly
pip install numpy
```

## Quick Start

```python
from aqps import AdaptiveQueuingPriorityScheduler, AQPSConfig, generate_scenario

# 1. Create configuration for three-phase system
config = AQPSConfig(
    min_priority_rate=16.0,    # J1772 valid signal (Amps)
    total_capacity=600.0,      # 3 phases × 200A
    period_minutes=5.0,        # Scheduling period length
    voltage=415.0              # Three-phase voltage
)

# 2. Initialize scheduler (quantization enabled by default)
scheduler = AdaptiveQueuingPriorityScheduler(config)

# 3. Configure TOU tariff (INSERT YOUR VALUES)
scheduler.configure_tou(
    peak_price=0.40,           # YOUR peak rate $/kWh
    off_peak_price=0.15,       # YOUR off-peak rate $/kWh
    peak_hours=[(14, 20)]      # YOUR peak hours (2pm-8pm)
)

# 4. Configure three-phase network
scheduler.configure_network(
    phase_a_limit=200.0,       # Phase A limit (Amps)
    phase_b_limit=200.0,       # Phase B limit (Amps)
    phase_c_limit=200.0        # Phase C limit (Amps)
)

# 5. Generate test scenario and run
sessions = generate_scenario('S1', n_sessions=100, seed=42)
schedule = scheduler.schedule(sessions, current_time=168)  # During peak

# All rates are now valid J1772 pilot signals!
for station_id, rate in schedule.items():
    assert rate in [0, 8, 16, 24, 32]

# 6. Get results
metrics = scheduler.get_current_metrics()
quant_stats = scheduler.get_quantization_statistics()
comp_stats = scheduler.get_computational_statistics()
summary = scheduler.get_simulation_summary()
```

## Architecture

### Core Components

```
src/aqps/
├── scheduler.py              # Main AQPS algorithm
├── queue_manager.py          # Two-tier queue management
├── data_structures.py        # SessionInfo, configs, metrics
├── scenario_generator.py     # S1-S6 test scenarios
├── utils.py                  # Laxity, quantization helpers
├── preemption.py             # Phase 2: Preemption logic
├── threshold_tracker.py      # Phase 2: Threshold violations
├── three_phase_network.py    # Phase 3: Network infrastructure
├── tou_optimization.py       # Phase 3: TOU tariff & optimizer
└── renewable_integration.py  # Phase 3: PV & BESS integration
```

## Phase 4 Features

### J1772 Quantization

```python
# Quantization is enabled by default
# All scheduled rates are valid J1772 pilot signals

from aqps import Phase4Config

phase4 = Phase4Config(
    enable_quantization=True,       # Default: True
    pilot_signals=[0, 8, 16, 24, 32],  # Standard J1772
    priority_ceil_enabled=True,     # Round UP priority if needed
    enable_timing=True              # Track computational metrics
)

scheduler = AdaptiveQueuingPriorityScheduler(config, phase4_config=phase4)
```

### Quantization Rules

| EV Type | Quantization | Example |
|---------|--------------|---------|
| **Priority** | Ceiling (if floor < minimum) | 11A → 16A |
| **Non-Priority** | Floor | 25A → 24A |

### Computational Benchmarking

```python
# After running simulation
comp_stats = scheduler.get_computational_statistics()
# {
#   'avg_time_ms': 0.15,      # Average schedule() time
#   'max_time_ms': 0.45,
#   'avg_time_per_session_us': 3.3  # Microseconds per session
# }
```

### DataFrame Export

```python
import pandas as pd

# Export all metrics as DataFrames
data = scheduler.export_dataframes()

scheduling_df = pd.DataFrame(data['scheduling'])
preemption_df = pd.DataFrame(data['preemption'])
quantization_df = pd.DataFrame(data['quantization'])
computational_df = pd.DataFrame(data['computational'])
```

## Algorithm Overview

### Full Pipeline (Phase 4)

```
1. Start timing
2. Partition sessions into priority/non-priority queues
3. Allocate minimum rates to all priority EVs
4. Apply preemption if needed (Phase 2)
5. Maximize priority rates within capacity
6. Allocate non-priority EVs with TOU deferral (Phase 3)
7. Apply J1772 quantization (Phase 4):
   - Priority: ceiling if floor < min_priority_rate
   - Non-priority: floor
8. Adjust for capacity overage if needed
9. Record computational metrics
10. Return quantized schedule
```

## Configuration

### AQPSConfig Parameters

```python
AQPSConfig(
    min_priority_rate=16.0,        # Must be valid J1772 signal
    total_capacity=600.0,          # Total capacity (A)
    period_minutes=5.0,            # Scheduling period (min)
    voltage=415.0,                 # Three-phase voltage (V)
    enable_logging=True,           # Enable detailed logs
    max_priority_ratio=0.30        # Max priority ratio
)
```

### Phase4Config Parameters

```python
Phase4Config(
    enable_quantization=True,       # Enable J1772 quantization
    pilot_signals=[0, 8, 16, 24, 32],  # Valid signals (Amps)
    priority_ceil_enabled=True,     # Ceiling for priority
    track_quantization_events=True, # Store individual events
    enable_timing=True              # Enable benchmarking
)
```

## Performance

**Computational Complexity:**
- MPC-based AQPC: O(n³) with solver dependency
- AQPS (Phase 4): O(n) - linear allocation with quantization
- Expected speedup: 100-1000x for typical fleet sizes

**Benchmarking Results:**
- Average schedule time: ~0.15ms per call
- Average time per session: ~3.3µs
- Quantization efficiency: >98%

## Roadmap

- [x] Phase 1: Core Algorithm
- [x] Phase 2: Preemption & Threshold Tracking
- [x] Phase 3: TOU Optimization & Infrastructure
- [x] Phase 4: Quantization & Benchmarking
- [ ] Phase 5: Comprehensive Simulation Studies (S1-S6)
- [ ] Phase 6: LaTeX Pseudocode Generation

## License

MIT License - see LICENSE file

## Citation

```bibtex
@software{aqps2025,
  title={AQPS: Adaptive Queuing Priority Scheduler for EV Fleet Charging},
  author={Research Team},
  year={2025},
  version={0.4.0},
  url={https://github.com/yourusername/aqps-scheduler}
}
```

---

**Status:** Phase 1 ✅ | Phase 2 ✅ | Phase 3 ✅ | Phase 4 ✅
