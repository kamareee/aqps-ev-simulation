# Adaptive Queuing Priority Scheduler (AQPS)

A computationally efficient, real-time deployable scheduler for priority-aware EV fleet charging.

## Overview

AQPS provides an **O(n)** simple partitioning approach as an alternative to MPC-based methods while maintaining:
- ✅ Priority guarantees (minimum charging rates)
- ✅ Cost optimization (TOU-aware allocation - Phase 2)
- ✅ Infrastructure constraint compliance
- ✅ Real-time performance without commercial solvers

### Current Status: Phase 1 Complete

**Implemented Features:**
- Two-tier priority queue structure (partitioned by is_priority flag)
- Simple partitioning (no laxity-based sorting)
- Guaranteed minimum rates for priority EVs
- Fair-share allocation for non-priority EVs
- Basic capacity constraint checking
- Comprehensive metrics collection
- S1-S6 scenario generation

**Coming in Phase 2-6:**
- Preemption policies (highest-laxity-first)
- TOU cost optimization
- Renewable energy integration
- Quantization for discrete pilot signals
- Advanced infrastructure constraint handling

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

# 1. Create configuration
config = AQPSConfig(
    min_priority_rate=11.0,    # Minimum rate for priority EVs (Amps)
    total_capacity=150.0,      # Total available capacity (Amps)
    period_minutes=5.0,        # Scheduling period length
    voltage=220.0              # Network voltage
)

# 2. Initialize scheduler
scheduler = AdaptiveQueuingPriorityScheduler(config)

# 3. Generate test scenario
sessions = generate_scenario('S1', n_sessions=100, seed=42)

# 4. Run scheduler
schedule = scheduler.schedule(sessions, current_time=0)

# 5. Examine results
metrics = scheduler.get_current_metrics()
print(f"Priority capacity: {metrics.priority_allocated_capacity:.1f}A")
print(f"Utilization: {metrics.capacity_utilization:.1f}%")
```

## Architecture

### Core Components

```
aqps/
├── scheduler.py              # Main AQPS algorithm
├── queue_manager.py          # Two-tier queue management
├── data_structures.py        # SessionInfo, configs, metrics
├── scenario_generator.py     # S1-S6 test scenarios
└── utils.py                  # Laxity calculation, helpers
```

### Key Classes

**AdaptiveQueuingPriorityScheduler**
- Main scheduling algorithm
- Phase 1: Basic priority allocation
- Future: Preemption, TOU optimization

**QueueManager**
- Partition sessions by priority status
- Calculate laxity for each session
- Sort queues (lowest laxity first)

**SessionInfo**
- EV session data structure
- Contains: arrival/departure, energy, rates, priority flag

**ScenarioGenerator**
- Generate S1-S6 test scenarios
- Automated priority selection
- Configurable arrival patterns

## Algorithm Overview

### Phase 1 Implementation

```
1. Partition sessions into priority/non-priority queues by is_priority flag
2. Allocate minimum rates to all priority EVs
3. Maximize priority rates within available capacity
4. Fair-share remaining capacity to non-priority EVs
5. Return schedule: Dict[station_id → rate]
```

### Session Processing

Sessions are processed in two tiers:
- **Priority Tier**: All sessions with `is_priority=True`
- **Non-Priority Tier**: All sessions with `is_priority=False`

Within each tier, sessions are processed in queue order (no specific sorting by laxity or urgency).

## Scenarios

Six predefined scenarios for testing:

| Scenario | Priority % | Arrival Pattern | Description |
|----------|-----------|----------------|-------------|
| **S1** | 27% | Uniform | Baseline scenario |
| **S2** | 10% | Uniform | Low priority demand |
| **S3** | 50% | Uniform | High priority demand |
| **S4** | 27% | Clustered AM | Morning rush |
| **S5** | 27% | Uniform | Cloudy day (reduced PV) |
| **S6** | 50% | Clustered PM | Peak stress test |

## Examples

### Basic Usage

```bash
python examples/basic_usage.py
```

Output:
```
AQPS Basic Usage Example
========================
1. Creating scheduler configuration...
   Config: 11.0A min, 150.0A total
2. Initializing AQPS scheduler...
...
```

### Scenario Comparison

```bash
python examples/scenario_comparison.py
```

Compares performance across all 6 scenarios.

## Testing

Run the test suite:

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_scheduler.py

# Run with coverage
python -m pytest tests/ --cov=src/aqps
```

Test coverage includes:
- Laxity calculation (edge cases, time progression)
- Queue management (partitioning, sorting, statistics)
- Scheduler (priority allocation, capacity constraints, metrics)
- Scenario generation

## Configuration

### AQPSConfig Parameters

```python
AQPSConfig(
    min_priority_rate=11.0,        # Minimum rate for priority EVs (A)
    total_capacity=150.0,          # Total charging capacity (A)
    period_minutes=5.0,            # Scheduling period length (min)
    voltage=220.0,                 # Network voltage (V)
    enable_logging=True,           # Enable detailed logs
    max_priority_ratio=0.30        # Max priority session ratio
)
```

### PriorityConfig Parameters

```python
PriorityConfig(
    max_priority_pct=0.27,         # Target priority percentage
    min_energy_kwh=10.0,           # Min energy for priority
    max_energy_kwh=30.0,           # Max energy for priority
    min_duration_hours=2.0,        # Min parking duration
    high_energy_threshold=25.0,    # High-demand threshold
    high_energy_min_duration=3.0,  # Min duration for high-demand
    max_high_energy_pct=0.06       # Max high-demand ratio
)
```

## Metrics

Each scheduling cycle collects:

```python
SchedulingMetrics(
    timestamp=0,                          # Current time period
    priority_sessions_active=27,          # Active priority EVs
    non_priority_sessions_active=73,      # Active non-priority EVs
    total_allocated_capacity=145.3,       # Total allocated (A)
    priority_allocated_capacity=98.1,     # Priority allocated (A)
    priority_sessions_at_min=5,           # EVs at minimum rate
    priority_sessions_at_max=22,          # EVs at maximum rate
    warnings=['...']                      # Any warnings/errors
)
```

## Performance

**Computational Complexity:**
- MPC-based AQPC: O(n³) with solver dependency
- AQPS (Phase 1): O(n) - simple partitioning and linear allocation
- Expected speedup: 100-1000x for typical fleet sizes

**Expected Results:**
- Priority fulfillment: 100% (when below threshold)
- Cost vs baseline: -5% to -8% (Phase 2)
- Non-priority fulfillment: 75-80%

## Research Context

AQPS is developed as part of research on computationally efficient EV fleet charging optimization. It demonstrates that:

1. Priority guarantees can be achieved with simple partitioning (no optimization or sorting required)
2. Two-tier queue structure is sufficient for priority management
3. Real-time deployment is feasible with O(n) complexity
4. Open-source alternatives to commercial solvers are viable

### Publications

*Coming soon - journal submission in progress*

## Roadmap

### Phase 2: Preemption Logic (Weeks 1-2)
- [ ] Implement highest-laxity-first preemption
- [ ] Implement proportional fallback
- [ ] Add threshold tracking

### Phase 3: TOU Optimization (Week 2)
- [ ] TOU schedule integration
- [ ] Departure-aware deferral logic
- [ ] Cost metrics

### Phase 4: Quantization & Polish (Weeks 2-3)
- [ ] Discrete pilot signal support
- [ ] Performance benchmarking vs MPC
- [ ] Publication-ready visualizations

### Phase 5: Simulation Studies (Weeks 3-4)
- [ ] Comprehensive S1-S6 validation
- [ ] Scalability analysis
- [ ] Cost-priority trade-off analysis

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file

## Citation

```bibtex
@software{aqps2025,
  title={AQPS: Adaptive Queuing Priority Scheduler for EV Fleet Charging},
  author={Research Team},
  year={2025},
  url={https://github.com/yourusername/aqps-scheduler}
}
```

## Contact

For questions or collaboration:
- Email: [your-email@domain.com]
- Issues: [GitHub Issues](https://github.com/yourusername/aqps-scheduler/issues)

---

**Status:** Phase 1 Complete ✅ | Phase 2 In Progress 🚧
