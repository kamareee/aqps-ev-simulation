# Phase 4: Quantization & Polish - Implementation Summary

## Overview

Phase 4 adds J1772 pilot signal quantization and computational performance benchmarking to the AQPS scheduler. This enables realistic discrete charging rates and provides timing data to validate O(n log n) complexity claims.

## Key Components Implemented

### 1. J1772 Pilot Signal Quantization

**Standard Pilot Signals:** `[0, 8, 16, 24, 32]` Amps

**Quantization Rules:**
- **Non-Priority EVs**: Always use floor quantization (round down to nearest valid signal)
- **Priority EVs**: Use ceiling quantization if floor would violate minimum guarantee

**Example:**
```python
# Priority EV allocated 11A (min_priority_rate=11A)
# Floor: 11A → 8A (violates guarantee!)
# Ceiling: 11A → 16A (maintains guarantee)
# Result: Priority EV gets 16A
```

### 2. Capacity Overage Adjustment

When priority ceiling quantization causes total allocation to exceed capacity, the scheduler iteratively reduces non-priority rates:

1. Sort non-priority EVs by current rate (highest first)
2. Reduce highest-rate EV by one pilot signal step (e.g., 32A → 24A)
3. Repeat until capacity constraint is satisfied
4. Log warning if adjustment impossible

### 3. Phase4Config

```python
from aqps import Phase4Config

phase4 = Phase4Config(
    enable_quantization=True,       # Enable J1772 quantization (default: True)
    pilot_signals=[0.0, 8.0, 16.0, 24.0, 32.0],  # Valid signals
    priority_ceil_enabled=True,     # Round UP priority if needed
    track_quantization_events=True, # Store individual events
    enable_timing=True              # Enable computational metrics
)
```

### 4. Quantization Metrics

```python
from aqps import QuantizationMetrics

# After scheduling:
quant_stats = scheduler.get_quantization_statistics()
# Returns:
# {
#   'enabled': True,
#   'total_timesteps': 288,
#   'avg_efficiency': 98.5,  # % of capacity retained
#   'total_capacity_lost': 450.0,  # Amps lost to floor
#   'total_capacity_gained': 120.0,  # Amps gained from ceiling
#   'total_priority_ceil': 45,  # Number of ceiling events
#   'total_energy_loss_kwh': 15.2,
#   'pilot_signals': [0.0, 8.0, 16.0, 24.0, 32.0]
# }
```

### 5. Computational Metrics

```python
from aqps import ComputationalMetrics

# After scheduling:
comp_stats = scheduler.get_computational_statistics()
# Returns:
# {
#   'enabled': True,
#   'total_timesteps': 288,
#   'avg_time_ms': 0.15,      # Average schedule() time
#   'max_time_ms': 0.45,      # Maximum time
#   'min_time_ms': 0.08,      # Minimum time
#   'total_time_ms': 43.2,
#   'avg_sessions': 45.3,     # Average sessions per call
#   'avg_time_per_session_us': 3.3  # Microseconds per session
# }
```

### 6. Simulation Summary

```python
from aqps import SimulationSummary

# Aggregate statistics across entire run:
summary = scheduler.get_simulation_summary()
# Returns SimulationSummary with:
# - total_timesteps
# - priority_fulfillment_rate
# - total_preemptions
# - total_threshold_violations
# - total_deferrals
# - avg_schedule_time_ms
# - quantization_efficiency_avg
```

### 7. DataFrame Export

```python
# Export all metrics for analysis
data = scheduler.export_dataframes()

# Returns dict with keys:
# - 'scheduling': List of scheduling metrics dicts
# - 'preemption': List of preemption event dicts
# - 'threshold_violations': List of threshold violation dicts
# - 'tou_deferrals': List of TOU deferral dicts
# - 'quantization': List of quantization metrics dicts
# - 'computational': List of computational metrics dicts

# Use with pandas:
import pandas as pd
scheduling_df = pd.DataFrame(data['scheduling'])
quantization_df = pd.DataFrame(data['quantization'])
```

## Usage Example

```python
from aqps import (
    AdaptiveQueuingPriorityScheduler,
    AQPSConfig,
    Phase4Config,
    generate_scenario
)

# Configure scheduler
config = AQPSConfig(
    min_priority_rate=16.0,  # J1772 valid signal
    total_capacity=600.0,
    voltage=415.0
)

phase4 = Phase4Config(
    enable_quantization=True,
    priority_ceil_enabled=True,
    enable_timing=True
)

scheduler = AdaptiveQueuingPriorityScheduler(config, phase4_config=phase4)

# Configure TOU and network
scheduler.configure_tou(peak_price=0.40, off_peak_price=0.15, peak_hours=[(14, 20)])
scheduler.configure_network(phase_a_limit=200.0, phase_b_limit=200.0, phase_c_limit=200.0)

# Run simulation
sessions = generate_scenario('S1', n_sessions=100, seed=42)
for t in range(288):  # 24 hours at 5-min intervals
    active = [s for s in sessions if s.arrival_time <= t < s.departure_time]
    schedule = scheduler.schedule(active, current_time=t)
    
    # All rates are now valid J1772 pilot signals
    for station_id, rate in schedule.items():
        assert rate in [0, 8, 16, 24, 32]

# Analyze results
print(f"Quantization: {scheduler.get_quantization_statistics()}")
print(f"Computational: {scheduler.get_computational_statistics()}")
print(f"Summary: {scheduler.get_simulation_summary()}")
```

## Key Design Decisions

| Decision | Implementation |
|----------|----------------|
| Default quantization | **Enabled** by default |
| Priority quantization | **Ceiling** if floor violates guarantee |
| Non-priority quantization | Always **floor** |
| Capacity overage | Iteratively reduce non-priority EVs |
| Timing scope | Overall `schedule()` time only |
| Export format | Separate DataFrames per metric type |

## Data Structures

### QuantizationEvent
Records individual rate adjustments:
- `session_id`, `station_id`
- `pre_quantization_rate`, `post_quantization_rate`
- `quantization_method` ('floor' or 'ceil')
- `reason`

### QuantizationMetrics
Per-timestep quantization statistics:
- `pre_quantization_capacity`, `post_quantization_capacity`
- `capacity_lost`, `capacity_gained`
- `priority_ceil_count`
- `energy_loss_kwh`

### ComputationalMetrics
Per-timestep timing data:
- `schedule_time_ms`
- `num_sessions`, `num_priority`, `num_non_priority`
- `time_per_session_us`

### SimulationSummary
Aggregate statistics:
- Fulfillment rates
- Total events (preemptions, violations, deferrals)
- Timing statistics
- Quantization efficiency

## Utility Functions

```python
from aqps import (
    quantize_rate_floor,
    quantize_rate_ceil,
    quantize_schedule_floor,
    calculate_quantization_loss,
    metrics_to_dataframe_data,
    export_to_csv_string,
    calculate_summary_statistics
)

# Quantize individual rate
quantize_rate_floor(25.5)  # → 24.0
quantize_rate_ceil(11.0)   # → 16.0

# Calculate loss from quantization
loss = calculate_quantization_loss(pre_schedule, post_schedule)
```

## File Structure

```
src/aqps/
├── __init__.py                  # Updated with Phase 4 exports
├── data_structures.py           # Added Phase 4 structures
├── scheduler.py                 # Added quantization, timing, summary
├── utils.py                     # Added quantization utilities
├── three_phase_network.py       # Phase 3 - unchanged
├── tou_optimization.py          # Phase 3 - unchanged
├── renewable_integration.py     # Phase 3 - unchanged
├── preemption.py                # Phase 2 - unchanged
├── queue_manager.py             # Phase 2 - unchanged
├── threshold_tracker.py         # Phase 2 - unchanged
└── scenario_generator.py        # Phase 1 - unchanged
```

## API Reference

### Scheduler Methods (Phase 4)

| Method | Description |
|--------|-------------|
| `get_quantization_statistics()` | Get quantization performance stats |
| `get_quantization_history()` | Get all quantization metrics |
| `get_computational_statistics()` | Get timing statistics |
| `get_computational_history()` | Get all computational metrics |
| `get_simulation_summary()` | Get aggregate simulation summary |
| `export_dataframes()` | Export all metrics as DataFrame-ready dicts |

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_quantization` | `True` | Enable J1772 quantization |
| `pilot_signals` | `[0, 8, 16, 24, 32]` | Valid pilot signals |
| `priority_ceil_enabled` | `True` | Round UP priority if needed |
| `track_quantization_events` | `True` | Store individual events |
| `enable_timing` | `True` | Enable computational metrics |

## Version

- **AQPS Version**: 0.4.0
- **Phase**: 4 (Quantization & Polish)
- **Author**: Research Team
