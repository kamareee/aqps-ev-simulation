# AQPS Phase 2 Implementation Summary

## Version: 0.2.0

## Overview

Phase 2 implements the **Queue Management & Preemption** functionality for the Adaptive Queuing Priority Scheduler, adding the ability to reclaim capacity from non-priority EVs when priority EVs cannot meet their minimum charging rate.

## What's New in Phase 2

### 1. Preemption Mechanism

A two-stage preemption policy ensures priority EVs always get their guaranteed minimum rate:

**Option B (Primary): Highest-Laxity-First**
- Sort non-priority EVs by laxity in descending order
- Preempt EVs with highest laxity (most flexible) first
- Can reduce rates to zero (complete preemption allowed)
- Stop when sufficient capacity is freed

**Option A (Fallback): Proportional Reduction**
- Used when Option B doesn't free enough capacity
- Proportionally reduce ALL remaining non-priority EVs
- Each EV reduced by: `(shortfall / total_reducible) * reducible_amount`

### 2. Threshold Tracking

Comprehensive data collection for journal-quality analysis:

- **Time-series snapshots**: System state at each timestep
- **Violation events**: When priority guarantees cannot be met
- **Distribution tracking**: Priority ratio and utilization histograms
- **Critical threshold detection**: Where violations begin to occur
- **Export capabilities**: JSON and DataFrame-ready formats

### 3. New Data Structures

```python
# Preemption event tracking
class PreemptionEvent:
    timestamp: int
    priority_session_id: str
    preempted_session_ids: List[str]
    capacity_needed: float
    capacity_freed: float
    method: PreemptionMethod  # HIGHEST_LAXITY, PROPORTIONAL, COMBINED
    success: bool

# Threshold violation tracking
class ThresholdViolationEvent:
    timestamp: int
    priority_count: int
    capacity_shortfall: float
    severity: str  # "minor", "moderate", "severe"
    
# Timestep snapshots
class TimestepSnapshot:
    timestamp: int
    priority_count: int
    non_priority_count: int
    utilization_pct: float
    priority_ratio: float
    headroom: float
    threshold_violated: bool
    preemption_occurred: bool
```

## Files Changed/Added

### New Files
- `src/aqps/preemption.py` - PreemptionManager class
- `src/aqps/threshold_tracker.py` - ThresholdTracker and TimestepSnapshot classes
- `tests/test_preemption.py` - Comprehensive preemption tests
- `tests/test_threshold_tracker.py` - Threshold tracking tests

### Modified Files
- `src/aqps/data_structures.py` - Added PreemptionMethod, PreemptionEvent, ThresholdViolationEvent
- `src/aqps/scheduler.py` - Integrated preemption and threshold tracking
- `src/aqps/__init__.py` - Updated exports

## Usage Examples

### Basic Preemption

```python
from aqps import AdaptiveQueuingPriorityScheduler, AQPSConfig

config = AQPSConfig(
    min_priority_rate=11.0,
    total_capacity=100.0
)
scheduler = AdaptiveQueuingPriorityScheduler(config)

# Scheduling automatically triggers preemption when needed
schedule = scheduler.schedule(sessions, current_time=0)

# Check preemption statistics
stats = scheduler.get_preemption_statistics()
print(f"Total preemptions: {stats['total_preemptions']}")
print(f"Option B used: {stats['option_b_count']} times")
print(f"Option A fallback: {stats['option_a_count']} times")
```

### Threshold Analysis

```python
# Get threshold violations
violations = scheduler.get_threshold_violations()
for v in violations:
    print(f"t={v.timestamp}: {v.severity} violation, shortfall={v.capacity_shortfall}A")

# Find critical threshold
critical = scheduler.get_critical_threshold()
print(f"System fails at priority ratio: {critical:.1%}")

# Get time series for plotting
time_series = scheduler.threshold_tracker.get_time_series_data()
# Keys: timestamp, priority_count, utilization_pct, priority_ratio, headroom, etc.
```

### Export for Visualization

```python
# Export all analysis data
export = scheduler.export_analysis_data()
# Contains: scheduling_metrics, preemption, preemption_events, threshold, config

# Export threshold data for matplotlib/seaborn
viz_data = scheduler.threshold_tracker.export_for_visualization()
# Contains: summary, snapshots, violations, distributions, critical_threshold

# Export to JSON file
scheduler.threshold_tracker.export_to_json('analysis_results.json')
```

## Design Decisions

Based on user confirmation:

1. **Laxity Sorting (Option B)**: FIFO allocation within tiers, laxity only used for preemption victim selection
2. **Preemption Trigger (Option C)**: Both capacity AND minimum rate checks
3. **Minimum Rates (Option A)**: Complete preemption allowed (can reduce to 0)
4. **Preemption Logging (Option D)**: Part of simulation results, accessible via APIs
5. **Threshold Tracker**: Designed for journal-quality graphs with comprehensive data collection

## Test Results

All Phase 2 tests pass:

✓ PreemptionManager - Option B (highest laxity first)
✓ PreemptionManager - Multiple victims in laxity order
✓ Scheduler preemption integration
✓ Threshold violation detection
✓ Export analysis data
✓ Time series data collection
✓ Safe operating limits calculation
✓ Statistics tracking

## Next Phases

- **Phase 3**: TOU Cost Optimization (departure-aware deferral)
- **Phase 4**: Renewable Integration (PV, BESS)
- **Phase 5**: Quantization (discrete pilot signals)
- **Phase 6**: Comprehensive Simulation Studies

## Performance

- Computational complexity: O(n log n) for preemption (sorting by laxity)
- Memory: O(n) for snapshots, O(v) for violations
- No external solver dependencies
