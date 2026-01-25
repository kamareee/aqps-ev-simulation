"""
Threshold tracking for Adaptive Queuing Priority Scheduler (AQPS).

This module tracks when the system cannot guarantee priority EV fulfillment,
collecting comprehensive data for journal-quality analysis and visualization.

Key Metrics Tracked:
- Threshold violation events (when/why priority guarantees fail)
- Capacity utilization over time
- Priority ratio trends
- System stress indicators
- Cumulative impact statistics

Visualization Support:
- Time-series plots (utilization, priority ratio over time)
- Violation event scatter plots
- Capacity headroom analysis
- Priority ratio vs. violation probability
- Heatmaps of system stress

Author: Research Team
Phase: 2 (Queue Management & Preemption)
"""

import logging
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from .data_structures import (
    PriorityQueueEntry,
    ThresholdViolationEvent,
    SchedulingMetrics
)

logger = logging.getLogger(__name__)


@dataclass
class TimestepSnapshot:
    """
    Complete system state snapshot at a single timestep.
    
    Used for time-series analysis and visualization.
    
    Attributes:
        timestamp: Simulation time (period)
        priority_count: Number of active priority EVs
        non_priority_count: Number of active non-priority EVs
        total_capacity: System capacity (Amps)
        priority_demand: Capacity needed for priority minimums (Amps)
        priority_allocated: Capacity actually allocated to priority (Amps)
        non_priority_allocated: Capacity allocated to non-priority (Amps)
        utilization_pct: Overall capacity utilization percentage
        priority_ratio: Ratio of priority to total sessions
        headroom: Remaining capacity after priority allocation (Amps)
        threshold_violated: Whether a violation occurred this timestep
        preemption_occurred: Whether preemption occurred this timestep
        preemption_count: Number of preemptions this timestep
    """
    timestamp: int
    priority_count: int = 0
    non_priority_count: int = 0
    total_capacity: float = 0.0
    priority_demand: float = 0.0
    priority_allocated: float = 0.0
    non_priority_allocated: float = 0.0
    utilization_pct: float = 0.0
    priority_ratio: float = 0.0
    headroom: float = 0.0
    threshold_violated: bool = False
    preemption_occurred: bool = False
    preemption_count: int = 0
    
    @property
    def total_sessions(self) -> int:
        return self.priority_count + self.non_priority_count
    
    @property
    def total_allocated(self) -> float:
        return self.priority_allocated + self.non_priority_allocated
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for DataFrame/JSON export."""
        return {
            'timestamp': self.timestamp,
            'priority_count': self.priority_count,
            'non_priority_count': self.non_priority_count,
            'total_sessions': self.total_sessions,
            'total_capacity': self.total_capacity,
            'priority_demand': self.priority_demand,
            'priority_allocated': self.priority_allocated,
            'non_priority_allocated': self.non_priority_allocated,
            'total_allocated': self.total_allocated,
            'utilization_pct': self.utilization_pct,
            'priority_ratio': self.priority_ratio,
            'headroom': self.headroom,
            'threshold_violated': self.threshold_violated,
            'preemption_occurred': self.preemption_occurred,
            'preemption_count': self.preemption_count,
        }


class ThresholdTracker:
    """
    Comprehensive threshold and system state tracker for AQPS.
    
    This class collects detailed data about system operation, threshold
    violations, and capacity utilization for journal-quality analysis.
    
    Data Collection Categories:
    1. Time-series snapshots (every scheduling period)
    2. Threshold violation events (when priority guarantees fail)
    3. Cumulative statistics
    4. Distribution data for histograms
    
    Visualization Support:
    - Time-series plots (utilization, priority ratio over time)
    - Violation event scatter plots
    - Capacity headroom analysis
    - Priority ratio vs. violation probability
    - Heatmaps of system stress
    
    Usage:
        tracker = ThresholdTracker(total_capacity=150.0, min_priority_rate=11.0)
        
        # Record each timestep
        violation = tracker.record_timestep(
            timestamp=t,
            priority_queue=priority_q,
            non_priority_queue=non_priority_q,
            schedule=schedule,
            preemption_occurred=True
        )
        
        # After simulation, export data
        data = tracker.export_for_visualization()
    
    Attributes:
        violations: List of ThresholdViolationEvent objects
        snapshots: List of TimestepSnapshot objects
        total_capacity: Total system capacity (Amps)
        min_priority_rate: Minimum rate for priority EVs (Amps)
    """
    
    def __init__(
        self,
        total_capacity: float = 150.0,
        min_priority_rate: float = 11.0,
        enable_snapshots: bool = True
    ):
        """
        Initialize the threshold tracker.
        
        Args:
            total_capacity: Total system capacity (Amps)
            min_priority_rate: Minimum rate for priority EVs (Amps)
            enable_snapshots: Whether to record per-timestep snapshots
        """
        self.total_capacity = total_capacity
        self.min_priority_rate = min_priority_rate
        self.enable_snapshots = enable_snapshots
        
        # Event storage
        self.violations: List[ThresholdViolationEvent] = []
        self.snapshots: List[TimestepSnapshot] = []
        
        # Running statistics
        self._total_timesteps = 0
        self._total_priority_sessions = 0
        self._total_non_priority_sessions = 0
        self._total_preemptions = 0
        
        # Distribution tracking for histograms
        self._priority_ratio_bins: Dict[float, int] = {}
        self._utilization_bins: Dict[float, int] = {}
        self._headroom_when_violated: List[float] = []
        self._priority_count_when_violated: List[int] = []
    
    def record_timestep(
        self,
        timestamp: int,
        priority_queue: List[PriorityQueueEntry],
        non_priority_queue: List[PriorityQueueEntry],
        schedule: Dict[str, float],
        preemption_occurred: bool = False,
        preemption_count: int = 0
    ) -> Optional[ThresholdViolationEvent]:
        """
        Record system state for a single timestep.
        
        This is the main entry point called by the scheduler after each
        scheduling cycle. It records a snapshot and checks for violations.
        
        Args:
            timestamp: Current simulation time (period)
            priority_queue: List of priority queue entries
            non_priority_queue: List of non-priority queue entries
            schedule: Final schedule (station_id → rate)
            preemption_occurred: Whether preemption happened this timestep
            preemption_count: Number of preemptions this timestep
        
        Returns:
            ThresholdViolationEvent if violation detected, None otherwise
        """
        self._total_timesteps += 1
        
        # Calculate statistics
        priority_count = len(priority_queue)
        non_priority_count = len(non_priority_queue)
        total_sessions = priority_count + non_priority_count
        
        self._total_priority_sessions += priority_count
        self._total_non_priority_sessions += non_priority_count
        if preemption_occurred:
            self._total_preemptions += 1
        
        # Calculate capacity metrics
        priority_demand = priority_count * self.min_priority_rate
        
        priority_allocated = sum(
            schedule.get(e.session.station_id, 0.0)
            for e in priority_queue
        )
        non_priority_allocated = sum(
            schedule.get(e.session.station_id, 0.0)
            for e in non_priority_queue
        )
        total_allocated = priority_allocated + non_priority_allocated
        
        utilization_pct = (total_allocated / self.total_capacity * 100) if self.total_capacity > 0 else 0.0
        priority_ratio = priority_count / total_sessions if total_sessions > 0 else 0.0
        headroom = self.total_capacity - total_allocated
        
        # Track distributions (bin to nearest 5%)
        self._update_distribution(
            self._priority_ratio_bins,
            round(priority_ratio * 20) / 20  # Round to nearest 5%
        )
        self._update_distribution(
            self._utilization_bins,
            round(utilization_pct / 5) * 5  # Round to nearest 5%
        )
        
        # Check for threshold violation
        violation = None
        threshold_violated = False
        
        if priority_demand > self.total_capacity:
            threshold_violated = True
            shortfall = priority_demand - self.total_capacity
            
            # Find affected priority EVs (those below minimum rate)
            affected_ids = [
                e.session.session_id
                for e in priority_queue
                if schedule.get(e.session.station_id, 0.0) < self.min_priority_rate - 0.1
            ]
            
            violation = ThresholdViolationEvent(
                timestamp=timestamp,
                priority_count=priority_count,
                non_priority_count=non_priority_count,
                capacity_required=priority_demand,
                capacity_available=self.total_capacity,
                capacity_shortfall=shortfall,
                affected_session_ids=affected_ids,
                system_utilization=utilization_pct,
                preemption_attempted=preemption_occurred
            )
            
            self.violations.append(violation)
            self._headroom_when_violated.append(headroom)
            self._priority_count_when_violated.append(priority_count)
            
            logger.warning(
                f"Threshold violation at t={timestamp}: "
                f"{priority_count} priority EVs need {priority_demand:.1f}A, "
                f"only {self.total_capacity:.1f}A available"
            )
        
        # Record snapshot if enabled
        if self.enable_snapshots:
            snapshot = TimestepSnapshot(
                timestamp=timestamp,
                priority_count=priority_count,
                non_priority_count=non_priority_count,
                total_capacity=self.total_capacity,
                priority_demand=priority_demand,
                priority_allocated=priority_allocated,
                non_priority_allocated=non_priority_allocated,
                utilization_pct=utilization_pct,
                priority_ratio=priority_ratio,
                headroom=headroom,
                threshold_violated=threshold_violated,
                preemption_occurred=preemption_occurred,
                preemption_count=preemption_count
            )
            self.snapshots.append(snapshot)
        
        return violation
    
    def _update_distribution(self, dist: Dict[float, int], value: float) -> None:
        """Update a distribution dictionary with a new value."""
        dist[value] = dist.get(value, 0) + 1
    
    def get_summary_statistics(self) -> Dict:
        """
        Get comprehensive summary statistics for the simulation.
        
        Returns:
            Dictionary with summary statistics suitable for paper results
        """
        if self._total_timesteps == 0:
            return {'error': 'No data recorded'}
        
        # Basic counts
        stats = {
            'total_timesteps': self._total_timesteps,
            'total_violations': len(self.violations),
            'violation_rate': len(self.violations) / self._total_timesteps,
            'total_preemptions': self._total_preemptions,
            'preemption_rate': self._total_preemptions / self._total_timesteps,
        }
        
        # Priority ratio statistics
        if self._total_timesteps > 0:
            avg_priority = self._total_priority_sessions / self._total_timesteps
            avg_non_priority = self._total_non_priority_sessions / self._total_timesteps
            avg_total = avg_priority + avg_non_priority
            stats['avg_priority_count'] = avg_priority
            stats['avg_non_priority_count'] = avg_non_priority
            stats['avg_priority_ratio'] = avg_priority / avg_total if avg_total > 0 else 0.0
        
        # Violation statistics
        if self.violations:
            shortfalls = [v.capacity_shortfall for v in self.violations]
            stats['avg_shortfall'] = sum(shortfalls) / len(shortfalls)
            stats['max_shortfall'] = max(shortfalls)
            stats['min_shortfall'] = min(shortfalls)
            
            # Severity breakdown
            severity_counts = {'minor': 0, 'moderate': 0, 'severe': 0}
            for v in self.violations:
                severity_counts[v.severity] += 1
            stats['violation_severity'] = severity_counts
        
        # Snapshot-based statistics
        if self.snapshots:
            utils = [s.utilization_pct for s in self.snapshots]
            stats['avg_utilization'] = sum(utils) / len(utils)
            stats['max_utilization'] = max(utils)
            stats['min_utilization'] = min(utils)
            
            headrooms = [s.headroom for s in self.snapshots]
            stats['avg_headroom'] = sum(headrooms) / len(headrooms)
            stats['min_headroom'] = min(headrooms)
        
        return stats
    
    def get_critical_threshold(self) -> Optional[float]:
        """
        Calculate the critical priority ratio where violations begin.
        
        This identifies the "tipping point" for the system.
        
        Returns:
            Critical priority ratio (float) or None if no violations
        """
        if not self.violations:
            return None
        
        # Find the minimum priority ratio at which a violation occurred
        min_ratio = min(v.priority_ratio for v in self.violations)
        return min_ratio
    
    def get_violation_probability_by_ratio(self) -> Dict[float, float]:
        """
        Calculate violation probability for each priority ratio bin.
        
        Returns:
            Dict mapping priority_ratio → violation_probability
        """
        if not self.snapshots:
            return {}
        
        # Count violations and total for each ratio bin
        ratio_totals: Dict[float, int] = {}
        ratio_violations: Dict[float, int] = {}
        
        for snapshot in self.snapshots:
            ratio_bin = round(snapshot.priority_ratio * 20) / 20  # 5% bins
            ratio_totals[ratio_bin] = ratio_totals.get(ratio_bin, 0) + 1
            if snapshot.threshold_violated:
                ratio_violations[ratio_bin] = ratio_violations.get(ratio_bin, 0) + 1
        
        # Calculate probabilities
        probabilities = {}
        for ratio, total in ratio_totals.items():
            violations = ratio_violations.get(ratio, 0)
            probabilities[ratio] = violations / total
        
        return dict(sorted(probabilities.items()))
    
    def export_for_visualization(self) -> Dict:
        """
        Export all data in format suitable for visualization.
        
        Returns comprehensive data structure for matplotlib/seaborn plotting.
        
        Returns:
            Dictionary with separate keys for different visualization needs
        """
        return {
            'summary': self.get_summary_statistics(),
            'snapshots': [s.to_dict() for s in self.snapshots],
            'violations': [v.to_dict() for v in self.violations],
            'distributions': {
                'priority_ratio': dict(sorted(self._priority_ratio_bins.items())),
                'utilization': dict(sorted(self._utilization_bins.items())),
            },
            'critical_threshold': self.get_critical_threshold(),
            'violation_probability_by_ratio': self.get_violation_probability_by_ratio(),
            'config': {
                'total_capacity': self.total_capacity,
                'min_priority_rate': self.min_priority_rate,
            }
        }
    
    def export_to_json(self, filepath: str) -> None:
        """
        Export all tracking data to a JSON file.
        
        Args:
            filepath: Path to output JSON file
        """
        data = self.export_for_visualization()
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Exported threshold tracking data to {filepath}")
    
    def get_snapshots_dataframe_data(self) -> List[Dict]:
        """
        Get snapshot data as list of dicts for DataFrame creation.
        
        Returns:
            List of dictionaries suitable for pandas DataFrame
        """
        return [s.to_dict() for s in self.snapshots]
    
    def get_violations_dataframe_data(self) -> List[Dict]:
        """
        Get violation data as list of dicts for DataFrame creation.
        
        Returns:
            List of dictionaries suitable for pandas DataFrame
        """
        return [v.to_dict() for v in self.violations]
    
    def calculate_safe_operating_limit(self) -> Dict:
        """
        Calculate the safe operating limits for the system.
        
        Based on observed data, determine:
        - Maximum safe priority count
        - Maximum safe priority ratio
        - Recommended capacity buffer
        
        Returns:
            Dictionary with safe operating parameters
        """
        max_safe_priority = int(self.total_capacity / self.min_priority_rate)
        max_safe_ratio = 0.0
        
        if self.snapshots:
            # Find highest ratio without violation
            safe_ratios = [
                s.priority_ratio for s in self.snapshots
                if not s.threshold_violated
            ]
            if safe_ratios:
                max_safe_ratio = max(safe_ratios)
        
        return {
            'theoretical_max_priority_count': max_safe_priority,
            'observed_max_safe_ratio': max_safe_ratio,
            'recommended_buffer_ratio': 0.9,  # 10% safety margin
            'recommended_max_priority': int(max_safe_priority * 0.9),
        }
    
    def get_time_series_data(self) -> Dict[str, List]:
        """
        Get time-series data organized by metric.
        
        Returns:
            Dict with metric names as keys and lists of values
        """
        if not self.snapshots:
            return {}
        
        return {
            'timestamp': [s.timestamp for s in self.snapshots],
            'priority_count': [s.priority_count for s in self.snapshots],
            'non_priority_count': [s.non_priority_count for s in self.snapshots],
            'total_sessions': [s.total_sessions for s in self.snapshots],
            'utilization_pct': [s.utilization_pct for s in self.snapshots],
            'priority_ratio': [s.priority_ratio for s in self.snapshots],
            'headroom': [s.headroom for s in self.snapshots],
            'threshold_violated': [s.threshold_violated for s in self.snapshots],
            'preemption_occurred': [s.preemption_occurred for s in self.snapshots],
        }
    
    def reset(self) -> None:
        """Reset all tracking data."""
        self.violations.clear()
        self.snapshots.clear()
        self._total_timesteps = 0
        self._total_priority_sessions = 0
        self._total_non_priority_sessions = 0
        self._total_preemptions = 0
        self._priority_ratio_bins.clear()
        self._utilization_bins.clear()
        self._headroom_when_violated.clear()
        self._priority_count_when_violated.clear()
        logger.info("Threshold tracker reset")
    
    def __repr__(self) -> str:
        return (
            f"ThresholdTracker("
            f"timesteps={self._total_timesteps}, "
            f"violations={len(self.violations)}, "
            f"capacity={self.total_capacity}A)"
        )
