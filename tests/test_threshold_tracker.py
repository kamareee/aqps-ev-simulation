"""
Tests for Phase 2: Threshold Tracker functionality.

This module tests the threshold tracking mechanism including:
- Timestep snapshot recording
- Threshold violation detection
- Statistics calculation
- Data export for visualization
"""

import pytest
import sys
sys.path.insert(0, './src')

from aqps import (
    AdaptiveQueuingPriorityScheduler,
    AQPSConfig,
    SessionInfo,
    PriorityQueueEntry,
    ThresholdTracker,
    TimestepSnapshot,
)


class TestThresholdTracker:
    """Tests for the ThresholdTracker class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tracker = ThresholdTracker(
            total_capacity=100.0,
            min_priority_rate=11.0,
            enable_snapshots=True
        )
    
    def create_queue_entry(
        self,
        session_id: str,
        station_id: str,
        laxity: float,
        is_priority: bool = False
    ) -> PriorityQueueEntry:
        """Helper to create queue entries."""
        session = SessionInfo(
            session_id=session_id,
            station_id=station_id,
            arrival_time=0,
            departure_time=100,
            energy_requested=20.0,
            max_rate=32.0,
            min_rate=0.0,
            is_priority=is_priority
        )
        return PriorityQueueEntry(
            session=session,
            laxity=laxity,
            is_priority=is_priority,
            actual_rate=0.0
        )
    
    def test_record_timestep_no_violation(self):
        """Test recording a timestep without violation."""
        priority_queue = [
            self.create_queue_entry("P1", "S1", laxity=10.0, is_priority=True),
            self.create_queue_entry("P2", "S2", laxity=15.0, is_priority=True),
        ]
        non_priority_queue = [
            self.create_queue_entry("NP1", "S3", laxity=20.0),
        ]
        schedule = {"S1": 11.0, "S2": 11.0, "S3": 20.0}
        
        violation = self.tracker.record_timestep(
            timestamp=0,
            priority_queue=priority_queue,
            non_priority_queue=non_priority_queue,
            schedule=schedule,
            preemption_occurred=False
        )
        
        assert violation is None
        assert len(self.tracker.snapshots) == 1
        assert len(self.tracker.violations) == 0
    
    def test_record_timestep_with_violation(self):
        """Test recording a timestep with threshold violation."""
        # 10 priority EVs × 11A = 110A, but capacity is only 100A
        priority_queue = [
            self.create_queue_entry(f"P{i}", f"S{i}", laxity=10.0, is_priority=True)
            for i in range(10)
        ]
        non_priority_queue = []
        schedule = {f"S{i}": 10.0 for i in range(10)}  # Can only give 10A each
        
        violation = self.tracker.record_timestep(
            timestamp=0,
            priority_queue=priority_queue,
            non_priority_queue=non_priority_queue,
            schedule=schedule,
            preemption_occurred=True
        )
        
        assert violation is not None
        assert violation.capacity_shortfall == 10.0  # 110A needed - 100A available
        assert violation.priority_count == 10
        assert len(self.tracker.violations) == 1
    
    def test_snapshot_contains_correct_data(self):
        """Test that snapshots contain all required data."""
        priority_queue = [
            self.create_queue_entry("P1", "S1", laxity=10.0, is_priority=True),
        ]
        non_priority_queue = [
            self.create_queue_entry("NP1", "S2", laxity=20.0),
            self.create_queue_entry("NP2", "S3", laxity=25.0),
        ]
        schedule = {"S1": 15.0, "S2": 20.0, "S3": 20.0}
        
        self.tracker.record_timestep(
            timestamp=5,
            priority_queue=priority_queue,
            non_priority_queue=non_priority_queue,
            schedule=schedule,
            preemption_occurred=True,
            preemption_count=2
        )
        
        snapshot = self.tracker.snapshots[0]
        assert snapshot.timestamp == 5
        assert snapshot.priority_count == 1
        assert snapshot.non_priority_count == 2
        assert snapshot.total_sessions == 3
        assert snapshot.priority_allocated == 15.0
        assert snapshot.non_priority_allocated == 40.0
        assert snapshot.preemption_occurred is True
        assert snapshot.preemption_count == 2
    
    def test_get_summary_statistics(self):
        """Test summary statistics calculation."""
        # Record multiple timesteps
        for t in range(10):
            priority_queue = [
                self.create_queue_entry(f"P{i}", f"SP{i}", laxity=10.0, is_priority=True)
                for i in range(3)
            ]
            non_priority_queue = [
                self.create_queue_entry(f"NP{i}", f"SNP{i}", laxity=20.0)
                for i in range(5)
            ]
            schedule = {f"SP{i}": 11.0 for i in range(3)}
            schedule.update({f"SNP{i}": 10.0 for i in range(5)})
            
            self.tracker.record_timestep(
                timestamp=t,
                priority_queue=priority_queue,
                non_priority_queue=non_priority_queue,
                schedule=schedule
            )
        
        stats = self.tracker.get_summary_statistics()
        
        assert stats['total_timesteps'] == 10
        assert stats['total_violations'] == 0
        assert stats['violation_rate'] == 0.0
        assert 'avg_priority_count' in stats
        assert 'avg_utilization' in stats
    
    def test_get_critical_threshold(self):
        """Test critical threshold calculation."""
        # Record some violations
        for priority_count in [5, 8, 10, 12]:  # Increasing priority counts
            priority_queue = [
                self.create_queue_entry(f"P{i}", f"S{i}", laxity=10.0, is_priority=True)
                for i in range(priority_count)
            ]
            
            # Violation occurs when priority_count × 11A > 100A (i.e., > 9 EVs)
            schedule = {f"S{i}": 11.0 for i in range(min(priority_count, 9))}
            
            self.tracker.record_timestep(
                timestamp=priority_count,
                priority_queue=priority_queue,
                non_priority_queue=[],
                schedule=schedule
            )
        
        threshold = self.tracker.get_critical_threshold()
        
        # Should find the minimum priority ratio where violation occurred
        if threshold is not None:
            assert threshold > 0
    
    def test_violation_probability_by_ratio(self):
        """Test violation probability calculation."""
        # Record timesteps with varying priority ratios
        for t in range(20):
            priority_count = t % 10 + 1
            priority_queue = [
                self.create_queue_entry(f"P{i}", f"SP{i}", laxity=10.0, is_priority=True)
                for i in range(priority_count)
            ]
            non_priority_queue = [
                self.create_queue_entry(f"NP{i}", f"SNP{i}", laxity=20.0)
                for i in range(10 - priority_count)
            ]
            
            schedule = {f"SP{i}": 11.0 for i in range(priority_count)}
            schedule.update({f"SNP{i}": 5.0 for i in range(10 - priority_count)})
            
            self.tracker.record_timestep(
                timestamp=t,
                priority_queue=priority_queue,
                non_priority_queue=non_priority_queue,
                schedule=schedule
            )
        
        probabilities = self.tracker.get_violation_probability_by_ratio()
        
        assert isinstance(probabilities, dict)
        # All keys should be between 0 and 1
        for ratio in probabilities.keys():
            assert 0 <= ratio <= 1
    
    def test_export_for_visualization(self):
        """Test export for visualization."""
        priority_queue = [
            self.create_queue_entry("P1", "S1", laxity=10.0, is_priority=True)
        ]
        schedule = {"S1": 11.0}
        
        self.tracker.record_timestep(
            timestamp=0,
            priority_queue=priority_queue,
            non_priority_queue=[],
            schedule=schedule
        )
        
        export = self.tracker.export_for_visualization()
        
        assert 'summary' in export
        assert 'snapshots' in export
        assert 'violations' in export
        assert 'distributions' in export
        assert 'config' in export
        assert export['config']['total_capacity'] == 100.0
    
    def test_time_series_data(self):
        """Test time series data extraction."""
        for t in range(5):
            priority_queue = [
                self.create_queue_entry("P1", "S1", laxity=10.0, is_priority=True)
            ]
            schedule = {"S1": 11.0}
            
            self.tracker.record_timestep(
                timestamp=t,
                priority_queue=priority_queue,
                non_priority_queue=[],
                schedule=schedule
            )
        
        time_series = self.tracker.get_time_series_data()
        
        assert 'timestamp' in time_series
        assert 'priority_count' in time_series
        assert 'utilization_pct' in time_series
        assert len(time_series['timestamp']) == 5
    
    def test_safe_operating_limit(self):
        """Test safe operating limit calculation."""
        limits = self.tracker.calculate_safe_operating_limit()
        
        # With 100A capacity and 11A minimum rate
        assert limits['theoretical_max_priority_count'] == 9  # 100/11 = 9.09
        assert limits['recommended_buffer_ratio'] == 0.9
        assert limits['recommended_max_priority'] == 8  # 9 × 0.9 = 8.1 → 8
    
    def test_reset(self):
        """Test tracker reset."""
        # Record some data
        priority_queue = [
            self.create_queue_entry("P1", "S1", laxity=10.0, is_priority=True)
        ]
        schedule = {"S1": 11.0}
        
        self.tracker.record_timestep(
            timestamp=0,
            priority_queue=priority_queue,
            non_priority_queue=[],
            schedule=schedule
        )
        
        assert len(self.tracker.snapshots) > 0
        
        self.tracker.reset()
        
        assert len(self.tracker.snapshots) == 0
        assert len(self.tracker.violations) == 0
        assert self.tracker._total_timesteps == 0


class TestTimestepSnapshot:
    """Tests for the TimestepSnapshot dataclass."""
    
    def test_snapshot_properties(self):
        """Test snapshot computed properties."""
        snapshot = TimestepSnapshot(
            timestamp=10,
            priority_count=3,
            non_priority_count=5,
            total_capacity=100.0,
            priority_demand=33.0,
            priority_allocated=33.0,
            non_priority_allocated=50.0,
            utilization_pct=83.0,
            priority_ratio=0.375,
            headroom=17.0,
        )
        
        assert snapshot.total_sessions == 8
        assert snapshot.total_allocated == 83.0
    
    def test_snapshot_to_dict(self):
        """Test snapshot dictionary conversion."""
        snapshot = TimestepSnapshot(
            timestamp=5,
            priority_count=2,
            non_priority_count=3,
            total_capacity=100.0,
            priority_demand=22.0,
            priority_allocated=22.0,
            non_priority_allocated=30.0,
            utilization_pct=52.0,
            priority_ratio=0.4,
            headroom=48.0,
        )
        
        d = snapshot.to_dict()
        
        assert d['timestamp'] == 5
        assert d['priority_count'] == 2
        assert d['total_sessions'] == 5
        assert d['utilization_pct'] == 52.0


class TestThresholdTrackerIntegration:
    """Test threshold tracker integration with scheduler."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = AQPSConfig(
            min_priority_rate=11.0,
            total_capacity=50.0,  # Low capacity
            period_minutes=5.0,
            voltage=220.0
        )
        self.scheduler = AdaptiveQueuingPriorityScheduler(self.config)
    
    def create_sessions(self, n_priority: int, n_non_priority: int) -> list:
        """Create test sessions."""
        sessions = []
        
        for i in range(n_priority):
            sessions.append(SessionInfo(
                session_id=f"P{i}",
                station_id=f"SP{i}",
                arrival_time=0,
                departure_time=100,
                energy_requested=20.0,
                max_rate=32.0,
                is_priority=True
            ))
        
        for i in range(n_non_priority):
            sessions.append(SessionInfo(
                session_id=f"NP{i}",
                station_id=f"SNP{i}",
                arrival_time=0,
                departure_time=100,
                energy_requested=20.0,
                max_rate=32.0,
                is_priority=False
            ))
        
        return sessions
    
    def test_scheduler_records_to_tracker(self):
        """Test that scheduler records to threshold tracker."""
        sessions = self.create_sessions(n_priority=2, n_non_priority=2)
        
        self.scheduler.schedule(sessions, current_time=0)
        
        stats = self.scheduler.get_threshold_statistics()
        assert stats['total_timesteps'] == 1
    
    def test_scheduler_detects_violations(self):
        """Test that scheduler detects threshold violations."""
        # 5 priority EVs × 11A = 55A, but only 50A capacity
        sessions = self.create_sessions(n_priority=5, n_non_priority=0)
        
        self.scheduler.schedule(sessions, current_time=0)
        
        violations = self.scheduler.get_threshold_violations()
        assert len(violations) > 0
        assert violations[0].capacity_shortfall == 5.0  # 55 - 50 = 5A
    
    def test_multiple_timesteps_tracking(self):
        """Test tracking across multiple timesteps."""
        for t in range(5):
            sessions = self.create_sessions(n_priority=2, n_non_priority=3)
            self.scheduler.schedule(sessions, current_time=t)
        
        stats = self.scheduler.get_threshold_statistics()
        assert stats['total_timesteps'] == 5
    
    def test_export_analysis_includes_threshold(self):
        """Test that export includes threshold data."""
        sessions = self.create_sessions(n_priority=2, n_non_priority=2)
        self.scheduler.schedule(sessions, current_time=0)
        
        data = self.scheduler.export_analysis_data()
        
        assert 'threshold' in data
        assert 'summary' in data['threshold']
        assert 'snapshots' in data['threshold']
    
    def test_reset_all_tracking(self):
        """Test that reset clears all tracking data."""
        sessions = self.create_sessions(n_priority=2, n_non_priority=2)
        self.scheduler.schedule(sessions, current_time=0)
        
        assert len(self.scheduler.metrics_history) > 0
        
        self.scheduler.reset_all_tracking()
        
        assert len(self.scheduler.metrics_history) == 0
        stats = self.scheduler.get_threshold_statistics()
        assert stats['total_timesteps'] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
