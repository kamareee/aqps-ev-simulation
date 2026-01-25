"""
Tests for Phase 2: Preemption functionality.

This module tests the preemption mechanism including:
- Option B (highest-laxity-first) preemption
- Option A (proportional) fallback
- PreemptionEvent recording
- Integration with scheduler
"""

import pytest
import sys
sys.path.insert(0, './src')

from aqps import (
    AdaptiveQueuingPriorityScheduler,
    AQPSConfig,
    SessionInfo,
    PriorityQueueEntry,
    PreemptionManager,
    PreemptionMethod,
)
from aqps.utils import calculate_laxity


class TestPreemptionManager:
    """Tests for the PreemptionManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = PreemptionManager(min_priority_rate=11.0)
    
    def create_queue_entry(
        self,
        session_id: str,
        station_id: str,
        laxity: float,
        is_priority: bool = False,
        max_rate: float = 32.0
    ) -> PriorityQueueEntry:
        """Helper to create queue entries."""
        session = SessionInfo(
            session_id=session_id,
            station_id=station_id,
            arrival_time=0,
            departure_time=100,
            energy_requested=20.0,
            max_rate=max_rate,
            min_rate=0.0,
            is_priority=is_priority
        )
        return PriorityQueueEntry(
            session=session,
            laxity=laxity,
            is_priority=is_priority,
            actual_rate=0.0
        )
    
    def test_check_preemption_not_needed_sufficient_capacity(self):
        """Test that preemption is not needed when capacity is sufficient."""
        priority = self.create_queue_entry("P1", "S1", laxity=5.0, is_priority=True)
        
        needed, shortfall = self.manager.check_preemption_needed(
            priority_entry=priority,
            current_allocation=0.0,
            min_required_rate=11.0,
            remaining_capacity=20.0
        )
        
        assert needed is False
        assert shortfall == 0.0
    
    def test_check_preemption_not_needed_already_met(self):
        """Test that preemption is not needed when minimum is already met."""
        priority = self.create_queue_entry("P1", "S1", laxity=5.0, is_priority=True)
        
        needed, shortfall = self.manager.check_preemption_needed(
            priority_entry=priority,
            current_allocation=11.0,  # Already at minimum
            min_required_rate=11.0,
            remaining_capacity=0.0
        )
        
        assert needed is False
        assert shortfall == 0.0
    
    def test_check_preemption_needed_insufficient_capacity(self):
        """Test that preemption is needed when capacity is insufficient."""
        priority = self.create_queue_entry("P1", "S1", laxity=5.0, is_priority=True)
        
        needed, shortfall = self.manager.check_preemption_needed(
            priority_entry=priority,
            current_allocation=0.0,
            min_required_rate=11.0,
            remaining_capacity=5.0  # Only 5A available, need 11A
        )
        
        assert needed is True
        assert shortfall == 6.0  # 11 - 5 = 6A shortfall
    
    def test_option_b_preemption_highest_laxity_first(self):
        """Test that Option B preempts highest laxity EVs first."""
        # Create non-priority entries with different laxities
        np1 = self.create_queue_entry("NP1", "S1", laxity=5.0)
        np2 = self.create_queue_entry("NP2", "S2", laxity=20.0)  # Highest laxity
        np3 = self.create_queue_entry("NP3", "S3", laxity=10.0)
        
        # Give them initial allocations
        schedule = {"S1": 15.0, "S2": 15.0, "S3": 15.0}
        np1.actual_rate = 15.0
        np2.actual_rate = 15.0
        np3.actual_rate = 15.0
        
        # Priority EV needs capacity
        priority = self.create_queue_entry("P1", "S4", laxity=2.0, is_priority=True)
        
        # Execute preemption - need 10A
        schedule, event = self.manager.execute_preemption(
            needed_capacity=10.0,
            priority_entry=priority,
            non_priority_queue=[np1, np2, np3],
            schedule=schedule,
            current_time=10
        )
        
        # NP2 should be preempted first (highest laxity = 20)
        assert "NP2" in event.preempted_session_ids
        assert event.method == PreemptionMethod.HIGHEST_LAXITY
        assert event.capacity_freed >= 10.0
        assert event.success is True
    
    def test_option_b_stops_when_sufficient(self):
        """Test that Option B stops when enough capacity is freed."""
        # Create multiple non-priority entries
        entries = [
            self.create_queue_entry(f"NP{i}", f"S{i}", laxity=float(i * 10))
            for i in range(5)
        ]
        
        schedule = {f"S{i}": 20.0 for i in range(5)}
        for e in entries:
            e.actual_rate = 20.0
        
        priority = self.create_queue_entry("P1", "S10", laxity=1.0, is_priority=True)
        
        # Need only 15A - should only preempt 1 EV (highest laxity = NP4 with 40)
        schedule, event = self.manager.execute_preemption(
            needed_capacity=15.0,
            priority_entry=priority,
            non_priority_queue=entries,
            schedule=schedule,
            current_time=10
        )
        
        assert event.capacity_freed >= 15.0
        assert event.num_victims == 1  # Should stop after freeing enough
        assert "NP4" in event.preempted_session_ids  # Highest laxity
    
    def test_option_a_proportional_fallback(self):
        """Test Option A proportional reduction when Option B insufficient."""
        # Create entries with low rates (Option B can't free enough from one)
        entries = [
            self.create_queue_entry(f"NP{i}", f"S{i}", laxity=float(i))
            for i in range(3)
        ]
        
        # Give each 10A
        schedule = {f"S{i}": 10.0 for i in range(3)}
        for e in entries:
            e.actual_rate = 10.0
        
        priority = self.create_queue_entry("P1", "S10", laxity=0.5, is_priority=True)
        
        # Need 25A - more than any single EV has, will need proportional
        schedule, event = self.manager.execute_preemption(
            needed_capacity=25.0,
            priority_entry=priority,
            non_priority_queue=entries,
            schedule=schedule,
            current_time=10
        )
        
        # Should use combined method (Option B then A)
        assert event.method == PreemptionMethod.COMBINED
        assert event.num_victims == 3  # All EVs affected
        assert event.capacity_freed >= 25.0 or event.capacity_freed == 30.0  # All available
    
    def test_no_preemption_when_zero_capacity_needed(self):
        """Test that no preemption occurs when capacity is 0."""
        priority = self.create_queue_entry("P1", "S1", laxity=1.0, is_priority=True)
        
        schedule, event = self.manager.execute_preemption(
            needed_capacity=0.0,
            priority_entry=priority,
            non_priority_queue=[],
            schedule={},
            current_time=10
        )
        
        assert event.method == PreemptionMethod.NONE
        assert event.num_victims == 0
        assert event.success is True
    
    def test_preemption_statistics(self):
        """Test preemption statistics tracking."""
        entries = [
            self.create_queue_entry(f"NP{i}", f"S{i}", laxity=float(i * 10))
            for i in range(3)
        ]
        schedule = {f"S{i}": 15.0 for i in range(3)}
        for e in entries:
            e.actual_rate = 15.0
        
        priority = self.create_queue_entry("P1", "S10", laxity=1.0, is_priority=True)
        
        # Execute preemption
        self.manager.execute_preemption(
            needed_capacity=10.0,
            priority_entry=priority,
            non_priority_queue=entries,
            schedule=schedule,
            current_time=10
        )
        
        stats = self.manager.get_statistics()
        assert stats['total_preemptions'] == 1
        assert stats['success_rate'] == 1.0
        assert stats['total_capacity_freed'] >= 10.0
    
    def test_preemption_event_to_dict(self):
        """Test that PreemptionEvent converts to dict correctly."""
        entries = [self.create_queue_entry("NP1", "S1", laxity=10.0)]
        schedule = {"S1": 15.0}
        entries[0].actual_rate = 15.0
        
        priority = self.create_queue_entry("P1", "S10", laxity=1.0, is_priority=True)
        
        _, event = self.manager.execute_preemption(
            needed_capacity=10.0,
            priority_entry=priority,
            non_priority_queue=entries,
            schedule=schedule,
            current_time=10
        )
        
        event_dict = event.to_dict()
        assert 'timestamp' in event_dict
        assert 'priority_session_id' in event_dict
        assert 'capacity_freed' in event_dict
        assert 'method' in event_dict
    
    def test_preempt_single_highest_laxity(self):
        """Test single highest laxity preemption helper."""
        entries = [
            self.create_queue_entry("NP1", "S1", laxity=5.0),
            self.create_queue_entry("NP2", "S2", laxity=15.0),  # Highest
            self.create_queue_entry("NP3", "S3", laxity=10.0),
        ]
        schedule = {"S1": 10.0, "S2": 10.0, "S3": 10.0}
        
        session_id, freed = self.manager.preempt_single_highest_laxity(
            needed_capacity=8.0,
            non_priority_queue=entries,
            schedule=schedule,
            current_time=10
        )
        
        assert session_id == "NP2"  # Highest laxity
        assert freed == 8.0


class TestPreemptionIntegration:
    """Test preemption integration with the scheduler."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = AQPSConfig(
            min_priority_rate=11.0,
            total_capacity=50.0,  # Low capacity to trigger preemption
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
                min_rate=0.0,
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
                min_rate=0.0,
                is_priority=False
            ))
        
        return sessions
    
    def test_scheduler_triggers_preemption_when_needed(self):
        """Test that scheduler triggers preemption when priority EVs need capacity."""
        # Create 3 priority EVs needing 33A total minimum
        # With only 50A capacity, preemption will be needed
        sessions = self.create_sessions(n_priority=3, n_non_priority=3)
        
        schedule = self.scheduler.schedule(sessions, current_time=0)
        
        # Check that priority EVs got their minimum
        for i in range(3):
            assert schedule.get(f"SP{i}", 0) >= 11.0, f"Priority EV P{i} didn't get minimum rate"
    
    def test_scheduler_preemption_stats_accessible(self):
        """Test that preemption statistics are accessible."""
        sessions = self.create_sessions(n_priority=2, n_non_priority=2)
        
        self.scheduler.schedule(sessions, current_time=0)
        
        stats = self.scheduler.get_preemption_statistics()
        assert 'total_preemptions' in stats
        assert 'option_b_count' in stats
        assert 'option_a_count' in stats
    
    def test_export_analysis_data_includes_preemption(self):
        """Test that export includes preemption data."""
        sessions = self.create_sessions(n_priority=2, n_non_priority=2)
        
        self.scheduler.schedule(sessions, current_time=0)
        
        data = self.scheduler.export_analysis_data()
        assert 'preemption' in data
        assert 'preemption_events' in data
    
    def test_preemption_with_high_priority_ratio(self):
        """Test preemption behavior with high priority ratio."""
        # 4 priority EVs × 11A = 44A needed, only 50A total
        sessions = self.create_sessions(n_priority=4, n_non_priority=2)
        
        schedule = self.scheduler.schedule(sessions, current_time=0)
        
        # All priority EVs should get at least minimum
        priority_total = sum(schedule.get(f"SP{i}", 0) for i in range(4))
        assert priority_total >= 44.0  # 4 × 11A minimum


class TestEdgeCases:
    """Test edge cases in preemption."""
    
    def test_no_non_priority_to_preempt(self):
        """Test behavior when there are no non-priority EVs to preempt."""
        config = AQPSConfig(
            min_priority_rate=11.0,
            total_capacity=20.0,  # Very low capacity
            period_minutes=5.0,
            voltage=220.0
        )
        scheduler = AdaptiveQueuingPriorityScheduler(config)
        
        # Only priority EVs, more than capacity can handle
        sessions = [
            SessionInfo(
                session_id=f"P{i}",
                station_id=f"S{i}",
                arrival_time=0,
                departure_time=100,
                energy_requested=20.0,
                max_rate=32.0,
                is_priority=True
            )
            for i in range(3)  # 3 × 11A = 33A, but only 20A capacity
        ]
        
        schedule = scheduler.schedule(sessions, current_time=0)
        
        # Check threshold violation was recorded
        violations = scheduler.get_threshold_violations()
        assert len(violations) > 0
    
    def test_preemption_manager_reset(self):
        """Test that preemption manager resets correctly."""
        manager = PreemptionManager()
        
        # Create some events
        entry = PriorityQueueEntry(
            session=SessionInfo(
                session_id="NP1", station_id="S1",
                arrival_time=0, departure_time=100,
                energy_requested=20.0, max_rate=32.0
            ),
            laxity=10.0,
            is_priority=False
        )
        schedule = {"S1": 15.0}
        priority = PriorityQueueEntry(
            session=SessionInfo(
                session_id="P1", station_id="SP1",
                arrival_time=0, departure_time=100,
                energy_requested=20.0, max_rate=32.0,
                is_priority=True
            ),
            laxity=5.0,
            is_priority=True
        )
        
        manager.execute_preemption(10.0, priority, [entry], schedule, 0)
        
        assert manager.total_preemptions > 0
        
        manager.reset()
        
        assert manager.total_preemptions == 0
        assert len(manager.preemption_history) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
