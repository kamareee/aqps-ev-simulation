"""
Unit tests for AdaptiveQueuingPriorityScheduler.
"""

import unittest
from src.aqps.scheduler import AdaptiveQueuingPriorityScheduler
from src.aqps.data_structures import SessionInfo, AQPSConfig


class TestAQPSScheduler(unittest.TestCase):
    """Test main AQPS scheduler."""
    
    def setUp(self):
        """Set up test scheduler."""
        self.config = AQPSConfig(
            min_priority_rate=11.0,
            total_capacity=150.0,
            period_minutes=5.0,
            voltage=220.0,
            enable_logging=False  # Disable for cleaner test output
        )
        self.scheduler = AdaptiveQueuingPriorityScheduler(self.config)
    
    def test_empty_schedule(self):
        """Test scheduling with no sessions."""
        schedule = self.scheduler.schedule([], current_time=0)
        self.assertEqual(schedule, {})
    
    def test_single_priority_session(self):
        """Test scheduling with one priority session."""
        session = SessionInfo(
            session_id="EV1",
            station_id="S1",
            arrival_time=0,
            departure_time=100,
            energy_requested=20.0,
            max_rate=32.0,
            min_rate=6.0,
            is_priority=True
        )
        
        schedule = self.scheduler.schedule([session], current_time=0)
        
        # Should get at least minimum priority rate
        self.assertIn("S1", schedule)
        self.assertGreaterEqual(schedule["S1"], self.config.min_priority_rate)
    
    def test_priority_gets_minimum_rate(self):
        """Test that priority EVs get minimum guaranteed rate."""
        sessions = [
            SessionInfo(
                session_id="EV_P1",
                station_id="S1",
                arrival_time=0,
                departure_time=100,
                energy_requested=20.0,
                max_rate=32.0,
                is_priority=True
            ),
            SessionInfo(
                session_id="EV_P2",
                station_id="S2",
                arrival_time=0,
                departure_time=100,
                energy_requested=20.0,
                max_rate=32.0,
                is_priority=True
            ),
        ]
        
        schedule = self.scheduler.schedule(sessions, current_time=0)
        
        # Both priority EVs should get at least minimum rate
        self.assertGreaterEqual(schedule["S1"], self.config.min_priority_rate)
        self.assertGreaterEqual(schedule["S2"], self.config.min_priority_rate)
    
    def test_capacity_constraint(self):
        """Test that total allocation doesn't exceed capacity."""
        sessions = [
            SessionInfo(
                session_id=f"EV{i}",
                station_id=f"S{i}",
                arrival_time=0,
                departure_time=100,
                energy_requested=20.0,
                max_rate=32.0,
                is_priority=(i < 3)  # First 3 are priority
            )
            for i in range(10)
        ]
        
        schedule = self.scheduler.schedule(sessions, current_time=0)
        
        total_allocated = sum(schedule.values())
        self.assertLessEqual(total_allocated, self.config.total_capacity + 0.1)
    
    def test_priority_maximization(self):
        """Test that priority EVs are maximized when capacity available."""
        # Create scenario with plenty of capacity
        config = AQPSConfig(
            min_priority_rate=11.0,
            total_capacity=200.0,  # Plenty of capacity
            enable_logging=False
        )
        scheduler = AdaptiveQueuingPriorityScheduler(config)
        
        session = SessionInfo(
            session_id="EV1",
            station_id="S1",
            arrival_time=0,
            departure_time=100,
            energy_requested=20.0,
            max_rate=32.0,
            min_rate=6.0,
            is_priority=True
        )
        
        schedule = scheduler.schedule([session], current_time=0)
        
        # With plenty of capacity, should get max rate
        self.assertGreaterEqual(schedule["S1"], 30.0)
    
    def test_non_priority_allocation(self):
        """Test that non-priority EVs get fair share of remaining capacity."""
        sessions = [
            SessionInfo(
                session_id="EV_P1",
                station_id="S1",
                arrival_time=0,
                departure_time=100,
                energy_requested=20.0,
                max_rate=32.0,
                is_priority=True
            ),
            SessionInfo(
                session_id="EV_N1",
                station_id="S2",
                arrival_time=0,
                departure_time=100,
                energy_requested=20.0,
                max_rate=32.0,
                is_priority=False
            ),
            SessionInfo(
                session_id="EV_N2",
                station_id="S3",
                arrival_time=0,
                departure_time=100,
                energy_requested=20.0,
                max_rate=32.0,
                is_priority=False
            ),
        ]
        
        schedule = self.scheduler.schedule(sessions, current_time=0)
        
        # Priority should get rate
        self.assertGreater(schedule["S1"], 0)
        
        # Non-priority should get roughly equal rates
        # (allowing for rounding differences)
        self.assertGreater(schedule["S2"], 0)
        self.assertGreater(schedule["S3"], 0)
        self.assertAlmostEqual(schedule["S2"], schedule["S3"], delta=1.0)
    
    def test_metrics_collection(self):
        """Test that metrics are collected."""
        session = SessionInfo(
            session_id="EV1",
            station_id="S1",
            arrival_time=0,
            departure_time=100,
            energy_requested=20.0,
            max_rate=32.0,
            is_priority=True
        )
        
        self.scheduler.schedule([session], current_time=0)
        
        metrics = self.scheduler.get_current_metrics()
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics.timestamp, 0)
        self.assertEqual(metrics.priority_sessions_active, 1)
        self.assertEqual(metrics.non_priority_sessions_active, 0)
    
    def test_threshold_warning(self):
        """Test warning when priority demand exceeds capacity."""
        # Create many priority EVs that exceed capacity
        sessions = [
            SessionInfo(
                session_id=f"EV_P{i}",
                station_id=f"S{i}",
                arrival_time=0,
                departure_time=100,
                energy_requested=20.0,
                max_rate=32.0,
                min_rate=6.0,
                is_priority=True
            )
            for i in range(20)  # 20 priority EVs * 11A = 220A > 150A capacity
        ]
        
        schedule = self.scheduler.schedule(sessions, current_time=0)
        
        # Should generate warning
        metrics = self.scheduler.get_current_metrics()
        self.assertGreater(len(metrics.warnings), 0)


if __name__ == '__main__':
    unittest.main()
