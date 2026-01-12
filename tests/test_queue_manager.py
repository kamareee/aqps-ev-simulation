"""
Unit tests for QueueManager.
"""

import unittest
from src.aqps.data_structures import SessionInfo
from src.aqps.queue_manager import QueueManager


class TestQueueManager(unittest.TestCase):
    """Test queue management functionality."""
    
    def setUp(self):
        """Set up test data."""
        self.queue_manager = QueueManager(voltage=220.0, period_minutes=5.0)
        
        # Create test sessions
        self.sessions = [
            SessionInfo(
                session_id="EV1",
                station_id="S1",
                arrival_time=0,
                departure_time=100,
                energy_requested=20.0,
                max_rate=32.0,
                is_priority=True
            ),
            SessionInfo(
                session_id="EV2",
                station_id="S2",
                arrival_time=0,
                departure_time=50,
                energy_requested=30.0,
                max_rate=32.0,
                is_priority=False
            ),
            SessionInfo(
                session_id="EV3",
                station_id="S3",
                arrival_time=0,
                departure_time=150,
                energy_requested=15.0,
                max_rate=32.0,
                is_priority=True
            ),
            SessionInfo(
                session_id="EV4",
                station_id="S4",
                arrival_time=0,
                departure_time=80,
                energy_requested=25.0,
                max_rate=32.0,
                is_priority=False
            ),
        ]
    
    def test_partition_sessions(self):
        """Test partitioning into priority and non-priority."""
        priority, non_priority = self.queue_manager.partition_sessions(self.sessions)
        
        self.assertEqual(len(priority), 2)
        self.assertEqual(len(non_priority), 2)
        
        priority_ids = {s.session_id for s in priority}
        self.assertEqual(priority_ids, {"EV1", "EV3"})
        
        non_priority_ids = {s.session_id for s in non_priority}
        self.assertEqual(non_priority_ids, {"EV2", "EV4"})
    
    def test_partition_and_sort(self):
        """Test partition by priority flag (no laxity sorting)."""
        priority_queue, non_priority_queue = self.queue_manager.partition_and_sort(
            self.sessions, current_time=0
        )
        
        # Check counts
        self.assertEqual(len(priority_queue), 2)
        self.assertEqual(len(non_priority_queue), 2)
        
        # Check that priority sessions are correctly identified
        priority_ids = {e.session.session_id for e in priority_queue}
        self.assertEqual(priority_ids, {"EV1", "EV3"})
        
        non_priority_ids = {e.session.session_id for e in non_priority_queue}
        self.assertEqual(non_priority_ids, {"EV2", "EV4"})
        
        # Check that all entries are PriorityQueueEntry objects
        for entry in priority_queue + non_priority_queue:
            self.assertIsNotNone(entry.laxity)  # Laxity calculated but not used for sorting
    
    def test_queue_statistics(self):
        """Test queue statistics calculation."""
        priority_queue, non_priority_queue = self.queue_manager.partition_and_sort(
            self.sessions, current_time=0
        )
        
        stats = self.queue_manager.get_queue_statistics(
            priority_queue, non_priority_queue
        )
        
        self.assertEqual(stats['priority_count'], 2)
        self.assertEqual(stats['non_priority_count'], 2)
        self.assertEqual(stats['total_count'], 4)
        self.assertIn('priority_min_laxity', stats)
        self.assertIn('priority_avg_laxity', stats)
    
    def test_filter_feasible_sessions(self):
        """Test filtering sessions by laxity threshold."""
        # Create a session with negative laxity
        infeasible_session = SessionInfo(
            session_id="EV_BAD",
            station_id="S5",
            arrival_time=0,
            departure_time=5,  # Very short time
            energy_requested=100.0,  # Too much energy
            max_rate=32.0,
            is_priority=False
        )
        
        test_sessions = self.sessions + [infeasible_session]
        
        feasible = self.queue_manager.filter_feasible_sessions(
            test_sessions, current_time=0, min_laxity=0.0
        )
        
        # Should filter out the infeasible session
        feasible_ids = {s.session_id for s in feasible}
        self.assertNotIn("EV_BAD", feasible_ids)
        self.assertGreaterEqual(len(feasible), 4)
    
    def test_empty_sessions(self):
        """Test handling empty session list."""
        priority, non_priority = self.queue_manager.partition_and_sort([], 0)
        
        self.assertEqual(len(priority), 0)
        self.assertEqual(len(non_priority), 0)


if __name__ == '__main__':
    unittest.main()
