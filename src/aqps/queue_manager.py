"""
Queue management for Adaptive Queuing Priority Scheduler (AQPS).

This module handles the two-tier queue structure, partitioning sessions
into priority and non-priority queues, and sorting by laxity.
"""

from typing import List, Tuple
from .data_structures import SessionInfo, PriorityQueueEntry
from .utils import calculate_laxity


class QueueManager:
    """
    Manages the two-tier queue structure for AQPS.
    
    The QueueManager is responsible for:
    1. Partitioning sessions into priority and non-priority queues
    2. Calculating laxity for each session
    3. Sorting queues by laxity (lowest first)
    4. Creating PriorityQueueEntry objects with metadata
    
    Attributes:
        voltage: Network voltage in Volts
        period_minutes: Length of each period in minutes
    """
    
    def __init__(self, voltage: float = 220.0, period_minutes: float = 5.0):
        """
        Initialize the queue manager.
        
        Args:
            voltage: Network voltage in Volts
            period_minutes: Length of each scheduling period in minutes
        """
        self.voltage = voltage
        self.period_minutes = period_minutes
    
    def partition_and_sort(
        self,
        sessions: List[SessionInfo],
        current_time: int
    ) -> Tuple[List[PriorityQueueEntry], List[PriorityQueueEntry]]:
        """
        Partition sessions by priority flag only.
        
        This is the main entry point for queue management. It performs:
        1. Partition into priority and non-priority based on is_priority flag
        2. Create queue entries (laxity calculated but not used for sorting)
        3. Return queues in original order (no sorting by laxity)
        
        Args:
            sessions: List of active sessions
            current_time: Current simulation time (period)
        
        Returns:
            Tuple of (priority_queue, non_priority_queue) in original order
        
        Examples:
            >>> manager = QueueManager()
            >>> sessions = [session1, session2, session3]
            >>> priority_q, non_priority_q = manager.partition_and_sort(sessions, 10)
        """
        # Create queue entries
        priority_entries = []
        non_priority_entries = []
        
        for session in sessions:
            # Calculate laxity for metrics/analysis (not used for sorting)
            laxity = calculate_laxity(
                session,
                current_time,
                self.voltage,
                self.period_minutes
            )
            
            # Create queue entry
            entry = PriorityQueueEntry(
                session=session,
                laxity=laxity,
                is_priority=session.is_priority,
                preferred_rate=0.0,  # Will be set by scheduler
                actual_rate=0.0
            )
            
            # Partition by priority status (no sorting within tier)
            if session.is_priority:
                priority_entries.append(entry)
            else:
                non_priority_entries.append(entry)
        
        # Return queues in original order (not sorted by laxity)
        return priority_entries, non_priority_entries
    
    def partition_sessions(
        self,
        sessions: List[SessionInfo]
    ) -> Tuple[List[SessionInfo], List[SessionInfo]]:
        """
        Partition sessions into priority and non-priority lists.
        
        This is a simpler version that doesn't calculate laxity or create
        queue entries. Useful for initial partitioning.
        
        Args:
            sessions: List of sessions to partition
        
        Returns:
            Tuple of (priority_sessions, non_priority_sessions)
        """
        priority_sessions = [s for s in sessions if s.is_priority]
        non_priority_sessions = [s for s in sessions if not s.is_priority]
        return priority_sessions, non_priority_sessions
    
    def create_queue_entries(
        self,
        sessions: List[SessionInfo],
        current_time: int
    ) -> List[PriorityQueueEntry]:
        """
        Create queue entries with laxity for a list of sessions.
        
        Args:
            sessions: List of sessions
            current_time: Current simulation time (period)
        
        Returns:
            List of PriorityQueueEntry objects
        """
        entries = []
        
        for session in sessions:
            laxity = calculate_laxity(
                session,
                current_time,
                self.voltage,
                self.period_minutes
            )
            
            entry = PriorityQueueEntry(
                session=session,
                laxity=laxity,
                is_priority=session.is_priority,
                preferred_rate=0.0,
                actual_rate=0.0
            )
            entries.append(entry)
        
        return entries
    
    def sort_by_laxity(
        self,
        entries: List[PriorityQueueEntry],
        reverse: bool = False
    ) -> List[PriorityQueueEntry]:
        """
        Sort queue entries by laxity.
        
        Args:
            entries: List of queue entries to sort
            reverse: If True, sort in descending order (highest laxity first)
        
        Returns:
            Sorted list of queue entries
        """
        return sorted(entries, key=lambda e: e.laxity, reverse=reverse)
    
    def get_queue_statistics(
        self,
        priority_queue: List[PriorityQueueEntry],
        non_priority_queue: List[PriorityQueueEntry]
    ) -> dict:
        """
        Calculate statistics about the queues.
        
        Args:
            priority_queue: Priority queue
            non_priority_queue: Non-priority queue
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            'priority_count': len(priority_queue),
            'non_priority_count': len(non_priority_queue),
            'total_count': len(priority_queue) + len(non_priority_queue),
        }
        
        if priority_queue:
            stats['priority_min_laxity'] = min(e.laxity for e in priority_queue)
            stats['priority_max_laxity'] = max(e.laxity for e in priority_queue)
            stats['priority_avg_laxity'] = sum(e.laxity for e in priority_queue) / len(priority_queue)
        
        if non_priority_queue:
            stats['non_priority_min_laxity'] = min(e.laxity for e in non_priority_queue)
            stats['non_priority_max_laxity'] = max(e.laxity for e in non_priority_queue)
            stats['non_priority_avg_laxity'] = sum(e.laxity for e in non_priority_queue) / len(non_priority_queue)
        
        return stats
    
    def filter_feasible_sessions(
        self,
        sessions: List[SessionInfo],
        current_time: int,
        min_laxity: float = 0.0
    ) -> List[SessionInfo]:
        """
        Filter sessions that have sufficient time to complete charging.
        
        Args:
            sessions: List of sessions to filter
            current_time: Current simulation time
            min_laxity: Minimum laxity required (negative means infeasible)
        
        Returns:
            List of sessions with laxity >= min_laxity
        """
        feasible = []
        
        for session in sessions:
            laxity = calculate_laxity(
                session,
                current_time,
                self.voltage,
                self.period_minutes
            )
            
            if laxity >= min_laxity:
                feasible.append(session)
        
        return feasible
