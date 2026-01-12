"""
Adaptive Queuing Priority Scheduler (AQPS) - Main Implementation.

This module implements the core AQPS algorithm, providing a computationally
efficient alternative to MPC-based approaches for priority-aware EV fleet charging.

Key Features:
- Two-tier priority queue (priority EVs always processed first)
- O(n) computational complexity (simple partitioning, no sorting)
- Guaranteed minimum rates for priority EVs
- Compatible with discrete pilot signals

Author: Research Team
Phase: 1 (Core Algorithm)
"""

import logging
from typing import Dict, List, Optional
from .data_structures import (
    SessionInfo,
    PriorityQueueEntry,
    AQPSConfig,
    SchedulingMetrics
)
from .queue_manager import QueueManager
from .utils import format_schedule, validate_sessions


logger = logging.getLogger(__name__)


class AdaptiveQueuingPriorityScheduler:
    """
    Adaptive Queuing Priority Scheduler for EV fleet charging.
    
    The AQPS algorithm implements priority-aware charging using a simple
    two-tier approach that achieves O(n) complexity while maintaining
    priority guarantees and cost optimization.
    
    Phase 1 Implementation:
    - Two-tier queue structure (priority/non-priority)
    - Simple partitioning by is_priority flag (no internal sorting)
    - Guaranteed minimum rates for priority EVs
    - Fair allocation of remaining capacity
    - Basic infrastructure constraint checking
    
    Attributes:
        config: AQPS configuration parameters
        queue_manager: Queue management instance
        current_time: Current simulation time (period)
        metrics_history: List of metrics from each scheduling cycle
    
    Examples:
        >>> config = AQPSConfig(
        ...     min_priority_rate=11.0,
        ...     total_capacity=150.0,
        ...     period_minutes=5.0,
        ...     voltage=220.0
        ... )
        >>> scheduler = AdaptiveQueuingPriorityScheduler(config)
        >>> schedule = scheduler.schedule(sessions, current_time=10)
    """
    
    def __init__(self, config: Optional[AQPSConfig] = None):
        """
        Initialize the AQPS scheduler.
        
        Args:
            config: Configuration parameters (uses defaults if None)
        """
        self.config = config or AQPSConfig()
        self.config.validate()
        
        self.queue_manager = QueueManager(
            voltage=self.config.voltage,
            period_minutes=self.config.period_minutes
        )
        
        self.current_time = 0
        self.metrics_history: List[SchedulingMetrics] = []
        
        if self.config.enable_logging:
            logging.basicConfig(level=logging.INFO)
            logger.info(f"AQPS Scheduler initialized with config: {self.config}")
    
    def schedule(
        self,
        sessions: List[SessionInfo],
        current_time: int
    ) -> Dict[str, float]:
        """
        Main scheduling method - generate charging rates for all sessions.
        
        This is the primary entry point for the scheduler. It implements the
        AQPS algorithm as outlined in the design document:
        
        1. Validate and partition sessions by is_priority flag
        2. Allocate minimum rates to priority EVs
        3. Maximize priority EV rates within capacity
        4. Allocate remaining capacity to non-priority EVs
        5. Return formatted schedule
        
        Args:
            sessions: List of active charging sessions
            current_time: Current simulation time (period)
        
        Returns:
            Dictionary mapping station_id → charging_rate (Amps)
        
        Raises:
            ValueError: If sessions contain validation errors
        """
        self.current_time = current_time
        
        # Validate sessions
        errors = validate_sessions(sessions)
        if errors:
            error_msg = "Session validation failed:\n" + "\n".join(errors)
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Handle empty session list
        if not sessions:
            logger.debug(f"No active sessions at t={current_time}")
            return {}
        
        # Initialize metrics for this cycle
        metrics = SchedulingMetrics(timestamp=current_time)
        
        # Step 1: Partition and sort sessions
        priority_queue, non_priority_queue = self.queue_manager.partition_and_sort(
            sessions, current_time
        )
        
        metrics.priority_sessions_active = len(priority_queue)
        metrics.non_priority_sessions_active = len(non_priority_queue)
        
        logger.info(
            f"t={current_time}: {len(priority_queue)} priority, "
            f"{len(non_priority_queue)} non-priority sessions"
        )
        
        # Step 2: Initialize schedule
        schedule: Dict[str, float] = {}
        
        # Step 3: Allocate priority EVs
        schedule = self._allocate_priority_evs(
            priority_queue,
            schedule,
            metrics
        )
        
        # Step 4: Allocate non-priority EVs
        schedule = self._allocate_non_priority_evs(
            non_priority_queue,
            schedule,
            metrics
        )
        
        # Calculate final metrics
        metrics.total_allocated_capacity = sum(schedule.values())
        metrics.capacity_utilization = (
            metrics.total_allocated_capacity / self.config.total_capacity * 100
        )
        
        # Store metrics
        self.metrics_history.append(metrics)
        
        # Log warnings if any
        for warning in metrics.warnings:
            logger.warning(f"t={current_time}: {warning}")
        
        logger.info(
            f"t={current_time}: Allocated {metrics.total_allocated_capacity:.1f}A "
            f"({metrics.capacity_utilization:.1f}% utilization)"
        )
        
        # Return formatted schedule
        return format_schedule(schedule)
    
    def _allocate_priority_evs(
        self,
        priority_queue: List[PriorityQueueEntry],
        schedule: Dict[str, float],
        metrics: SchedulingMetrics
    ) -> Dict[str, float]:
        """
        Allocate charging rates to priority EVs.
        
        Phase 1 Implementation:
        1. Assign minimum guaranteed rate (11A) to all priority EVs
        2. Check if this violates capacity constraints
        3. Maximize rates up to max_rate within available capacity
        
        Args:
            priority_queue: Sorted list of priority EVs (by laxity)
            schedule: Current schedule being built
            metrics: Metrics object to update
        
        Returns:
            Updated schedule dictionary
        """
        if not priority_queue:
            return schedule
        
        # Phase 1A: Assign minimum rates
        priority_min_total = 0.0
        
        for entry in priority_queue:
            session = entry.session
            min_rate = max(session.min_rate, self.config.min_priority_rate)
            
            # Assign minimum rate
            schedule[session.station_id] = min_rate
            entry.actual_rate = min_rate
            priority_min_total += min_rate
        
        # Check if minimum allocation exceeds capacity
        if priority_min_total > self.config.total_capacity:
            warning = (
                f"Priority minimum rates ({priority_min_total:.1f}A) exceed "
                f"total capacity ({self.config.total_capacity:.1f}A). "
                f"Cannot guarantee all priority EVs. Threshold exceeded."
            )
            metrics.add_warning(warning)
            logger.warning(warning)
            
            # Phase 2 will handle this - for now, just track the warning
            # Don't raise exception, just log
        
        metrics.priority_allocated_capacity = priority_min_total
        metrics.priority_sessions_at_min = len(priority_queue)
        
        # Phase 1B: Maximize priority EV rates within capacity
        remaining_capacity = self.config.total_capacity - priority_min_total
        
        if remaining_capacity > 0:
            schedule = self._maximize_priority_rates(
                priority_queue,
                schedule,
                remaining_capacity,
                metrics
            )
        
        return schedule
    
    def _maximize_priority_rates(
        self,
        priority_queue: List[PriorityQueueEntry],
        schedule: Dict[str, float],
        available_capacity: float,
        metrics: SchedulingMetrics
    ) -> Dict[str, float]:
        """
        Maximize charging rates for priority EVs within available capacity.
        
        This method attempts to increase rates from minimum to maximum for
        priority EVs, respecting capacity constraints. EVs are processed in
        the order they appear in the queue (no specific ordering).
        
        Args:
            priority_queue: List of priority EVs
            schedule: Current schedule
            available_capacity: Remaining capacity after minimum allocation
            metrics: Metrics to update
        
        Returns:
            Updated schedule
        """
        # Track how much capacity we can still allocate
        capacity_pool = available_capacity
        
        # Process priority EVs in queue order
        for entry in priority_queue:
            session = entry.session
            current_rate = schedule[session.station_id]
            
            # Calculate how much we could increase this EV's rate
            max_increase = session.max_rate - current_rate
            
            if max_increase > 0 and capacity_pool > 0:
                # Allocate as much as possible up to max_rate
                increase = min(max_increase, capacity_pool)
                new_rate = current_rate + increase
                
                schedule[session.station_id] = new_rate
                entry.actual_rate = new_rate
                capacity_pool -= increase
                
                # Update metrics
                if new_rate >= session.max_rate - 0.1:  # Within tolerance
                    metrics.priority_sessions_at_max += 1
                    metrics.priority_sessions_at_min -= 1
        
        # Update total priority capacity
        metrics.priority_allocated_capacity = sum(
            schedule[e.session.station_id] for e in priority_queue
        )
        
        logger.debug(
            f"Priority maximization: allocated "
            f"{available_capacity - capacity_pool:.1f}A additional capacity, "
            f"{capacity_pool:.1f}A remaining"
        )
        
        return schedule
    
    def _allocate_non_priority_evs(
        self,
        non_priority_queue: List[PriorityQueueEntry],
        schedule: Dict[str, float],
        metrics: SchedulingMetrics
    ) -> Dict[str, float]:
        """
        Allocate charging rates to non-priority EVs.
        
        Phase 1 Implementation:
        - Calculate remaining capacity after priority allocation
        - Fair-share allocation among non-priority EVs
        - Respect min/max rate constraints
        
        Args:
            non_priority_queue: Sorted list of non-priority EVs
            schedule: Current schedule
            metrics: Metrics to update
        
        Returns:
            Updated schedule
        """
        if not non_priority_queue:
            return schedule
        
        # Calculate remaining capacity
        priority_usage = metrics.priority_allocated_capacity
        remaining_capacity = self.config.total_capacity - priority_usage
        
        if remaining_capacity <= 0:
            logger.info("No capacity remaining for non-priority EVs")
            for entry in non_priority_queue:
                schedule[entry.session.station_id] = 0.0
                entry.actual_rate = 0.0
            return schedule
        
        # Fair share allocation
        n_non_priority = len(non_priority_queue)
        fair_share = remaining_capacity / n_non_priority
        
        logger.debug(
            f"Non-priority allocation: {remaining_capacity:.1f}A available, "
            f"fair share = {fair_share:.1f}A per EV"
        )
        
        # Allocate to each non-priority EV
        capacity_pool = remaining_capacity
        
        for entry in non_priority_queue:
            session = entry.session
            
            # Desired rate is fair share, clamped to [min_rate, max_rate]
            desired_rate = max(session.min_rate, min(fair_share, session.max_rate))
            
            # Allocate (for Phase 1, we just assign directly)
            if capacity_pool >= desired_rate:
                allocated_rate = desired_rate
            else:
                # Not enough capacity, give what's left
                allocated_rate = max(0.0, capacity_pool)
            
            schedule[session.station_id] = allocated_rate
            entry.actual_rate = allocated_rate
            capacity_pool -= allocated_rate
        
        return schedule
    
    def get_metrics(self, last_n: Optional[int] = None) -> List[SchedulingMetrics]:
        """
        Retrieve scheduling metrics history.
        
        Args:
            last_n: Return only the last n metrics (None = all)
        
        Returns:
            List of SchedulingMetrics objects
        """
        if last_n is None:
            return self.metrics_history.copy()
        else:
            return self.metrics_history[-last_n:]
    
    def get_current_metrics(self) -> Optional[SchedulingMetrics]:
        """Get the most recent metrics."""
        if self.metrics_history:
            return self.metrics_history[-1]
        return None
    
    def reset_metrics(self) -> None:
        """Clear metrics history."""
        self.metrics_history.clear()
        logger.info("Metrics history cleared")
    
    def __repr__(self) -> str:
        return (
            f"AdaptiveQueuingPriorityScheduler("
            f"capacity={self.config.total_capacity}A, "
            f"min_priority_rate={self.config.min_priority_rate}A)"
        )
