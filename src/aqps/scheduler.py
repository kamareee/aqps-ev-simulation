"""
Adaptive Queuing Priority Scheduler (AQPS) - Main Implementation.

This module implements the core AQPS algorithm, providing a computationally
efficient alternative to MPC-based approaches for priority-aware EV fleet charging.

Key Features:
- Two-tier priority queue (priority EVs always processed first)
- O(n) computational complexity (simple partitioning, no sorting)
- Guaranteed minimum rates for priority EVs
- Preemption mechanism for capacity reallocation (Phase 2)
- Threshold tracking for analysis (Phase 2)
- TOU-aware deferral for non-priority EVs (Phase 3)
- Three-phase network infrastructure support (Phase 3)
- PV/BESS renewable integration (Phase 3)
- J1772 pilot signal quantization (Phase 4)
- Computational performance benchmarking (Phase 4)

Author: Research Team
Phase: 4 (Quantization & Polish)
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Set
from .data_structures import (
    SessionInfo,
    PriorityQueueEntry,
    AQPSConfig,
    SchedulingMetrics,
    PreemptionEvent,
    ThresholdViolationEvent,
    TOUDeferralEvent,
    Phase3Config,
    TOUMetrics,
    # Phase 4 data structures
    Phase4Config,
    QuantizationMetrics,
    QuantizationEvent,
    ComputationalMetrics,
    SimulationSummary,
    J1772_PILOT_SIGNALS
)
from .queue_manager import QueueManager
from .preemption import PreemptionManager
from .threshold_tracker import ThresholdTracker
from .utils import format_schedule, validate_sessions


logger = logging.getLogger(__name__)


class AdaptiveQueuingPriorityScheduler:
    """
    Adaptive Queuing Priority Scheduler for EV fleet charging.
    
    The AQPS algorithm implements priority-aware charging using a simple
    two-tier approach that achieves O(n) complexity while maintaining
    priority guarantees and cost optimization.
    
    Phase 2 Implementation:
    - Two-tier queue structure (priority/non-priority)
    - Simple partitioning by is_priority flag (FIFO within tiers)
    - Guaranteed minimum rates for priority EVs
    - Preemption mechanism (Option B: highest-laxity-first + Option A fallback)
    - Threshold tracking for journal-quality analysis
    - Fair allocation of remaining capacity
    
    Phase 3 Implementation:
    - TOU-aware deferral for non-priority EVs (aggressive policy)
    - Three-phase network infrastructure with per-phase limits
    - PV and BESS renewable integration
    - Deferral window tracking to prevent congestion
    
    Phase 4 Implementation:
    - J1772 discrete pilot signal quantization
    - Priority-aware quantization (ceil for priority, floor for non-priority)
    - Computational performance benchmarking
    - Simulation summary statistics
    
    Attributes:
        config: AQPS configuration parameters
        phase3_config: Phase 3 TOU/renewable configuration
        phase4_config: Phase 4 quantization/benchmarking configuration
        queue_manager: Queue management instance
        preemption_manager: Preemption logic handler (Phase 2)
        threshold_tracker: Threshold violation tracker (Phase 2)
        tou_optimizer: TOU optimization handler (Phase 3)
        network: Three-phase network infrastructure (Phase 3)
        renewable: PV/BESS integration (Phase 3)
        current_time: Current simulation time (period)
        metrics_history: List of metrics from each scheduling cycle
        tou_metrics_history: List of TOU metrics from each cycle (Phase 3)
        quantization_history: List of quantization metrics (Phase 4)
        computational_history: List of computational metrics (Phase 4)
    
    Examples:
        >>> config = AQPSConfig(
        ...     min_priority_rate=16.0,  # J1772 minimum
        ...     total_capacity=600.0,    # 3 phases * 200A
        ...     period_minutes=5.0,
        ...     voltage=415.0            # Three-phase voltage
        ... )
        >>> scheduler = AdaptiveQueuingPriorityScheduler(config)
        >>> schedule = scheduler.schedule(sessions, current_time=10)
    """
    
    def __init__(
        self,
        config: Optional[AQPSConfig] = None,
        phase3_config: Optional[Phase3Config] = None,
        phase4_config: Optional[Phase4Config] = None
    ):
        """
        Initialize the AQPS scheduler.
        
        Args:
            config: Configuration parameters (uses defaults if None)
            phase3_config: Phase 3 TOU/renewable config (uses defaults if None)
            phase4_config: Phase 4 quantization/benchmarking config (uses defaults if None)
        """
        self.config = config or AQPSConfig()
        self.config.validate()
        
        self.phase3_config = phase3_config or Phase3Config()
        self.phase4_config = phase4_config or Phase4Config()
        
        # Validate Phase 4 config
        phase4_errors = self.phase4_config.validate()
        if phase4_errors:
            raise ValueError(f"Phase 4 config errors: {phase4_errors}")
        
        self.queue_manager = QueueManager(
            voltage=self.config.voltage,
            period_minutes=self.config.period_minutes
        )
        
        # Phase 2: Initialize preemption manager
        self.preemption_manager = PreemptionManager(
            min_priority_rate=self.config.min_priority_rate
        )
        
        # Phase 2: Initialize threshold tracker
        self.threshold_tracker = ThresholdTracker(
            total_capacity=self.config.total_capacity,
            min_priority_rate=self.config.min_priority_rate,
            enable_snapshots=True
        )
        
        # Phase 3: Initialize TOU optimizer, network, and renewables (lazy loading)
        self._tou_optimizer = None
        self._network = None
        self._renewable = None
        self._tou_tariff = None
        
        self.current_time = 0
        self.metrics_history: List[SchedulingMetrics] = []
        
        # Phase 2: Track preemption events per scheduling cycle
        self._current_preemption_events: List[PreemptionEvent] = []
        
        # Phase 3: Track TOU deferral events and metrics
        self._tou_deferral_events: List[TOUDeferralEvent] = []
        self.tou_metrics_history: List[TOUMetrics] = []
        self._deferred_sessions: Dict[str, int] = {}  # session_id → target_period
        
        # Phase 4: Track quantization and computational metrics
        self.quantization_history: List[QuantizationMetrics] = []
        self.computational_history: List[ComputationalMetrics] = []
        self._priority_session_ids: Set[str] = set()  # Track priority sessions for quantization
        
        if self.config.enable_logging:
            logging.basicConfig(level=logging.INFO)
            logger.info(f"AQPS Scheduler initialized with config: {self.config}")
            if self.phase4_config.enable_quantization:
                logger.info(f"Quantization enabled with signals: {self.phase4_config.pilot_signals}")
    
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
        3. Apply preemption if needed to guarantee priority minimums (Phase 2)
        4. Maximize priority EV rates within capacity
        5. Allocate remaining capacity to non-priority EVs
        6. Apply TOU deferral for non-priority EVs (Phase 3)
        7. Quantize to J1772 pilot signals (Phase 4)
        8. Record threshold tracking data (Phase 2)
        9. Return formatted schedule
        
        Args:
            sessions: List of active charging sessions
            current_time: Current simulation time (period)
        
        Returns:
            Dictionary mapping station_id → charging_rate (Amps)
        
        Raises:
            ValueError: If sessions contain validation errors
        """
        # Phase 4: Start timing
        start_time = time.perf_counter() if self.phase4_config.enable_timing else 0
        
        self.current_time = current_time
        self._current_preemption_events = []  # Reset for this cycle
        
        # Validate sessions
        errors = validate_sessions(sessions)
        if errors:
            error_msg = "Session validation failed:\n" + "\n".join(errors)
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Handle empty session list
        if not sessions:
            logger.debug(f"No active sessions at t={current_time}")
            self._record_computational_metrics(
                current_time, start_time, 0, 0, 0, False
            )
            return {}
        
        # Initialize metrics for this cycle
        metrics = SchedulingMetrics(timestamp=current_time)
        
        # Step 1: Partition sessions (FIFO order within tiers)
        priority_queue, non_priority_queue = self.queue_manager.partition_and_sort(
            sessions, current_time
        )
        
        # Phase 4: Track priority session IDs for quantization
        self._priority_session_ids = {
            entry.session.session_id for entry in priority_queue
        }
        
        metrics.priority_sessions_active = len(priority_queue)
        metrics.non_priority_sessions_active = len(non_priority_queue)
        
        logger.info(
            f"t={current_time}: {len(priority_queue)} priority, "
            f"{len(non_priority_queue)} non-priority sessions"
        )
        
        # Step 2: Initialize schedule
        schedule: Dict[str, float] = {}
        
        # Step 3: Allocate priority EVs with preemption support (Phase 2)
        schedule, preemption_count = self._allocate_priority_evs_with_preemption(
            priority_queue,
            non_priority_queue,
            schedule,
            metrics
        )
        
        # Step 4: Allocate non-priority EVs
        schedule = self._allocate_non_priority_evs(
            non_priority_queue,
            schedule,
            metrics
        )
        
        # Phase 4: Apply quantization to J1772 pilot signals
        if self.phase4_config.enable_quantization:
            schedule = self._apply_quantization(
                schedule,
                priority_queue,
                non_priority_queue,
                metrics
            )
        
        # Calculate final metrics
        metrics.total_allocated_capacity = sum(schedule.values())
        metrics.capacity_utilization = (
            metrics.total_allocated_capacity / self.config.total_capacity * 100
        )
        
        # Phase 2: Record timestep in threshold tracker
        violation = self.threshold_tracker.record_timestep(
            timestamp=current_time,
            priority_queue=priority_queue,
            non_priority_queue=non_priority_queue,
            schedule=schedule,
            preemption_occurred=preemption_count > 0,
            preemption_count=preemption_count
        )
        
        if violation:
            metrics.add_warning(f"Threshold violation: {violation}")
        
        # Store metrics
        self.metrics_history.append(metrics)
        
        # Phase 4: Record computational metrics
        self._record_computational_metrics(
            current_time,
            start_time,
            len(sessions),
            len(priority_queue),
            len(non_priority_queue),
            preemption_count > 0
        )
        
        # Log warnings if any
        for warning in metrics.warnings:
            logger.warning(f"t={current_time}: {warning}")
        
        logger.info(
            f"t={current_time}: Allocated {metrics.total_allocated_capacity:.1f}A "
            f"({metrics.capacity_utilization:.1f}% utilization)"
        )
        
        # Return formatted schedule
        return format_schedule(schedule)
    
    def _allocate_priority_evs_with_preemption(
        self,
        priority_queue: List[PriorityQueueEntry],
        non_priority_queue: List[PriorityQueueEntry],
        schedule: Dict[str, float],
        metrics: SchedulingMetrics
    ) -> Tuple[Dict[str, float], int]:
        """
        Allocate charging rates to priority EVs with preemption support.
        
        Phase 2 Implementation:
        1. First, allocate fair share to non-priority EVs (tentative)
        2. Assign minimum guaranteed rate (11A) to all priority EVs
        3. If capacity insufficient, preempt non-priority EVs
        4. Maximize rates up to max_rate within available capacity
        
        Args:
            priority_queue: List of priority EVs (in FIFO order)
            non_priority_queue: List of non-priority EVs (in FIFO order)
            schedule: Current schedule being built
            metrics: Metrics object to update
        
        Returns:
            Tuple of (updated_schedule, preemption_count)
        """
        preemption_count = 0
        
        if not priority_queue:
            return schedule, preemption_count
        
        # First, give non-priority EVs tentative allocations
        # This allows us to preempt from them if needed
        if non_priority_queue:
            remaining_for_non_priority = max(
                0,
                self.config.total_capacity - len(priority_queue) * self.config.min_priority_rate
            )
            fair_share = remaining_for_non_priority / len(non_priority_queue) if non_priority_queue else 0
            
            for entry in non_priority_queue:
                session = entry.session
                tentative_rate = min(fair_share, session.max_rate)
                schedule[session.station_id] = tentative_rate
                entry.actual_rate = tentative_rate
        
        # Calculate capacity used by non-priority
        non_priority_usage = sum(
            schedule.get(e.session.station_id, 0.0) for e in non_priority_queue
        )
        
        # Phase 2A: Assign minimum rates to priority EVs
        priority_min_total = 0.0
        
        for entry in priority_queue:
            session = entry.session
            min_rate = max(session.min_rate, self.config.min_priority_rate)
            
            # Check available capacity
            current_usage = priority_min_total + non_priority_usage
            available = self.config.total_capacity - current_usage
            
            if available >= min_rate:
                # Enough capacity without preemption
                schedule[session.station_id] = min_rate
                entry.actual_rate = min_rate
                priority_min_total += min_rate
            else:
                # Need preemption (Phase 2)
                shortfall = min_rate - available
                
                # Check if preemption is needed
                preemption_needed, capacity_shortfall = self.preemption_manager.check_preemption_needed(
                    priority_entry=entry,
                    current_allocation=0.0,
                    min_required_rate=min_rate,
                    remaining_capacity=available
                )
                
                if preemption_needed and non_priority_queue:
                    # Execute preemption
                    schedule, preemption_event = self.preemption_manager.execute_preemption(
                        needed_capacity=shortfall,
                        priority_entry=entry,
                        non_priority_queue=non_priority_queue,
                        schedule=schedule,
                        current_time=self.current_time
                    )
                    
                    self._current_preemption_events.append(preemption_event)
                    preemption_count += 1
                    
                    # Recalculate non-priority usage after preemption
                    non_priority_usage = sum(
                        schedule.get(e.session.station_id, 0.0) for e in non_priority_queue
                    )
                    
                    # Now allocate to priority EV
                    current_usage = priority_min_total + non_priority_usage
                    available = self.config.total_capacity - current_usage
                    
                    if available >= min_rate:
                        schedule[session.station_id] = min_rate
                        entry.actual_rate = min_rate
                        priority_min_total += min_rate
                    else:
                        # Still can't meet minimum - threshold exceeded
                        allocated = max(0, available)
                        schedule[session.station_id] = allocated
                        entry.actual_rate = allocated
                        priority_min_total += allocated
                        
                        warning = (
                            f"Priority EV {session.session_id} allocated {allocated:.1f}A "
                            f"(below minimum {min_rate:.1f}A). Threshold exceeded."
                        )
                        metrics.add_warning(warning)
                        logger.warning(warning)
                else:
                    # Allocate what's available
                    allocated = max(0, available)
                    schedule[session.station_id] = allocated
                    entry.actual_rate = allocated
                    priority_min_total += allocated
                    
                    if allocated < min_rate:
                        warning = (
                            f"Priority EV {session.session_id} allocated {allocated:.1f}A "
                            f"(below minimum {min_rate:.1f}A). No non-priority EVs to preempt."
                        )
                        metrics.add_warning(warning)
        
        metrics.priority_allocated_capacity = priority_min_total
        metrics.priority_sessions_at_min = len(priority_queue)
        
        # Phase 2B: Maximize priority EV rates within capacity
        non_priority_usage = sum(
            schedule.get(e.session.station_id, 0.0) for e in non_priority_queue
        )
        remaining_capacity = self.config.total_capacity - priority_min_total - non_priority_usage
        
        if remaining_capacity > 0:
            schedule = self._maximize_priority_rates(
                priority_queue,
                schedule,
                remaining_capacity,
                metrics
            )
        
        return schedule, preemption_count
    
    def _allocate_priority_evs(
        self,
        priority_queue: List[PriorityQueueEntry],
        schedule: Dict[str, float],
        metrics: SchedulingMetrics
    ) -> Dict[str, float]:
        """
        Allocate charging rates to priority EVs (legacy method without preemption).
        
        Phase 1 Implementation (kept for compatibility):
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
        Allocate charging rates to non-priority EVs with TOU optimization.
        
        Phase 3 Implementation:
        - Calculate remaining capacity after priority allocation
        - Apply TOU deferral logic (aggressive policy)
        - Fair-share allocation among non-deferred EVs
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
        
        # Phase 3: Separate deferred vs immediate charging sessions
        immediate_entries = []
        deferred_entries = []
        
        for entry in non_priority_queue:
            session = entry.session
            
            # Check if TOU optimization is enabled and we have an optimizer
            if (self.phase3_config.enable_tou_optimization and 
                self._tou_optimizer is not None):
                
                # Check if this session should be deferred
                decision = self._tou_optimizer.should_defer(
                    session_id=session.session_id,
                    departure_period=session.departure_time,
                    remaining_demand_kwh=session.remaining_demand,
                    max_rate_amps=session.max_rate,
                    current_period=self.current_time,
                    is_priority=False
                )
                
                if decision.can_defer and decision.target_period is not None:
                    # Record deferral
                    entry.deferred = True
                    self._deferred_sessions[session.session_id] = decision.target_period
                    
                    # Log deferral event
                    deferral_event = TOUDeferralEvent(
                        timestamp=self.current_time,
                        session_id=session.session_id,
                        station_id=session.station_id,
                        original_period=self.current_time,
                        target_period=decision.target_period,
                        original_price=decision.current_price,
                        target_price=decision.target_price,
                        savings_per_kwh=decision.savings_per_kwh,
                        energy_deferred_kwh=session.remaining_demand,
                        reason=decision.reason
                    )
                    self._tou_deferral_events.append(deferral_event)
                    
                    deferred_entries.append(entry)
                    logger.debug(
                        f"TOU deferral: {session.session_id} deferred to period "
                        f"{decision.target_period} (save ${decision.savings_per_kwh:.3f}/kWh)"
                    )
                    continue
            
            # Check if this was a previously deferred session reaching its target
            if session.session_id in self._deferred_sessions:
                target_period = self._deferred_sessions[session.session_id]
                if self.current_time >= target_period:
                    # Time to charge this deferred session
                    del self._deferred_sessions[session.session_id]
                    entry.deferred = False
                    immediate_entries.append(entry)
                    logger.debug(
                        f"Deferred session {session.session_id} now charging "
                        f"(reached target period {target_period})"
                    )
                else:
                    # Still waiting
                    deferred_entries.append(entry)
                    entry.deferred = True
            else:
                # Not deferred, charge immediately
                immediate_entries.append(entry)
        
        # Allocate to deferred entries (minimum rate to maintain connection)
        for entry in deferred_entries:
            session = entry.session
            # Deferred sessions get 0 rate (not charging yet)
            schedule[session.station_id] = 0.0
            entry.actual_rate = 0.0
        
        # Fair share allocation for immediate entries
        if not immediate_entries:
            return schedule
        
        n_immediate = len(immediate_entries)
        fair_share = remaining_capacity / n_immediate
        
        logger.debug(
            f"Non-priority allocation: {remaining_capacity:.1f}A available, "
            f"{n_immediate} immediate, {len(deferred_entries)} deferred, "
            f"fair share = {fair_share:.1f}A per EV"
        )
        
        # Allocate to each immediate non-priority EV
        capacity_pool = remaining_capacity
        
        for entry in immediate_entries:
            session = entry.session
            
            # Desired rate is fair share, clamped to [min_rate, max_rate]
            desired_rate = max(session.min_rate, min(fair_share, session.max_rate))
            
            # Allocate
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
    
    # ==========================================================================
    # Phase 2: Preemption and Threshold Tracking Methods
    # ==========================================================================
    
    def get_preemption_statistics(self) -> Dict:
        """
        Get preemption statistics from the preemption manager.
        
        Returns:
            Dictionary with preemption statistics
        """
        return self.preemption_manager.get_statistics()
    
    def get_preemption_history(self) -> List[PreemptionEvent]:
        """
        Get the full preemption event history.
        
        Returns:
            List of PreemptionEvent objects
        """
        return self.preemption_manager.get_preemption_history()
    
    def get_threshold_statistics(self) -> Dict:
        """
        Get threshold tracking statistics.
        
        Returns:
            Dictionary with threshold statistics
        """
        return self.threshold_tracker.get_summary_statistics()
    
    def get_threshold_violations(self) -> List[ThresholdViolationEvent]:
        """
        Get all threshold violation events.
        
        Returns:
            List of ThresholdViolationEvent objects
        """
        return self.threshold_tracker.violations.copy()
    
    def get_critical_threshold(self) -> Optional[float]:
        """
        Get the critical priority ratio where violations begin.
        
        Returns:
            Critical priority ratio or None if no violations
        """
        return self.threshold_tracker.get_critical_threshold()
    
    def export_analysis_data(self) -> Dict:
        """
        Export all analysis data for visualization and reporting.
        
        Returns:
            Comprehensive dictionary with all tracking data
        """
        return {
            'scheduling_metrics': [
                {
                    'timestamp': m.timestamp,
                    'priority_sessions': m.priority_sessions_active,
                    'non_priority_sessions': m.non_priority_sessions_active,
                    'total_capacity': m.total_allocated_capacity,
                    'priority_capacity': m.priority_allocated_capacity,
                    'utilization': m.capacity_utilization,
                }
                for m in self.metrics_history
            ],
            'preemption': self.preemption_manager.get_statistics(),
            'preemption_events': self.preemption_manager.get_preemption_dataframe_data(),
            'threshold': self.threshold_tracker.export_for_visualization(),
            'config': {
                'total_capacity': self.config.total_capacity,
                'min_priority_rate': self.config.min_priority_rate,
                'max_priority_ratio': self.config.max_priority_ratio,
            }
        }
    
    def reset_all_tracking(self) -> None:
        """Reset all tracking data (metrics, preemption, threshold, TOU)."""
        self.reset_metrics()
        self.preemption_manager.reset()
        self.threshold_tracker.reset()
        # Phase 3: Reset TOU tracking
        self._tou_deferral_events.clear()
        self.tou_metrics_history.clear()
        self._deferred_sessions.clear()
        if self._tou_optimizer:
            self._tou_optimizer.reset()
        # Phase 4: Reset quantization and computational tracking
        self.quantization_history.clear()
        self.computational_history.clear()
        self._priority_session_ids.clear()
        logger.info("All tracking data reset")
    
    # ==========================================================================
    # Phase 4: Quantization and Computational Metrics Methods
    # ==========================================================================
    
    def _apply_quantization(
        self,
        schedule: Dict[str, float],
        priority_queue: List[PriorityQueueEntry],
        non_priority_queue: List[PriorityQueueEntry],
        metrics: SchedulingMetrics
    ) -> Dict[str, float]:
        """
        Apply J1772 pilot signal quantization to the schedule.
        
        Priority EVs use ceiling quantization if floor would violate minimum
        guarantee. Non-priority EVs always use floor quantization.
        
        Args:
            schedule: Current schedule with continuous rates
            priority_queue: Priority session entries
            non_priority_queue: Non-priority session entries
            metrics: Metrics object to update with warnings
        
        Returns:
            Quantized schedule with valid pilot signals
        """
        quant_metrics = QuantizationMetrics(timestamp=self.current_time)
        quantized_schedule: Dict[str, float] = {}
        
        # Build lookup for session info
        session_lookup: Dict[str, Tuple[SessionInfo, bool]] = {}
        for entry in priority_queue:
            session_lookup[entry.session.station_id] = (entry.session, True)
        for entry in non_priority_queue:
            session_lookup[entry.session.station_id] = (entry.session, False)
        
        quant_metrics.total_sessions = len(schedule)
        quant_metrics.priority_sessions = len(priority_queue)
        quant_metrics.non_priority_sessions = len(non_priority_queue)
        quant_metrics.pre_quantization_capacity = sum(schedule.values())
        
        for station_id, rate in schedule.items():
            session_info = session_lookup.get(station_id)
            is_priority = session_info[1] if session_info else False
            session = session_info[0] if session_info else None
            
            if rate <= 0:
                quantized_schedule[station_id] = 0.0
                continue
            
            # Get floor quantized rate
            floor_rate = self.phase4_config.get_floor_signal(rate)
            
            # For priority EVs, check if floor violates minimum guarantee
            if is_priority and self.phase4_config.priority_ceil_enabled:
                min_required = self.config.min_priority_rate
                
                if floor_rate < min_required and rate >= min_required:
                    # Floor would violate guarantee, use ceiling
                    ceil_rate = self.phase4_config.get_ceil_signal(rate)
                    quantized_rate = ceil_rate
                    method = 'ceil'
                    reason = f"priority_guarantee (floor={floor_rate}A < min={min_required}A)"
                    quant_metrics.priority_ceil_count += 1
                    quant_metrics.capacity_gained += (ceil_rate - rate)
                else:
                    quantized_rate = floor_rate
                    method = 'floor'
                    reason = "standard"
                    quant_metrics.capacity_lost += (rate - floor_rate)
            else:
                # Non-priority: always use floor
                quantized_rate = floor_rate
                method = 'floor'
                reason = "standard"
                quant_metrics.capacity_lost += (rate - floor_rate)
            
            quantized_schedule[station_id] = quantized_rate
            
            # Track individual events if enabled
            if self.phase4_config.track_quantization_events and rate != quantized_rate:
                event = QuantizationEvent(
                    session_id=session.session_id if session else station_id,
                    station_id=station_id,
                    is_priority=is_priority,
                    pre_quantization_rate=rate,
                    post_quantization_rate=quantized_rate,
                    quantization_method=method,
                    reason=reason
                )
                quant_metrics.events.append(event)
        
        # Calculate final metrics
        quant_metrics.post_quantization_capacity = sum(quantized_schedule.values())
        
        # Post-quantization capacity adjustment: if priority ceiling caused overage,
        # reduce non-priority rates to respect capacity limit
        if quant_metrics.post_quantization_capacity > self.config.total_capacity:
            overage = quant_metrics.post_quantization_capacity - self.config.total_capacity
            quantized_schedule = self._adjust_for_capacity_overage(
                quantized_schedule, session_lookup, overage, quant_metrics
            )
            quant_metrics.post_quantization_capacity = sum(quantized_schedule.values())
        
        # Estimate energy loss (per period)
        energy_per_amp_per_period = (
            self.config.voltage * (self.config.period_minutes / 60.0) / 1000.0
        )
        quant_metrics.energy_loss_kwh = (
            quant_metrics.capacity_lost * energy_per_amp_per_period
        )
        
        # Store metrics
        self.quantization_history.append(quant_metrics)
        
        # Log if significant capacity change
        if abs(quant_metrics.net_capacity_change) > 1.0:
            logger.debug(
                f"Quantization: {quant_metrics.pre_quantization_capacity:.1f}A → "
                f"{quant_metrics.post_quantization_capacity:.1f}A "
                f"(efficiency={quant_metrics.quantization_efficiency:.1f}%, "
                f"ceil_count={quant_metrics.priority_ceil_count})"
            )
        
        return quantized_schedule
    
    def _adjust_for_capacity_overage(
        self,
        schedule: Dict[str, float],
        session_lookup: Dict[str, Tuple[SessionInfo, bool]],
        overage: float,
        quant_metrics: QuantizationMetrics
    ) -> Dict[str, float]:
        """
        Adjust non-priority rates to compensate for capacity overage from priority ceiling.
        
        Uses iterative reduction of non-priority EVs by one pilot signal step at a time
        until capacity constraint is satisfied or no further reduction is possible.
        
        Args:
            schedule: Current quantized schedule
            session_lookup: Lookup table for session info
            overage: Amount over capacity (Amps)
            quant_metrics: Metrics object to update
        
        Returns:
            Adjusted schedule respecting capacity limit
        """
        adjusted_schedule = schedule.copy()
        remaining_overage = overage
        max_iterations = 20  # Safety limit
        
        for iteration in range(max_iterations):
            if remaining_overage <= 0.01:  # Close enough to capacity
                break
            
            # Get non-priority stations that can still be reduced, sorted by rate (highest first)
            reducible_stations = []
            for station_id, rate in adjusted_schedule.items():
                if station_id in session_lookup and not session_lookup[station_id][1]:
                    if rate > 0:
                        # Check if we can reduce this further
                        next_lower = self.phase4_config.get_floor_signal(rate - 0.01)
                        if next_lower < rate:
                            reducible_stations.append((station_id, rate, rate - next_lower))
            
            if not reducible_stations:
                # No more stations can be reduced
                break
            
            # Sort by current rate (reduce highest first for fairness)
            reducible_stations.sort(key=lambda x: x[1], reverse=True)
            
            # Reduce the highest-rate non-priority EV by one step
            station_id, current_rate, reduction = reducible_stations[0]
            new_rate = self.phase4_config.get_floor_signal(current_rate - 0.01)
            
            adjusted_schedule[station_id] = new_rate
            remaining_overage -= reduction
            quant_metrics.capacity_lost += reduction
            
            logger.debug(
                f"Capacity adjustment (iter {iteration}): {station_id} reduced "
                f"{current_rate}A → {new_rate}A (remaining overage: {remaining_overage:.1f}A)"
            )
        
        if remaining_overage > 0.01:
            logger.warning(
                f"Could not fully compensate for priority ceiling overage: "
                f"{remaining_overage:.1f}A still over capacity"
            )
        
        return adjusted_schedule
    
    def _record_computational_metrics(
        self,
        timestamp: int,
        start_time: float,
        num_sessions: int,
        num_priority: int,
        num_non_priority: int,
        preemption_occurred: bool
    ) -> None:
        """
        Record computational performance metrics for this scheduling cycle.
        
        Args:
            timestamp: Current simulation time
            start_time: Start time from time.perf_counter()
            num_sessions: Total sessions processed
            num_priority: Number of priority sessions
            num_non_priority: Number of non-priority sessions
            preemption_occurred: Whether preemption was triggered
        """
        if not self.phase4_config.enable_timing:
            return
        
        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time) * 1000.0
        
        comp_metrics = ComputationalMetrics(
            timestamp=timestamp,
            schedule_time_ms=elapsed_ms,
            num_sessions=num_sessions,
            num_priority=num_priority,
            num_non_priority=num_non_priority,
            preemption_occurred=preemption_occurred,
            quantization_enabled=self.phase4_config.enable_quantization
        )
        
        self.computational_history.append(comp_metrics)
        
        logger.debug(
            f"t={timestamp}: schedule() completed in {elapsed_ms:.3f}ms "
            f"({comp_metrics.time_per_session_us:.1f}µs/session)"
        )
    
    def get_quantization_statistics(self) -> Dict:
        """
        Get quantization performance statistics.
        
        Returns:
            Dictionary with quantization statistics
        """
        if not self.quantization_history:
            return {
                'enabled': self.phase4_config.enable_quantization,
                'total_timesteps': 0,
                'avg_efficiency': 100.0,
                'total_capacity_lost': 0.0,
                'total_priority_ceil': 0,
            }
        
        total_pre = sum(m.pre_quantization_capacity for m in self.quantization_history)
        total_post = sum(m.post_quantization_capacity for m in self.quantization_history)
        
        return {
            'enabled': self.phase4_config.enable_quantization,
            'total_timesteps': len(self.quantization_history),
            'avg_efficiency': (total_post / total_pre * 100.0) if total_pre > 0 else 100.0,
            'total_capacity_lost': sum(m.capacity_lost for m in self.quantization_history),
            'total_capacity_gained': sum(m.capacity_gained for m in self.quantization_history),
            'total_priority_ceil': sum(m.priority_ceil_count for m in self.quantization_history),
            'total_energy_loss_kwh': sum(m.energy_loss_kwh for m in self.quantization_history),
            'pilot_signals': self.phase4_config.pilot_signals,
        }
    
    def get_quantization_history(self) -> List[QuantizationMetrics]:
        """Get the full quantization metrics history."""
        return self.quantization_history.copy()
    
    def get_computational_statistics(self) -> Dict:
        """
        Get computational performance statistics.
        
        Returns:
            Dictionary with timing statistics
        """
        if not self.computational_history:
            return {
                'enabled': self.phase4_config.enable_timing,
                'total_timesteps': 0,
                'avg_time_ms': 0.0,
                'max_time_ms': 0.0,
                'min_time_ms': 0.0,
                'total_time_ms': 0.0,
                'avg_sessions': 0.0,
                'avg_time_per_session_us': 0.0,
            }
        
        times = [m.schedule_time_ms for m in self.computational_history]
        
        return {
            'enabled': self.phase4_config.enable_timing,
            'total_timesteps': len(self.computational_history),
            'avg_time_ms': sum(times) / len(times),
            'max_time_ms': max(times),
            'min_time_ms': min(times),
            'total_time_ms': sum(times),
            'avg_sessions': sum(m.num_sessions for m in self.computational_history) / len(self.computational_history),
            'avg_time_per_session_us': sum(m.time_per_session_us for m in self.computational_history) / len(self.computational_history),
        }
    
    def get_computational_history(self) -> List[ComputationalMetrics]:
        """Get the full computational metrics history."""
        return self.computational_history.copy()
    
    def get_simulation_summary(self) -> SimulationSummary:
        """
        Generate a comprehensive summary of the simulation run.
        
        Returns:
            SimulationSummary with aggregate statistics
        """
        summary = SimulationSummary()
        
        # Basic counts
        summary.total_timesteps = len(self.metrics_history)
        
        if self.metrics_history:
            # Session counts from metrics
            summary.priority_sessions_total = sum(
                m.priority_sessions_active for m in self.metrics_history
            )
            summary.non_priority_sessions_total = sum(
                m.non_priority_sessions_active for m in self.metrics_history
            )
            summary.avg_capacity_utilization = sum(
                m.capacity_utilization for m in self.metrics_history
            ) / len(self.metrics_history)
        
        # Preemption statistics
        preempt_stats = self.get_preemption_statistics()
        summary.total_preemptions = preempt_stats.get('total_preemptions', 0)
        
        # Threshold statistics
        threshold_stats = self.get_threshold_statistics()
        summary.total_threshold_violations = threshold_stats.get('violation_count', 0)
        
        # TOU statistics
        if self._tou_optimizer:
            tou_stats = self.get_tou_statistics()
            summary.total_deferrals = tou_stats.get('total_deferrals', 0)
        
        # Computational statistics
        if self.computational_history:
            times = [m.schedule_time_ms for m in self.computational_history]
            summary.avg_schedule_time_ms = sum(times) / len(times)
            summary.max_schedule_time_ms = max(times)
            summary.min_schedule_time_ms = min(times)
        
        # Quantization statistics
        if self.quantization_history:
            quant_stats = self.get_quantization_statistics()
            summary.quantization_efficiency_avg = quant_stats.get('avg_efficiency', 100.0)
        
        return summary
    
    def export_dataframes(self) -> Dict[str, List[Dict]]:
        """
        Export all metrics as lists of dictionaries suitable for DataFrame creation.
        
        Returns:
            Dictionary with keys for each metric type, values are lists of dicts
        
        Example:
            >>> data = scheduler.export_dataframes()
            >>> import pandas as pd
            >>> scheduling_df = pd.DataFrame(data['scheduling'])
            >>> preemption_df = pd.DataFrame(data['preemption'])
        """
        return {
            'scheduling': [
                {
                    'timestamp': m.timestamp,
                    'priority_sessions': m.priority_sessions_active,
                    'non_priority_sessions': m.non_priority_sessions_active,
                    'total_capacity': m.total_allocated_capacity,
                    'priority_capacity': m.priority_allocated_capacity,
                    'utilization_pct': m.capacity_utilization,
                    'priority_at_min': m.priority_sessions_at_min,
                    'priority_at_max': m.priority_sessions_at_max,
                }
                for m in self.metrics_history
            ],
            'preemption': [
                e.to_dict() for e in self.preemption_manager.get_preemption_history()
            ],
            'threshold_violations': [
                e.to_dict() for e in self.threshold_tracker.violations
            ],
            'tou_deferrals': [
                e.to_dict() for e in self._tou_deferral_events
            ],
            'quantization': [
                m.to_dict() for m in self.quantization_history
            ],
            'computational': [
                m.to_dict() for m in self.computational_history
            ],
        }
    
    # ==========================================================================
    # Phase 3: TOU Optimization and Infrastructure Methods
    # ==========================================================================
    
    def configure_tou(
        self,
        peak_price: float,
        off_peak_price: float,
        shoulder_price: Optional[float] = None,
        peak_hours: List[Tuple[float, float]] = None,
        shoulder_hours: List[Tuple[float, float]] = None,
        tariff_name: str = "Custom TOU"
    ) -> None:
        """
        Configure TOU tariff for optimization.
        
        This is the main interface for manually inserting tariff details.
        
        Args:
            peak_price: Peak rate ($/kWh)
            off_peak_price: Off-peak rate ($/kWh)
            shoulder_price: Optional shoulder/mid-peak rate ($/kWh)
            peak_hours: List of (start_hour, end_hour) tuples for peak periods
            shoulder_hours: List of (start_hour, end_hour) tuples for shoulder
            tariff_name: Name identifier for the tariff
        
        Examples:
            >>> scheduler.configure_tou(
            ...     peak_price=0.40,
            ...     off_peak_price=0.15,
            ...     peak_hours=[(14, 20)]  # 2pm - 8pm
            ... )
        """
        from .tou_optimization import TOUTariffConfig, TOUTariff, TOUOptimizer
        
        config = TOUTariffConfig(
            name=tariff_name,
            peak_price=peak_price,
            off_peak_price=off_peak_price,
            shoulder_price=shoulder_price,
            peak_hours=peak_hours or [(14.0, 20.0)],
            shoulder_hours=shoulder_hours or []
        )
        
        self._tou_tariff = TOUTariff(
            config=config,
            period_minutes=self.config.period_minutes
        )
        
        self._tou_optimizer = TOUOptimizer(
            tariff=self._tou_tariff,
            max_evses=54,  # Default, can be overridden
            deferral_safety_margin=self.phase3_config.congestion_safety_margin,
            renewable=self._renewable,
            start_hour=self.phase3_config.simulation_start_hour,
            voltage=self.config.voltage,
            period_minutes=self.config.period_minutes
        )
        
        logger.info(
            f"TOU configured: {tariff_name}, peak=${peak_price:.3f}, "
            f"off_peak=${off_peak_price:.3f}"
        )
    
    def configure_network(
        self,
        phase_a_limit: float = 200.0,
        phase_b_limit: float = 200.0,
        phase_c_limit: float = 200.0,
        total_evses: int = 54
    ) -> None:
        """
        Configure three-phase network infrastructure.
        
        Args:
            phase_a_limit: Phase A transformer limit (Amps)
            phase_b_limit: Phase B transformer limit (Amps)
            phase_c_limit: Phase C transformer limit (Amps)
            total_evses: Total number of EVSEs (must be divisible by 3)
        """
        from .three_phase_network import ThreePhaseNetwork, ThreePhaseNetworkConfig
        
        config = ThreePhaseNetworkConfig(
            total_evses=total_evses,
            evses_per_phase=total_evses // 3,
            phase_a_limit=phase_a_limit,
            phase_b_limit=phase_b_limit,
            phase_c_limit=phase_c_limit,
            voltage=self.config.voltage
        )
        
        self._network = ThreePhaseNetwork(config)
        
        # Update total capacity to match network
        self.config.total_capacity = config.total_capacity
        
        logger.info(
            f"Network configured: {total_evses} EVSEs, "
            f"capacity={config.total_capacity:.0f}A "
            f"({phase_a_limit}/{phase_b_limit}/{phase_c_limit}A per phase)"
        )
    
    def configure_renewables(
        self,
        pv_capacity_kwp: float = 100.0,
        bess_capacity_kwh: float = 200.0,
        bess_max_power_kw: float = 50.0
    ) -> None:
        """
        Configure PV and BESS renewable integration.
        
        Args:
            pv_capacity_kwp: PV system capacity (kWp)
            bess_capacity_kwh: BESS capacity (kWh)
            bess_max_power_kw: BESS max charge/discharge power (kW)
        """
        from .renewable_integration import (
            RenewableIntegration, RenewableIntegrationConfig,
            PVSystemConfig, BESSConfig
        )
        
        config = RenewableIntegrationConfig(
            pv_config=PVSystemConfig(
                capacity_kwp=pv_capacity_kwp,
                inverter_capacity_kw=pv_capacity_kwp
            ),
            bess_config=BESSConfig(
                capacity_kwh=bess_capacity_kwh,
                max_charge_kw=bess_max_power_kw,
                max_discharge_kw=bess_max_power_kw
            )
        )
        
        self._renewable = RenewableIntegration(config)
        
        # Update TOU optimizer if exists
        if self._tou_optimizer:
            self._tou_optimizer.renewable = self._renewable
        
        logger.info(
            f"Renewables configured: PV={pv_capacity_kwp}kWp, "
            f"BESS={bess_capacity_kwh}kWh"
        )
    
    def load_pv_forecast(self, values: List[float], start_period: int = 0) -> None:
        """
        Load PV generation forecast.
        
        Args:
            values: List of generation values (kW) per period
            start_period: Starting period for the forecast
        """
        if self._renewable is None:
            self.configure_renewables()
        self._renewable.load_pv_forecast(values, start_period)
    
    def load_bess_forecast(
        self,
        soc_values: List[float],
        power_values: Optional[List[float]] = None,
        start_period: int = 0
    ) -> None:
        """
        Load BESS state forecast.
        
        Args:
            soc_values: List of SoC values (%) per period
            power_values: Optional list of power values (kW, positive=discharge)
            start_period: Starting period for the forecast
        """
        if self._renewable is None:
            self.configure_renewables()
        self._renewable.load_bess_forecast(soc_values, power_values, start_period)
    
    def get_tou_statistics(self) -> Dict:
        """
        Get TOU optimization statistics.
        
        Returns:
            Dictionary with TOU/deferral statistics
        """
        if self._tou_optimizer is None:
            return {
                'enabled': False,
                'total_deferrals': 0,
                'total_savings': 0.0,
            }
        
        stats = self._tou_optimizer.get_statistics()
        stats['enabled'] = True
        stats['deferral_events'] = len(self._tou_deferral_events)
        stats['currently_deferred'] = len(self._deferred_sessions)
        
        return stats
    
    def get_tou_deferral_history(self) -> List[TOUDeferralEvent]:
        """Get the full TOU deferral event history."""
        return self._tou_deferral_events.copy()
    
    def get_network_status(self) -> Optional[Dict]:
        """
        Get current network phase status.
        
        Returns:
            Dictionary with phase utilization or None if not configured
        """
        if self._network is None:
            return None
        return self._network.get_phase_summary()
    
    def get_renewable_status(self, period: Optional[int] = None) -> Optional[Dict]:
        """
        Get current renewable energy status.
        
        Args:
            period: Specific period to query (uses current_time if None)
        
        Returns:
            Dictionary with PV/BESS status or None if not configured
        """
        if self._renewable is None:
            return None
        return self._renewable.get_renewable_status(
            period or self.current_time
        )
    
    def export_phase3_data(self) -> Dict:
        """
        Export Phase 3 analysis data.
        
        Returns:
            Dictionary with TOU, network, and renewable data
        """
        data = {
            'tou_enabled': self._tou_optimizer is not None,
            'network_enabled': self._network is not None,
            'renewable_enabled': self._renewable is not None,
            'deferral_events': [e.to_dict() for e in self._tou_deferral_events],
            'tou_metrics': [m.to_dict() for m in self.tou_metrics_history],
            'currently_deferred': dict(self._deferred_sessions),
        }
        
        if self._tou_optimizer:
            data['tou_statistics'] = self._tou_optimizer.get_statistics()
        
        if self._network:
            data['network_status'] = self._network.get_phase_summary()
        
        return data
    
    def __repr__(self) -> str:
        tou_status = "TOU" if self._tou_optimizer else "no-TOU"
        quant_status = "quantized" if self.phase4_config.enable_quantization else "continuous"
        return (
            f"AdaptiveQueuingPriorityScheduler("
            f"capacity={self.config.total_capacity}A, "
            f"min_priority_rate={self.config.min_priority_rate}A, "
            f"{tou_status}, {quant_status})"
        )
