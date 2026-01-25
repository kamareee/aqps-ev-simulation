"""
Preemption logic for Adaptive Queuing Priority Scheduler (AQPS).

This module implements the two-stage preemption mechanism:
- Option B (Primary): Highest-laxity-first preemption
- Option A (Fallback): Proportional reduction of all non-priority EVs

The preemption system ensures priority EVs can always obtain their
guaranteed minimum charging rate by reducing non-priority allocations.

Author: Research Team
Phase: 2 (Queue Management & Preemption)
"""

import logging
from typing import Dict, List, Tuple, Optional
from .data_structures import (
    PriorityQueueEntry,
    PreemptionEvent,
    PreemptionMethod
)

logger = logging.getLogger(__name__)


class PreemptionManager:
    """
    Manages preemption of non-priority EVs to guarantee priority EV rates.
    
    The PreemptionManager implements a two-stage preemption policy:
    
    Stage 1 (Option B - Highest Laxity First):
        - Sort non-priority EVs by laxity in descending order
        - Preempt EVs with highest laxity first (most flexible)
        - Reduce each victim's rate to zero (complete preemption allowed)
        - Stop when sufficient capacity is freed
    
    Stage 2 (Option A - Proportional Fallback):
        - If Option B doesn't free enough capacity
        - Proportionally reduce ALL remaining non-priority EVs
        - Each EV reduced by: (shortfall / total_reducible) * reducible_amount
    
    Attributes:
        preemption_history: List of all preemption events
        total_preemptions: Running count of preemption events
        option_b_count: Count of Option B (highest laxity) preemptions
        option_a_count: Count of Option A (proportional) preemptions
        min_priority_rate: Minimum rate guaranteed to priority EVs
    """
    
    def __init__(self, min_priority_rate: float = 11.0):
        """
        Initialize the preemption manager.
        
        Args:
            min_priority_rate: Minimum rate guaranteed to priority EVs (Amps)
        """
        self.min_priority_rate = min_priority_rate
        self.preemption_history: List[PreemptionEvent] = []
        self.total_preemptions: int = 0
        self.option_b_count: int = 0
        self.option_a_count: int = 0
    
    def check_preemption_needed(
        self,
        priority_entry: PriorityQueueEntry,
        current_allocation: float,
        min_required_rate: float,
        remaining_capacity: float
    ) -> Tuple[bool, float]:
        """
        Check if preemption is needed for a priority EV.
        
        Preemption is needed when:
        - Current allocation < minimum required rate, AND
        - Remaining capacity is insufficient to meet the minimum
        
        Args:
            priority_entry: The priority EV queue entry
            current_allocation: Current allocated rate for this EV (Amps)
            min_required_rate: Minimum rate required (Amps)
            remaining_capacity: Remaining system capacity (Amps)
        
        Returns:
            Tuple of (preemption_needed: bool, capacity_shortfall: float)
        """
        rate_deficit = min_required_rate - current_allocation
        
        if rate_deficit <= 0:
            # Already meeting minimum rate
            return False, 0.0
        
        if remaining_capacity >= rate_deficit:
            # Enough capacity available without preemption
            return False, 0.0
        
        # Need preemption - calculate shortfall
        shortfall = rate_deficit - remaining_capacity
        return True, shortfall
    
    def execute_preemption(
        self,
        needed_capacity: float,
        priority_entry: PriorityQueueEntry,
        non_priority_queue: List[PriorityQueueEntry],
        schedule: Dict[str, float],
        current_time: int
    ) -> Tuple[Dict[str, float], PreemptionEvent]:
        """
        Execute preemption to free capacity for a priority EV.
        
        This is the main entry point for preemption. It attempts Option B
        first, then falls back to Option A if needed.
        
        Args:
            needed_capacity: Amount of capacity needed (Amps)
            priority_entry: The priority EV that needs capacity
            non_priority_queue: List of non-priority queue entries
            schedule: Current schedule (station_id → rate)
            current_time: Current simulation time (period)
        
        Returns:
            Tuple of (updated_schedule, PreemptionEvent)
        """
        if needed_capacity <= 0:
            # No preemption needed
            return schedule, PreemptionEvent(
                timestamp=current_time,
                priority_session_id=priority_entry.session.session_id,
                preempted_session_ids=[],
                capacity_needed=0.0,
                capacity_freed=0.0,
                method=PreemptionMethod.NONE,
                priority_laxity=priority_entry.laxity,
                success=True
            )
        
        # Initialize tracking
        freed_capacity = 0.0
        preempted_ids: List[str] = []
        rate_reductions: Dict[str, float] = {}
        victim_laxities: Dict[str, float] = {}
        method_used = PreemptionMethod.HIGHEST_LAXITY
        
        # Get non-priority EVs with their current allocations
        non_priority_with_allocation = [
            (entry, schedule.get(entry.session.station_id, 0.0))
            for entry in non_priority_queue
        ]
        
        # Filter to only those with positive allocations
        reducible_entries = [
            (entry, rate) for entry, rate in non_priority_with_allocation
            if rate > 0
        ]
        
        if not reducible_entries:
            # No non-priority EVs to preempt
            logger.warning(
                f"Preemption requested but no non-priority EVs have allocations"
            )
            return schedule, PreemptionEvent(
                timestamp=current_time,
                priority_session_id=priority_entry.session.session_id,
                preempted_session_ids=[],
                capacity_needed=needed_capacity,
                capacity_freed=0.0,
                method=PreemptionMethod.NONE,
                priority_laxity=priority_entry.laxity,
                success=False
            )
        
        # Option B: Highest laxity first (most flexible EVs)
        sorted_by_laxity = sorted(
            reducible_entries,
            key=lambda x: x[0].laxity,
            reverse=True  # Highest laxity first
        )
        
        logger.debug(
            f"Option B: Attempting to free {needed_capacity:.1f}A from "
            f"{len(sorted_by_laxity)} non-priority EVs"
        )
        
        for entry, current_rate in sorted_by_laxity:
            if freed_capacity >= needed_capacity:
                break
            
            session = entry.session
            station_id = session.station_id
            
            # Option A design: Can reduce to 0 (complete preemption allowed)
            min_rate = 0.0
            reducible = current_rate - min_rate
            
            if reducible > 0:
                # Calculate reduction amount
                still_needed = needed_capacity - freed_capacity
                reduction = min(reducible, still_needed)
                
                # Apply reduction
                new_rate = current_rate - reduction
                schedule[station_id] = new_rate
                entry.actual_rate = new_rate
                
                # Track the preemption
                freed_capacity += reduction
                preempted_ids.append(session.session_id)
                rate_reductions[session.session_id] = reduction
                victim_laxities[session.session_id] = entry.laxity
                
                logger.debug(
                    f"  Preempted {session.session_id}: "
                    f"{current_rate:.1f}A → {new_rate:.1f}A "
                    f"(freed {reduction:.1f}A, laxity={entry.laxity:.1f})"
                )
        
        # Option A Fallback: Proportional reduction if Option B insufficient
        if freed_capacity < needed_capacity:
            shortfall = needed_capacity - freed_capacity
            method_used = PreemptionMethod.COMBINED
            
            logger.debug(
                f"Option A fallback: {shortfall:.1f}A still needed after Option B"
            )
            
            # Calculate total reducible capacity from remaining EVs
            remaining_reducible = []
            for entry, _ in sorted_by_laxity:
                station_id = entry.session.station_id
                current_rate = schedule.get(station_id, 0.0)
                if current_rate > 0:
                    remaining_reducible.append((entry, current_rate))
            
            total_remaining = sum(rate for _, rate in remaining_reducible)
            
            if total_remaining > 0:
                # Calculate proportional reduction ratio
                ratio = min(1.0, shortfall / total_remaining)
                
                for entry, current_rate in remaining_reducible:
                    session = entry.session
                    station_id = session.station_id
                    
                    # Proportional reduction
                    reduction = current_rate * ratio
                    new_rate = current_rate - reduction
                    
                    schedule[station_id] = new_rate
                    entry.actual_rate = new_rate
                    
                    freed_capacity += reduction
                    
                    # Track if not already preempted
                    if session.session_id not in preempted_ids:
                        preempted_ids.append(session.session_id)
                        victim_laxities[session.session_id] = entry.laxity
                    
                    # Update rate reductions
                    if session.session_id in rate_reductions:
                        rate_reductions[session.session_id] += reduction
                    else:
                        rate_reductions[session.session_id] = reduction
                
                self.option_a_count += 1
        else:
            self.option_b_count += 1
        
        # Check success
        success = freed_capacity >= needed_capacity - 0.01  # Small tolerance
        
        # Create preemption event
        event = PreemptionEvent(
            timestamp=current_time,
            priority_session_id=priority_entry.session.session_id,
            preempted_session_ids=preempted_ids,
            capacity_needed=needed_capacity,
            capacity_freed=freed_capacity,
            method=method_used,
            rate_reductions=rate_reductions,
            priority_laxity=priority_entry.laxity,
            victim_laxities=victim_laxities,
            success=success
        )
        
        # Record in history
        self.preemption_history.append(event)
        self.total_preemptions += 1
        
        logger.info(
            f"Preemption complete: freed {freed_capacity:.1f}A "
            f"(needed {needed_capacity:.1f}A), {len(preempted_ids)} victims, "
            f"method={method_used.value}"
        )
        
        return schedule, event
    
    def preempt_single_highest_laxity(
        self,
        needed_capacity: float,
        non_priority_queue: List[PriorityQueueEntry],
        schedule: Dict[str, float],
        current_time: int
    ) -> Tuple[Optional[str], float]:
        """
        Preempt a single non-priority EV with highest laxity.
        
        Used for incremental preemption during allocation.
        
        Args:
            needed_capacity: Capacity needed (Amps)
            non_priority_queue: Non-priority sessions to consider
            schedule: Current schedule
            current_time: Current time (for logging)
        
        Returns:
            Tuple of (preempted_session_id, freed_capacity) or (None, 0.0)
        """
        if not non_priority_queue:
            return None, 0.0
        
        # Find session with highest laxity that has reducible capacity
        best_entry = None
        best_laxity = float('-inf')
        
        for entry in non_priority_queue:
            current_rate = schedule.get(entry.session.station_id, 0.0)
            if current_rate > 0 and entry.laxity > best_laxity:
                best_laxity = entry.laxity
                best_entry = entry
        
        if best_entry is None:
            return None, 0.0
        
        # Reduce this session
        session_id = best_entry.session.session_id
        station_id = best_entry.session.station_id
        current_rate = schedule.get(station_id, 0.0)
        
        # Calculate reduction (can go to 0)
        reduction = min(current_rate, needed_capacity)
        schedule[station_id] = current_rate - reduction
        best_entry.actual_rate = current_rate - reduction
        
        logger.debug(
            f"Single preemption: {session_id} reduced by {reduction:.1f}A "
            f"(laxity={best_laxity:.1f})"
        )
        
        return session_id, reduction
    
    def get_statistics(self) -> Dict:
        """
        Get preemption statistics.
        
        Returns:
            Dictionary with preemption statistics for analysis
        """
        if not self.preemption_history:
            return {
                'total_preemptions': 0,
                'option_b_count': 0,
                'option_a_count': 0,
                'success_rate': 1.0,
                'avg_capacity_freed': 0.0,
                'avg_victims_per_event': 0.0,
                'total_capacity_freed': 0.0,
                'total_victims': 0,
            }
        
        successful = sum(1 for e in self.preemption_history if e.success)
        total_freed = sum(e.capacity_freed for e in self.preemption_history)
        total_victims = sum(e.num_victims for e in self.preemption_history)
        
        return {
            'total_preemptions': self.total_preemptions,
            'option_b_count': self.option_b_count,
            'option_a_count': self.option_a_count,
            'success_rate': successful / len(self.preemption_history),
            'avg_capacity_freed': total_freed / len(self.preemption_history),
            'avg_victims_per_event': total_victims / len(self.preemption_history),
            'total_capacity_freed': total_freed,
            'total_victims': total_victims,
            'unique_victims': len(set(
                sid for e in self.preemption_history
                for sid in e.preempted_session_ids
            )),
        }
    
    def get_preemption_history(self) -> List[PreemptionEvent]:
        """Get the full preemption history."""
        return self.preemption_history.copy()
    
    def get_preemption_dataframe_data(self) -> List[Dict]:
        """
        Get preemption history as list of dicts for DataFrame creation.
        
        Returns:
            List of dictionaries suitable for pandas DataFrame
        """
        return [event.to_dict() for event in self.preemption_history]
    
    def reset(self) -> None:
        """Reset preemption history and counters."""
        self.preemption_history.clear()
        self.total_preemptions = 0
        self.option_b_count = 0
        self.option_a_count = 0
        logger.info("Preemption manager reset")
    
    def __repr__(self) -> str:
        return (
            f"PreemptionManager("
            f"events={len(self.preemption_history)}, "
            f"min_rate={self.min_priority_rate}A)"
        )
