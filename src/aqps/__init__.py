"""
Adaptive Queuing Priority Scheduler (AQPS)

A computationally efficient, real-time deployable scheduler for priority-aware
EV fleet charging. AQPS provides an O(n log n) heuristic alternative to 
MPC-based approaches while maintaining priority guarantees and cost optimization.

Main Components:
- AdaptiveQueuingPriorityScheduler: Main scheduling algorithm
- ScenarioGenerator: Generate S1-S6 test scenarios
- SessionInfo: EV session data structure
- QueueManager: Two-tier queue management
- AQPSConfig: Configuration parameters

Quick Start:
    >>> from aqps import AdaptiveQueuingPriorityScheduler, AQPSConfig, generate_scenario
    >>> 
    >>> config = AQPSConfig(min_priority_rate=11.0, total_capacity=150.0)
    >>> scheduler = AdaptiveQueuingPriorityScheduler(config)
    >>> 
    >>> sessions = generate_scenario('S1', n_sessions=100, seed=42)
    >>> schedule = scheduler.schedule(sessions, current_time=0)

Author: Research Team
Version: 0.1.0 (Phase 1)
"""

from .scheduler import AdaptiveQueuingPriorityScheduler
from .data_structures import (
    SessionInfo,
    PriorityQueueEntry,
    PriorityConfig,
    AQPSConfig,
    SchedulingMetrics
)
from .queue_manager import QueueManager
from .scenario_generator import ScenarioGenerator, generate_scenario
from .utils import (
    calculate_laxity,
    calculate_remaining_energy,
    energy_deliverable_at_rate,
    rate_needed_for_energy
)

__version__ = "0.1.0"
__author__ = "Research Team"

__all__ = [
    # Main scheduler
    'AdaptiveQueuingPriorityScheduler',
    
    # Data structures
    'SessionInfo',
    'PriorityQueueEntry',
    'PriorityConfig',
    'AQPSConfig',
    'SchedulingMetrics',
    
    # Queue management
    'QueueManager',
    
    # Scenario generation
    'ScenarioGenerator',
    'generate_scenario',
    
    # Utilities
    'calculate_laxity',
    'calculate_remaining_energy',
    'energy_deliverable_at_rate',
    'rate_needed_for_energy',
]
