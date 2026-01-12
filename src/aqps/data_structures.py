"""
Data structures for Adaptive Queuing Priority Scheduler (AQPS).

This module defines the core data structures used throughout the AQPS algorithm,
including session information, queue entries, and configuration classes.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SessionInfo:
    """
    Information about a single EV charging session.
    
    This is a standalone representation of an EV session, compatible with
    but independent of ACN-Sim's EV class structure.
    
    Attributes:
        session_id: Unique identifier for the session
        station_id: ID of the charging station
        arrival_time: Arrival period (discrete time step)
        departure_time: Estimated departure period
        energy_requested: Total energy requested (kWh)
        requested_energy: Alias for energy_requested (compatibility)
        max_rate: Maximum charging rate (Amps)
        min_rate: Minimum charging rate (Amps)
        current_charge: Current battery charge (kWh)
        is_priority: Whether this session has priority status
        estimated_departure: Alias for departure_time (compatibility)
    """
    session_id: str
    station_id: str
    arrival_time: int
    departure_time: int
    energy_requested: float
    max_rate: float
    min_rate: float = 0.0
    current_charge: float = 0.0
    is_priority: bool = False
    
    @property
    def requested_energy(self) -> float:
        """Alias for energy_requested (compatibility)."""
        return self.energy_requested
    
    @property
    def estimated_departure(self) -> int:
        """Alias for departure_time (compatibility)."""
        return self.departure_time
    
    @property
    def remaining_demand(self) -> float:
        """Calculate remaining energy demand (kWh)."""
        return max(0.0, self.energy_requested - self.current_charge)
    
    def __repr__(self) -> str:
        priority_tag = "[P]" if self.is_priority else "[N]"
        return (f"SessionInfo({priority_tag} {self.session_id}, "
                f"station={self.station_id}, "
                f"demand={self.remaining_demand:.1f}kWh)")


@dataclass
class PriorityQueueEntry:
    """
    Entry in the priority or non-priority queue.
    
    This structure wraps a SessionInfo with additional scheduling metadata
    used by the AQPS algorithm.
    
    Attributes:
        session: The underlying session information
        laxity: Time slack for completing charging (periods)
        is_priority: Whether this is a priority session
        preferred_rate: Desired charging rate based on TOU and constraints
        actual_rate: Actually allocated charging rate
        deferred: Whether charging has been deferred for TOU optimization
    """
    session: SessionInfo
    laxity: float
    is_priority: bool
    preferred_rate: float = 0.0
    actual_rate: float = 0.0
    deferred: bool = False
    
    def __repr__(self) -> str:
        priority_tag = "PRIORITY" if self.is_priority else "NON-PRIORITY"
        return (f"QueueEntry({priority_tag}, {self.session.session_id}, "
                f"laxity={self.laxity:.1f}, rate={self.actual_rate:.1f}A)")


@dataclass
class PriorityConfig:
    """
    Configuration for priority EV selection automation.
    
    This matches the configuration used in the priority_ev_automation module
    for automated priority selection based on session characteristics.
    
    Attributes:
        max_priority_pct: Maximum percentage of sessions that can be priority
        min_energy_kwh: Minimum energy requirement for priority consideration
        max_energy_kwh: Maximum energy for priority (avoid high-demand EVs)
        min_duration_hours: Minimum parking duration for priority
        high_energy_threshold: Energy threshold for "high-demand" classification
        high_energy_min_duration: Min duration required for high-demand EVs
        max_high_energy_pct: Max percentage of high-demand priority EVs
    """
    max_priority_pct: float = 0.27
    min_energy_kwh: float = 10.0
    max_energy_kwh: float = 30.0
    min_duration_hours: float = 2.0
    high_energy_threshold: float = 25.0
    high_energy_min_duration: float = 3.0
    max_high_energy_pct: float = 0.06


@dataclass
class AQPSConfig:
    """
    Configuration for the Adaptive Queuing Priority Scheduler.
    
    This controls the behavior of the AQPS algorithm including priority
    guarantees, capacity limits, and operational modes.
    
    Attributes:
        min_priority_rate: Minimum guaranteed rate for priority EVs (Amps)
        total_capacity: Total available charging capacity (Amps)
        period_minutes: Length of each scheduling period (minutes)
        voltage: Network voltage (Volts)
        enable_logging: Whether to enable detailed logging
        max_priority_ratio: Maximum ratio of priority to total sessions
    """
    min_priority_rate: float = 11.0
    total_capacity: float = 150.0
    period_minutes: float = 5.0
    voltage: float = 220.0
    enable_logging: bool = True
    max_priority_ratio: float = 0.30
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.min_priority_rate <= 0:
            raise ValueError("min_priority_rate must be positive")
        if self.total_capacity <= 0:
            raise ValueError("total_capacity must be positive")
        if self.period_minutes <= 0:
            raise ValueError("period_minutes must be positive")
        if not 0 < self.max_priority_ratio <= 1:
            raise ValueError("max_priority_ratio must be between 0 and 1")


@dataclass
class SchedulingMetrics:
    """
    Metrics collected during a scheduling cycle.
    
    These metrics are used for performance analysis and debugging.
    
    Attributes:
        timestamp: Current simulation time (period)
        priority_sessions_active: Number of active priority sessions
        non_priority_sessions_active: Number of active non-priority sessions
        total_allocated_capacity: Total capacity allocated (Amps)
        priority_allocated_capacity: Capacity allocated to priority (Amps)
        priority_sessions_at_min: Number of priority EVs at minimum rate
        priority_sessions_at_max: Number of priority EVs at maximum rate
        warnings: List of warning messages from this cycle
    """
    timestamp: int
    priority_sessions_active: int = 0
    non_priority_sessions_active: int = 0
    total_allocated_capacity: float = 0.0
    priority_allocated_capacity: float = 0.0
    priority_sessions_at_min: int = 0
    priority_sessions_at_max: int = 0
    capacity_utilization: float = 0.0
    warnings: list = field(default_factory=list)
    
    def add_warning(self, message: str) -> None:
        """Add a warning message to this cycle's metrics."""
        self.warnings.append(message)
    
    def __repr__(self) -> str:
        return (f"Metrics(t={self.timestamp}, "
                f"priority={self.priority_sessions_active}, "
                f"non_priority={self.non_priority_sessions_active}, "
                f"capacity={self.total_allocated_capacity:.1f}A)")
