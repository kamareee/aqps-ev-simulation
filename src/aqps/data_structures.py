"""
Data structures for Adaptive Queuing Priority Scheduler (AQPS).

This module defines the core data structures used throughout the AQPS algorithm,
including session information, queue entries, configuration classes, and
Phase 2 preemption/threshold tracking structures.

Author: Research Team
Phase: 2 (Queue Management & Preemption)
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict
from enum import Enum


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


# =============================================================================
# Phase 2: Preemption Data Structures
# =============================================================================

class PreemptionMethod(Enum):
    """Method used for preemption."""
    HIGHEST_LAXITY = "highest_laxity"  # Option B: Preempt most flexible EV first
    PROPORTIONAL = "proportional"       # Option A: Reduce all non-priority proportionally
    COMBINED = "combined"               # Both methods used in sequence
    NONE = "none"                        # No preemption needed


@dataclass
class PreemptionEvent:
    """
    Record of a preemption event for analysis and visualization.
    
    This dataclass captures all details of a preemption event, enabling
    post-simulation analysis and journal-quality visualizations.
    
    Attributes:
        timestamp: Simulation time when preemption occurred (period)
        priority_session_id: Session ID of the priority EV that triggered preemption
        preempted_session_ids: List of non-priority session IDs affected
        capacity_needed: Amount of capacity needed by priority EV (Amps)
        capacity_freed: Amount of capacity actually freed (Amps)
        method: PreemptionMethod used (HIGHEST_LAXITY, PROPORTIONAL, or COMBINED)
        rate_reductions: Dict mapping session_id → rate reduction amount (Amps)
        priority_laxity: Laxity of the priority EV at time of preemption
        victim_laxities: Dict mapping preempted session_id → their laxity
        success: Whether preemption successfully freed enough capacity
    """
    timestamp: int
    priority_session_id: str
    preempted_session_ids: List[str]
    capacity_needed: float
    capacity_freed: float
    method: PreemptionMethod
    rate_reductions: Dict[str, float] = field(default_factory=dict)
    priority_laxity: float = 0.0
    victim_laxities: Dict[str, float] = field(default_factory=dict)
    success: bool = True
    
    @property
    def num_victims(self) -> int:
        """Number of sessions affected by preemption."""
        return len(self.preempted_session_ids)
    
    @property
    def capacity_shortfall(self) -> float:
        """Capacity still needed after preemption (0 if successful)."""
        return max(0.0, self.capacity_needed - self.capacity_freed)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for DataFrame/JSON export."""
        return {
            'timestamp': self.timestamp,
            'priority_session_id': self.priority_session_id,
            'num_victims': self.num_victims,
            'capacity_needed': self.capacity_needed,
            'capacity_freed': self.capacity_freed,
            'method': self.method.value,
            'success': self.success,
            'priority_laxity': self.priority_laxity,
            'preempted_session_ids': self.preempted_session_ids.copy(),
        }
    
    def __repr__(self) -> str:
        status = "SUCCESS" if self.success else "PARTIAL"
        return (f"PreemptionEvent(t={self.timestamp}, {status}, "
                f"method={self.method.value}, "
                f"freed={self.capacity_freed:.1f}A/{self.capacity_needed:.1f}A, "
                f"victims={self.num_victims})")


@dataclass
class ThresholdViolationEvent:
    """
    Record of a threshold violation when priority guarantees cannot be met.
    
    This occurs when the total minimum rate required for priority EVs
    exceeds the available system capacity, even after preemption.
    
    Attributes:
        timestamp: Simulation time when violation occurred (period)
        priority_count: Number of active priority EVs
        non_priority_count: Number of active non-priority EVs
        capacity_required: Total capacity needed for priority minimums (Amps)
        capacity_available: Total system capacity (Amps)
        capacity_shortfall: How much capacity is lacking (Amps)
        affected_session_ids: Priority EVs that couldn't get minimum rate
        system_utilization: Overall utilization at time of violation
        preemption_attempted: Whether preemption was attempted before violation
    """
    timestamp: int
    priority_count: int
    non_priority_count: int
    capacity_required: float
    capacity_available: float
    capacity_shortfall: float
    affected_session_ids: List[str] = field(default_factory=list)
    system_utilization: float = 100.0
    preemption_attempted: bool = False
    
    @property
    def severity(self) -> str:
        """Categorize severity of the violation."""
        ratio = self.capacity_shortfall / self.capacity_available if self.capacity_available > 0 else 1.0
        if ratio < 0.1:
            return "minor"
        elif ratio < 0.25:
            return "moderate"
        else:
            return "severe"
    
    @property
    def priority_ratio(self) -> float:
        """Ratio of priority to total sessions."""
        total = self.priority_count + self.non_priority_count
        return self.priority_count / total if total > 0 else 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for DataFrame/JSON export."""
        return {
            'timestamp': self.timestamp,
            'priority_count': self.priority_count,
            'non_priority_count': self.non_priority_count,
            'total_sessions': self.priority_count + self.non_priority_count,
            'capacity_required': self.capacity_required,
            'capacity_available': self.capacity_available,
            'capacity_shortfall': self.capacity_shortfall,
            'severity': self.severity,
            'priority_ratio': self.priority_ratio,
            'system_utilization': self.system_utilization,
            'preemption_attempted': self.preemption_attempted,
            'num_affected': len(self.affected_session_ids),
        }
    
    def __repr__(self) -> str:
        return (f"ThresholdViolation(t={self.timestamp}, {self.severity.upper()}, "
                f"shortfall={self.capacity_shortfall:.1f}A, "
                f"priority_ratio={self.priority_ratio:.1%})")


# =============================================================================
# Phase 3: TOU Optimization Data Structures
# =============================================================================

@dataclass
class TOUDeferralEvent:
    """
    Record of a TOU-based charging deferral event.
    
    This dataclass captures when a non-priority EV's charging is deferred
    to a cheaper TOU period.
    
    Attributes:
        timestamp: Simulation time when deferral was decided (period)
        session_id: Session ID of the deferred EV
        station_id: Station ID where EV is connected
        original_period: Period when charging would have occurred
        target_period: Period to which charging is deferred
        original_price: TOU price at original period ($/kWh)
        target_price: TOU price at target period ($/kWh)
        savings_per_kwh: Cost savings per kWh from deferral
        energy_deferred_kwh: Amount of energy deferred (kWh)
        reason: Reason for deferral decision
    """
    timestamp: int
    session_id: str
    station_id: str
    original_period: int
    target_period: int
    original_price: float
    target_price: float
    savings_per_kwh: float
    energy_deferred_kwh: float = 0.0
    reason: str = "TOU optimization"
    
    @property
    def total_savings(self) -> float:
        """Calculate total cost savings from this deferral."""
        return self.savings_per_kwh * self.energy_deferred_kwh
    
    @property
    def deferral_periods(self) -> int:
        """Number of periods charging is deferred."""
        return self.target_period - self.original_period
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for DataFrame/JSON export."""
        return {
            'timestamp': self.timestamp,
            'session_id': self.session_id,
            'station_id': self.station_id,
            'original_period': self.original_period,
            'target_period': self.target_period,
            'original_price': self.original_price,
            'target_price': self.target_price,
            'savings_per_kwh': self.savings_per_kwh,
            'energy_deferred_kwh': self.energy_deferred_kwh,
            'total_savings': self.total_savings,
            'deferral_periods': self.deferral_periods,
            'reason': self.reason,
        }
    
    def __repr__(self) -> str:
        return (f"TOUDeferralEvent(t={self.timestamp}, {self.session_id}, "
                f"defer {self.deferral_periods} periods, "
                f"save ${self.savings_per_kwh:.3f}/kWh)")


@dataclass
class Phase3Config:
    """
    Configuration for Phase 3 TOU optimization features.
    
    This extends AQPSConfig with TOU-specific settings.
    
    Attributes:
        enable_tou_optimization: Enable TOU-based deferral
        enable_renewable_integration: Enable PV/BESS integration
        deferral_policy: 'aggressive', 'conservative', or 'balanced'
        apply_to_priority: Whether to apply TOU deferral to priority EVs
        min_savings_threshold: Minimum savings to trigger deferral ($/kWh)
        max_deferral_periods: Maximum periods to defer charging
        congestion_safety_margin: Safety margin for deferral windows (0-1)
        simulation_start_hour: Hour at which simulation starts (0-24)
    """
    enable_tou_optimization: bool = True
    enable_renewable_integration: bool = True
    deferral_policy: str = "aggressive"  # aggressive, conservative, balanced
    apply_to_priority: bool = False  # Never defer priority (per design)
    min_savings_threshold: float = 0.0  # Defer for any savings in aggressive mode
    max_deferral_periods: int = 288  # Max 24 hours lookahead
    congestion_safety_margin: float = 0.8  # 80% EVSE utilization cap
    simulation_start_hour: float = 0.0  # Midnight start default
    
    def validate(self) -> List[str]:
        """Validate Phase 3 configuration."""
        errors = []
        
        if self.deferral_policy not in ['aggressive', 'conservative', 'balanced']:
            errors.append(f"Invalid deferral_policy: {self.deferral_policy}")
        
        if not 0 < self.congestion_safety_margin <= 1:
            errors.append("congestion_safety_margin must be between 0 and 1")
        
        if self.min_savings_threshold < 0:
            errors.append("min_savings_threshold must be non-negative")
        
        return errors


@dataclass
class TOUMetrics:
    """
    Metrics for TOU optimization performance.
    
    Attributes:
        timestamp: Current simulation time (period)
        current_tou_period: Current TOU period type (peak/shoulder/off_peak)
        current_price: Current electricity price ($/kWh)
        sessions_deferred: Number of sessions currently deferred
        sessions_charging_deferred: Sessions charging that were previously deferred
        total_deferrals: Cumulative deferral count
        total_savings_kwh: Cumulative savings ($/kWh accumulated)
        avg_deferral_duration: Average deferral duration (periods)
        pv_generation_kw: Current PV generation (kW)
        bess_power_kw: Current BESS power flow (kW)
    """
    timestamp: int
    current_tou_period: str = "off_peak"
    current_price: float = 0.0
    sessions_deferred: int = 0
    sessions_charging_deferred: int = 0
    total_deferrals: int = 0
    total_savings_kwh: float = 0.0
    avg_deferral_duration: float = 0.0
    pv_generation_kw: float = 0.0
    bess_power_kw: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for export."""
        return {
            'timestamp': self.timestamp,
            'current_tou_period': self.current_tou_period,
            'current_price': self.current_price,
            'sessions_deferred': self.sessions_deferred,
            'sessions_charging_deferred': self.sessions_charging_deferred,
            'total_deferrals': self.total_deferrals,
            'total_savings_kwh': self.total_savings_kwh,
            'avg_deferral_duration': self.avg_deferral_duration,
            'pv_generation_kw': self.pv_generation_kw,
            'bess_power_kw': self.bess_power_kw,
        }
    
    def __repr__(self) -> str:
        return (f"TOUMetrics(t={self.timestamp}, {self.current_tou_period}, "
                f"${self.current_price:.3f}/kWh, "
                f"deferred={self.sessions_deferred})")


# =============================================================================
# Phase 4: Quantization & Polish Data Structures
# =============================================================================

# Standard J1772 pilot signals (Amps)
J1772_PILOT_SIGNALS = [0.0, 8.0, 16.0, 24.0, 32.0]


@dataclass
class QuantizationEvent:
    """
    Record of a quantization adjustment for a single session.
    
    Attributes:
        session_id: Session ID being quantized
        station_id: Station ID
        is_priority: Whether this is a priority session
        pre_quantization_rate: Rate before quantization (Amps)
        post_quantization_rate: Rate after quantization (Amps)
        quantization_method: 'floor' or 'ceil' (ceil for priority guarantee)
        rate_change: Change in rate (negative = reduction)
        reason: Reason for quantization method choice
    """
    session_id: str
    station_id: str
    is_priority: bool
    pre_quantization_rate: float
    post_quantization_rate: float
    quantization_method: str  # 'floor' or 'ceil'
    rate_change: float = 0.0
    reason: str = "standard"
    
    def __post_init__(self):
        self.rate_change = self.post_quantization_rate - self.pre_quantization_rate
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for export."""
        return {
            'session_id': self.session_id,
            'station_id': self.station_id,
            'is_priority': self.is_priority,
            'pre_quantization_rate': self.pre_quantization_rate,
            'post_quantization_rate': self.post_quantization_rate,
            'quantization_method': self.quantization_method,
            'rate_change': self.rate_change,
            'reason': self.reason,
        }
    
    def __repr__(self) -> str:
        direction = "↑" if self.rate_change > 0 else "↓" if self.rate_change < 0 else "="
        return (f"QuantizationEvent({self.session_id}, "
                f"{self.pre_quantization_rate:.1f}A {direction} "
                f"{self.post_quantization_rate:.1f}A, {self.quantization_method})")


@dataclass
class QuantizationMetrics:
    """
    Metrics for quantization performance at a single timestep.
    
    Attributes:
        timestamp: Simulation time (period)
        total_sessions: Total sessions quantized
        priority_sessions: Number of priority sessions
        non_priority_sessions: Number of non-priority sessions
        pre_quantization_capacity: Total capacity before quantization (Amps)
        post_quantization_capacity: Total capacity after quantization (Amps)
        capacity_lost: Capacity lost due to floor quantization (Amps)
        capacity_gained: Capacity gained due to ceil quantization (Amps)
        priority_ceil_count: Number of priority sessions rounded up
        energy_loss_kwh: Estimated energy loss from quantization (kWh)
        events: List of individual quantization events
    """
    timestamp: int
    total_sessions: int = 0
    priority_sessions: int = 0
    non_priority_sessions: int = 0
    pre_quantization_capacity: float = 0.0
    post_quantization_capacity: float = 0.0
    capacity_lost: float = 0.0
    capacity_gained: float = 0.0
    priority_ceil_count: int = 0
    energy_loss_kwh: float = 0.0
    events: List[QuantizationEvent] = field(default_factory=list)
    
    @property
    def net_capacity_change(self) -> float:
        """Net change in capacity from quantization."""
        return self.post_quantization_capacity - self.pre_quantization_capacity
    
    @property
    def quantization_efficiency(self) -> float:
        """Percentage of capacity retained after quantization."""
        if self.pre_quantization_capacity <= 0:
            return 100.0
        return (self.post_quantization_capacity / self.pre_quantization_capacity) * 100.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for export."""
        return {
            'timestamp': self.timestamp,
            'total_sessions': self.total_sessions,
            'priority_sessions': self.priority_sessions,
            'non_priority_sessions': self.non_priority_sessions,
            'pre_quantization_capacity': self.pre_quantization_capacity,
            'post_quantization_capacity': self.post_quantization_capacity,
            'capacity_lost': self.capacity_lost,
            'capacity_gained': self.capacity_gained,
            'net_capacity_change': self.net_capacity_change,
            'priority_ceil_count': self.priority_ceil_count,
            'quantization_efficiency': self.quantization_efficiency,
            'energy_loss_kwh': self.energy_loss_kwh,
        }
    
    def __repr__(self) -> str:
        return (f"QuantizationMetrics(t={self.timestamp}, "
                f"{self.pre_quantization_capacity:.1f}A → "
                f"{self.post_quantization_capacity:.1f}A, "
                f"efficiency={self.quantization_efficiency:.1f}%)")


@dataclass
class ComputationalMetrics:
    """
    Metrics for computational performance of the scheduler.
    
    Used to validate O(n log n) complexity claims and benchmark against MPC.
    
    Attributes:
        timestamp: Simulation time (period)
        schedule_time_ms: Total time for schedule() call (milliseconds)
        num_sessions: Number of sessions processed
        num_priority: Number of priority sessions
        num_non_priority: Number of non-priority sessions
        preemption_occurred: Whether preemption was triggered
        quantization_enabled: Whether quantization was applied
    """
    timestamp: int
    schedule_time_ms: float = 0.0
    num_sessions: int = 0
    num_priority: int = 0
    num_non_priority: int = 0
    preemption_occurred: bool = False
    quantization_enabled: bool = True
    
    @property
    def time_per_session_us(self) -> float:
        """Average time per session in microseconds."""
        if self.num_sessions <= 0:
            return 0.0
        return (self.schedule_time_ms * 1000.0) / self.num_sessions
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for export."""
        return {
            'timestamp': self.timestamp,
            'schedule_time_ms': self.schedule_time_ms,
            'num_sessions': self.num_sessions,
            'num_priority': self.num_priority,
            'num_non_priority': self.num_non_priority,
            'time_per_session_us': self.time_per_session_us,
            'preemption_occurred': self.preemption_occurred,
            'quantization_enabled': self.quantization_enabled,
        }
    
    def __repr__(self) -> str:
        return (f"ComputationalMetrics(t={self.timestamp}, "
                f"{self.schedule_time_ms:.3f}ms, "
                f"n={self.num_sessions}, "
                f"{self.time_per_session_us:.1f}µs/session)")


@dataclass
class Phase4Config:
    """
    Configuration for Phase 4 quantization and benchmarking features.
    
    Attributes:
        enable_quantization: Enable J1772 pilot signal quantization
        pilot_signals: Valid pilot signal levels (Amps)
        priority_ceil_enabled: Round UP priority EVs if floor violates minimum
        track_quantization_events: Store individual quantization events
        enable_timing: Enable computational timing metrics
    """
    enable_quantization: bool = True
    pilot_signals: List[float] = field(default_factory=lambda: J1772_PILOT_SIGNALS.copy())
    priority_ceil_enabled: bool = True  # Round UP for priority if needed
    track_quantization_events: bool = True
    enable_timing: bool = True
    
    def get_floor_signal(self, rate: float) -> float:
        """Get the floor (round down) pilot signal for a rate."""
        if rate <= 0:
            return 0.0
        valid = [s for s in self.pilot_signals if s <= rate]
        return max(valid) if valid else 0.0
    
    def get_ceil_signal(self, rate: float) -> float:
        """Get the ceiling (round up) pilot signal for a rate."""
        if rate <= 0:
            return 0.0
        valid = [s for s in self.pilot_signals if s >= rate]
        return min(valid) if valid else max(self.pilot_signals)
    
    def validate(self) -> List[str]:
        """Validate Phase 4 configuration."""
        errors = []
        
        if not self.pilot_signals:
            errors.append("pilot_signals cannot be empty")
        
        if 0.0 not in self.pilot_signals:
            errors.append("pilot_signals must include 0.0")
        
        # Check signals are sorted
        if self.pilot_signals != sorted(self.pilot_signals):
            errors.append("pilot_signals must be sorted in ascending order")
        
        return errors


@dataclass 
class SimulationSummary:
    """
    Aggregate summary statistics across a complete simulation run.
    
    Attributes:
        total_timesteps: Number of scheduling cycles
        total_sessions_served: Unique sessions that received charging
        priority_fulfillment_rate: Percentage of priority EVs fully served
        non_priority_fulfillment_rate: Percentage of non-priority EVs served
        total_preemptions: Total preemption events
        total_threshold_violations: Total threshold violation events
        total_deferrals: Total TOU deferral events
        avg_schedule_time_ms: Average scheduling time (ms)
        max_schedule_time_ms: Maximum scheduling time (ms)
        total_energy_delivered_kwh: Total energy delivered
        total_energy_cost: Total energy cost ($)
        quantization_efficiency_avg: Average quantization efficiency (%)
    """
    total_timesteps: int = 0
    total_sessions_served: int = 0
    priority_sessions_total: int = 0
    priority_sessions_fulfilled: int = 0
    non_priority_sessions_total: int = 0
    non_priority_energy_delivered_pct: float = 0.0
    total_preemptions: int = 0
    total_threshold_violations: int = 0
    total_deferrals: int = 0
    avg_schedule_time_ms: float = 0.0
    max_schedule_time_ms: float = 0.0
    min_schedule_time_ms: float = 0.0
    total_energy_delivered_kwh: float = 0.0
    total_energy_cost: float = 0.0
    quantization_efficiency_avg: float = 100.0
    avg_capacity_utilization: float = 0.0
    
    @property
    def priority_fulfillment_rate(self) -> float:
        """Percentage of priority EVs that were fully served."""
        if self.priority_sessions_total <= 0:
            return 100.0
        return (self.priority_sessions_fulfilled / self.priority_sessions_total) * 100.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for export."""
        return {
            'total_timesteps': self.total_timesteps,
            'total_sessions_served': self.total_sessions_served,
            'priority_sessions_total': self.priority_sessions_total,
            'priority_sessions_fulfilled': self.priority_sessions_fulfilled,
            'priority_fulfillment_rate': self.priority_fulfillment_rate,
            'non_priority_sessions_total': self.non_priority_sessions_total,
            'non_priority_energy_delivered_pct': self.non_priority_energy_delivered_pct,
            'total_preemptions': self.total_preemptions,
            'total_threshold_violations': self.total_threshold_violations,
            'total_deferrals': self.total_deferrals,
            'avg_schedule_time_ms': self.avg_schedule_time_ms,
            'max_schedule_time_ms': self.max_schedule_time_ms,
            'min_schedule_time_ms': self.min_schedule_time_ms,
            'total_energy_delivered_kwh': self.total_energy_delivered_kwh,
            'total_energy_cost': self.total_energy_cost,
            'quantization_efficiency_avg': self.quantization_efficiency_avg,
            'avg_capacity_utilization': self.avg_capacity_utilization,
        }
    
    def __repr__(self) -> str:
        return (f"SimulationSummary(timesteps={self.total_timesteps}, "
                f"priority_rate={self.priority_fulfillment_rate:.1f}%, "
                f"preemptions={self.total_preemptions}, "
                f"avg_time={self.avg_schedule_time_ms:.3f}ms)")
