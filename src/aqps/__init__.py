"""
Adaptive Queuing Priority Scheduler (AQPS)

A computationally efficient, real-time deployable scheduler for priority-aware
EV fleet charging. AQPS provides an O(n) heuristic alternative to 
MPC-based approaches while maintaining priority guarantees and cost optimization.

Main Components:
- AdaptiveQueuingPriorityScheduler: Main scheduling algorithm
- ScenarioGenerator: Generate S1-S6 test scenarios
- SessionInfo: EV session data structure
- QueueManager: Two-tier queue management
- AQPSConfig: Configuration parameters

Phase 2 Components:
- PreemptionManager: Handles preemption logic (Option B + A)
- ThresholdTracker: Tracks threshold violations for analysis
- PreemptionEvent: Records preemption events
- ThresholdViolationEvent: Records threshold violations

Phase 3 Components:
- ThreePhaseNetwork: Three-phase infrastructure with per-phase limits
- TOUTariff: Time-of-Use tariff with configurable rates
- TOUOptimizer: Aggressive deferral logic for non-priority EVs
- RenewableIntegration: PV and BESS forecast integration
- DeferralTracker: Prevents deferral window congestion

Phase 4 Components:
- Phase4Config: Quantization and benchmarking configuration
- QuantizationMetrics: Per-timestep quantization statistics
- QuantizationEvent: Individual quantization adjustments
- ComputationalMetrics: Timing and performance data
- SimulationSummary: Aggregate simulation statistics

Quick Start:
    >>> from aqps import AdaptiveQueuingPriorityScheduler, AQPSConfig, generate_scenario
    >>> 
    >>> config = AQPSConfig(min_priority_rate=16.0, total_capacity=600.0, voltage=415.0)
    >>> scheduler = AdaptiveQueuingPriorityScheduler(config)
    >>> 
    >>> # Phase 3: Configure TOU (insert your tariff details)
    >>> scheduler.configure_tou(
    ...     peak_price=0.40,
    ...     off_peak_price=0.15,
    ...     peak_hours=[(14, 20)]
    ... )
    >>> 
    >>> # Phase 3: Configure three-phase network
    >>> scheduler.configure_network(
    ...     phase_a_limit=200.0,
    ...     phase_b_limit=200.0,
    ...     phase_c_limit=200.0
    ... )
    >>> 
    >>> sessions = generate_scenario('S1', n_sessions=100, seed=42)
    >>> schedule = scheduler.schedule(sessions, current_time=0)
    >>> 
    >>> # Get analysis data
    >>> tou_stats = scheduler.get_tou_statistics()
    >>> network_status = scheduler.get_network_status()
    >>> quant_stats = scheduler.get_quantization_statistics()  # Phase 4
    >>> comp_stats = scheduler.get_computational_statistics()  # Phase 4
    >>> summary = scheduler.get_simulation_summary()  # Phase 4

Author: Research Team
Version: 0.4.0 (Phase 4 - Quantization & Polish)
"""

from .scheduler import AdaptiveQueuingPriorityScheduler
from .data_structures import (
    SessionInfo,
    PriorityQueueEntry,
    PriorityConfig,
    AQPSConfig,
    SchedulingMetrics,
    # Phase 2 data structures
    PreemptionMethod,
    PreemptionEvent,
    ThresholdViolationEvent,
    # Phase 3 data structures
    TOUDeferralEvent,
    Phase3Config,
    TOUMetrics,
    # Phase 4 data structures
    Phase4Config,
    QuantizationMetrics,
    QuantizationEvent,
    ComputationalMetrics,
    SimulationSummary,
    J1772_PILOT_SIGNALS,
)
from .queue_manager import QueueManager
from .scenario_generator import ScenarioGenerator, generate_scenario
from .utils import (
    calculate_laxity,
    calculate_remaining_energy,
    energy_deliverable_at_rate,
    rate_needed_for_energy,
    # Phase 4 utilities
    quantize_rate_floor,
    quantize_rate_ceil,
    quantize_schedule_floor,
    calculate_quantization_loss,
    metrics_to_dataframe_data,
    export_to_csv_string,
    calculate_summary_statistics,
)
# Phase 2 modules
from .preemption import PreemptionManager
from .threshold_tracker import ThresholdTracker, TimestepSnapshot

# Phase 3 modules
from .three_phase_network import (
    ThreePhaseNetwork,
    ThreePhaseNetworkConfig,
    EVSESpecification,
    Phase,
    PhaseCapacity,
    create_standard_network
)
from .tou_optimization import (
    TOUTariff,
    TOUTariffConfig,
    TOUOptimizer,
    TOUPeriodType,
    DeferralTracker,
    DeferralDecision,
    create_australian_tou_tariff
)
from .renewable_integration import (
    RenewableIntegration,
    RenewableIntegrationConfig,
    PVSystem,
    PVSystemConfig,
    PVForecastPoint,
    BESSController,
    BESSConfig,
    BESSStatePoint,
    BESSDispatchMode,
    generate_typical_pv_profile,
    generate_typical_bess_profile
)

__version__ = "0.4.0"
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
    
    # Phase 2 data structures
    'PreemptionMethod',
    'PreemptionEvent',
    'ThresholdViolationEvent',
    
    # Phase 3 data structures
    'TOUDeferralEvent',
    'Phase3Config',
    'TOUMetrics',
    
    # Phase 4 data structures
    'Phase4Config',
    'QuantizationMetrics',
    'QuantizationEvent',
    'ComputationalMetrics',
    'SimulationSummary',
    'J1772_PILOT_SIGNALS',
    
    # Queue management
    'QueueManager',
    
    # Phase 2: Preemption
    'PreemptionManager',
    
    # Phase 2: Threshold tracking
    'ThresholdTracker',
    'TimestepSnapshot',
    
    # Phase 3: Three-phase network
    'ThreePhaseNetwork',
    'ThreePhaseNetworkConfig',
    'EVSESpecification',
    'Phase',
    'PhaseCapacity',
    'create_standard_network',
    
    # Phase 3: TOU optimization
    'TOUTariff',
    'TOUTariffConfig',
    'TOUOptimizer',
    'TOUPeriodType',
    'DeferralTracker',
    'DeferralDecision',
    'create_australian_tou_tariff',
    
    # Phase 3: Renewable integration
    'RenewableIntegration',
    'RenewableIntegrationConfig',
    'PVSystem',
    'PVSystemConfig',
    'PVForecastPoint',
    'BESSController',
    'BESSConfig',
    'BESSStatePoint',
    'BESSDispatchMode',
    'generate_typical_pv_profile',
    'generate_typical_bess_profile',
    
    # Scenario generation
    'ScenarioGenerator',
    'generate_scenario',
    
    # Utilities
    'calculate_laxity',
    'calculate_remaining_energy',
    'energy_deliverable_at_rate',
    'rate_needed_for_energy',
    
    # Phase 4 utilities
    'quantize_rate_floor',
    'quantize_rate_ceil',
    'quantize_schedule_floor',
    'calculate_quantization_loss',
    'metrics_to_dataframe_data',
    'export_to_csv_string',
    'calculate_summary_statistics',
]
