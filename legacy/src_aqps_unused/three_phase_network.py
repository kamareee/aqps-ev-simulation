"""
Three-Phase Network Infrastructure for AQPS.

This module implements a three-phase electrical network topology for EV charging
infrastructure. It uses a balanced static configuration (18/18/18 EVSEs per phase)
and enforces per-phase transformer capacity limits.

Based on ACN-Sim patterns and AeroVironment EVSE specifications:
- J1772 discrete pilot signals: [0, 8, 16, 24, 32] Amps
- 415V three-phase voltage
- 54 total EVSEs across 3 phases

Author: Research Team
Phase: 3 (TOU Optimization with Infrastructure)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class Phase(Enum):
    """Electrical phase identifiers."""
    A = "A"
    B = "B"
    C = "C"


@dataclass
class EVSESpecification:
    """
    EVSE hardware specification based on AeroVironment J1772 standard.
    
    Attributes:
        evse_id: Unique identifier for the EVSE
        phase: Electrical phase assignment (A, B, or C)
        max_rate: Maximum charging rate (Amps)
        min_rate: Minimum non-zero charging rate (Amps)
        pilot_signals: Available discrete pilot signal levels (Amps)
    """
    evse_id: str
    phase: Phase
    max_rate: float = 32.0
    min_rate: float = 8.0  # Minimum non-zero rate per J1772
    pilot_signals: List[float] = field(default_factory=lambda: [0.0, 8.0, 16.0, 24.0, 32.0])
    
    def quantize_rate(self, rate: float) -> float:
        """
        Quantize a continuous rate to the nearest valid pilot signal.
        
        Args:
            rate: Continuous charging rate (Amps)
        
        Returns:
            Nearest valid pilot signal (Amps)
        """
        if rate <= 0:
            return 0.0
        
        # Find the closest pilot signal
        valid_signals = [s for s in self.pilot_signals if s <= rate]
        if not valid_signals:
            return 0.0
        return max(valid_signals)
    
    def __repr__(self) -> str:
        return f"EVSE({self.evse_id}, phase={self.phase.value}, max={self.max_rate}A)"


@dataclass
class PhaseCapacity:
    """
    Per-phase capacity tracking.
    
    Attributes:
        phase: Phase identifier
        limit_amps: Maximum capacity for this phase (Amps)
        allocated_amps: Currently allocated capacity (Amps)
        evse_ids: List of EVSE IDs connected to this phase
    """
    phase: Phase
    limit_amps: float
    allocated_amps: float = 0.0
    evse_ids: List[str] = field(default_factory=list)
    
    @property
    def available_amps(self) -> float:
        """Calculate available capacity on this phase."""
        return max(0.0, self.limit_amps - self.allocated_amps)
    
    @property
    def utilization(self) -> float:
        """Calculate utilization percentage."""
        if self.limit_amps <= 0:
            return 100.0
        return (self.allocated_amps / self.limit_amps) * 100.0
    
    def can_allocate(self, amps: float) -> bool:
        """Check if allocation is feasible within phase limit."""
        return self.allocated_amps + amps <= self.limit_amps
    
    def allocate(self, amps: float) -> bool:
        """
        Attempt to allocate capacity on this phase.
        
        Returns:
            True if allocation succeeded, False if would exceed limit
        """
        if not self.can_allocate(amps):
            return False
        self.allocated_amps += amps
        return True
    
    def deallocate(self, amps: float) -> None:
        """Remove allocated capacity from this phase."""
        self.allocated_amps = max(0.0, self.allocated_amps - amps)
    
    def reset(self) -> None:
        """Reset allocated capacity to zero."""
        self.allocated_amps = 0.0
    
    def __repr__(self) -> str:
        return (f"PhaseCapacity({self.phase.value}: "
                f"{self.allocated_amps:.1f}/{self.limit_amps:.1f}A, "
                f"{self.utilization:.1f}%)")


@dataclass
class ThreePhaseNetworkConfig:
    """
    Configuration for three-phase network topology.
    
    Attributes:
        total_evses: Total number of EVSEs (default: 54)
        evses_per_phase: Number of EVSEs per phase (default: 18 for balanced)
        phase_a_limit: Transformer limit for phase A (Amps)
        phase_b_limit: Transformer limit for phase B (Amps)
        phase_c_limit: Transformer limit for phase C (Amps)
        voltage: System voltage (Volts) - 415V for three-phase
        evse_max_rate: Maximum rate per EVSE (Amps)
        evse_min_rate: Minimum non-zero rate per EVSE (Amps)
        pilot_signals: Available J1772 pilot signals (Amps)
    """
    total_evses: int = 54
    evses_per_phase: int = 18
    phase_a_limit: float = 200.0  # Amps - configurable per your transformer
    phase_b_limit: float = 200.0  # Amps
    phase_c_limit: float = 200.0  # Amps
    voltage: float = 415.0  # Three-phase voltage
    evse_max_rate: float = 32.0
    evse_min_rate: float = 8.0
    pilot_signals: List[float] = field(default_factory=lambda: [0.0, 8.0, 16.0, 24.0, 32.0])
    
    @property
    def total_capacity(self) -> float:
        """Total system capacity across all phases."""
        return self.phase_a_limit + self.phase_b_limit + self.phase_c_limit
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.total_evses != self.evses_per_phase * 3:
            raise ValueError(
                f"total_evses ({self.total_evses}) must equal "
                f"evses_per_phase * 3 ({self.evses_per_phase * 3})"
            )
        if self.phase_a_limit <= 0 or self.phase_b_limit <= 0 or self.phase_c_limit <= 0:
            raise ValueError("Phase limits must be positive")
        if self.voltage <= 0:
            raise ValueError("Voltage must be positive")


class ThreePhaseNetwork:
    """
    Three-phase electrical network for EV charging infrastructure.
    
    This class manages:
    - Balanced static EVSE-to-phase assignment (18/18/18)
    - Per-phase transformer capacity limits
    - Feasibility checking for charging schedules
    - J1772 pilot signal quantization
    
    The network uses a balanced static layout where:
    - EVSEs 1-18 → Phase A
    - EVSEs 19-36 → Phase B
    - EVSEs 37-54 → Phase C
    
    Attributes:
        config: Network configuration
        evses: Dictionary of EVSE specifications by ID
        phases: Dictionary of phase capacity trackers
        station_to_evse: Mapping from station_id to EVSE specification
    
    Examples:
        >>> config = ThreePhaseNetworkConfig(
        ...     phase_a_limit=150.0,
        ...     phase_b_limit=150.0,
        ...     phase_c_limit=150.0
        ... )
        >>> network = ThreePhaseNetwork(config)
        >>> feasible, violations = network.check_schedule_feasibility(schedule)
    """
    
    def __init__(self, config: Optional[ThreePhaseNetworkConfig] = None):
        """
        Initialize the three-phase network.
        
        Args:
            config: Network configuration (uses defaults if None)
        """
        self.config = config or ThreePhaseNetworkConfig()
        self.config.validate()
        
        # Initialize EVSE specifications
        self.evses: Dict[str, EVSESpecification] = {}
        self._create_evses()
        
        # Initialize phase capacity trackers
        self.phases: Dict[Phase, PhaseCapacity] = {
            Phase.A: PhaseCapacity(
                phase=Phase.A,
                limit_amps=self.config.phase_a_limit,
                evse_ids=[eid for eid, evse in self.evses.items() if evse.phase == Phase.A]
            ),
            Phase.B: PhaseCapacity(
                phase=Phase.B,
                limit_amps=self.config.phase_b_limit,
                evse_ids=[eid for eid, evse in self.evses.items() if evse.phase == Phase.B]
            ),
            Phase.C: PhaseCapacity(
                phase=Phase.C,
                limit_amps=self.config.phase_c_limit,
                evse_ids=[eid for eid, evse in self.evses.items() if evse.phase == Phase.C]
            ),
        }
        
        # Station ID to EVSE mapping (for scheduler integration)
        self.station_to_evse: Dict[str, EVSESpecification] = {}
        
        logger.info(
            f"ThreePhaseNetwork initialized: {self.config.total_evses} EVSEs, "
            f"capacity={self.config.total_capacity:.0f}A "
            f"({self.config.phase_a_limit}/{self.config.phase_b_limit}/"
            f"{self.config.phase_c_limit}A per phase)"
        )
    
    def _create_evses(self) -> None:
        """Create EVSE specifications with balanced phase assignment."""
        phases = [Phase.A, Phase.B, Phase.C]
        
        for i in range(self.config.total_evses):
            evse_id = f"EVSE_{i + 1:02d}"
            
            # Balanced static assignment: 18/18/18
            phase_idx = i // self.config.evses_per_phase
            phase = phases[phase_idx]
            
            self.evses[evse_id] = EVSESpecification(
                evse_id=evse_id,
                phase=phase,
                max_rate=self.config.evse_max_rate,
                min_rate=self.config.evse_min_rate,
                pilot_signals=self.config.pilot_signals.copy()
            )
    
    def register_station(self, station_id: str, evse_id: str) -> bool:
        """
        Register a station ID to an EVSE.
        
        Args:
            station_id: Station identifier (from SessionInfo)
            evse_id: EVSE identifier
        
        Returns:
            True if registration succeeded
        """
        if evse_id not in self.evses:
            logger.warning(f"Unknown EVSE ID: {evse_id}")
            return False
        
        self.station_to_evse[station_id] = self.evses[evse_id]
        return True
    
    def auto_register_stations(self, station_ids: List[str]) -> None:
        """
        Automatically register station IDs to EVSEs in order.
        
        This is a convenience method for simulation setup where station IDs
        are mapped to EVSEs sequentially.
        
        Args:
            station_ids: List of station IDs to register
        """
        evse_list = sorted(self.evses.keys())
        
        for i, station_id in enumerate(station_ids):
            if i < len(evse_list):
                self.station_to_evse[station_id] = self.evses[evse_list[i]]
            else:
                logger.warning(f"No EVSE available for station {station_id}")
    
    def get_station_phase(self, station_id: str) -> Optional[Phase]:
        """Get the phase for a given station ID."""
        evse = self.station_to_evse.get(station_id)
        if evse:
            return evse.phase
        return None
    
    def get_evse_for_station(self, station_id: str) -> Optional[EVSESpecification]:
        """Get the EVSE specification for a station."""
        return self.station_to_evse.get(station_id)
    
    def reset_allocations(self) -> None:
        """Reset all phase allocations to zero."""
        for phase in self.phases.values():
            phase.reset()
    
    def check_schedule_feasibility(
        self,
        schedule: Dict[str, float]
    ) -> Tuple[bool, List[str]]:
        """
        Check if a schedule is feasible within phase constraints.
        
        Args:
            schedule: Dictionary mapping station_id → charging rate (Amps)
        
        Returns:
            Tuple of (is_feasible, list_of_violations)
        """
        # Reset allocations for fresh check
        self.reset_allocations()
        
        violations = []
        
        # Calculate per-phase totals
        phase_totals: Dict[Phase, float] = {Phase.A: 0.0, Phase.B: 0.0, Phase.C: 0.0}
        
        for station_id, rate in schedule.items():
            if rate <= 0:
                continue
            
            phase = self.get_station_phase(station_id)
            if phase is None:
                # Station not registered - try to find by EVSE ID pattern
                evse_match = self._match_station_to_evse(station_id)
                if evse_match:
                    phase = evse_match.phase
                else:
                    violations.append(f"Unknown station: {station_id}")
                    continue
            
            phase_totals[phase] += rate
        
        # Check against limits
        is_feasible = True
        
        for phase, total in phase_totals.items():
            limit = self.phases[phase].limit_amps
            if total > limit:
                is_feasible = False
                violations.append(
                    f"Phase {phase.value} exceeded: {total:.1f}A > {limit:.1f}A"
                )
            else:
                self.phases[phase].allocated_amps = total
        
        return is_feasible, violations
    
    def _match_station_to_evse(self, station_id: str) -> Optional[EVSESpecification]:
        """Try to match a station ID to an EVSE by pattern."""
        # Try direct match (e.g., "EVSE_01")
        if station_id in self.evses:
            return self.evses[station_id]
        
        # Try numeric extraction (e.g., "S1" → EVSE_01)
        try:
            # Extract number from station_id
            num_str = ''.join(filter(str.isdigit, station_id))
            if num_str:
                num = int(num_str)
                evse_id = f"EVSE_{num:02d}"
                if evse_id in self.evses:
                    return self.evses[evse_id]
        except ValueError:
            pass
        
        return None
    
    def get_max_feasible_rate(
        self,
        station_id: str,
        desired_rate: float,
        current_schedule: Dict[str, float]
    ) -> float:
        """
        Calculate the maximum feasible rate for a station given phase constraints.
        
        Args:
            station_id: Station to allocate to
            desired_rate: Desired charging rate (Amps)
            current_schedule: Current schedule (for calculating phase usage)
        
        Returns:
            Maximum feasible rate (may be less than desired if phase constrained)
        """
        phase = self.get_station_phase(station_id)
        if phase is None:
            evse = self._match_station_to_evse(station_id)
            if evse:
                phase = evse.phase
            else:
                return 0.0  # Unknown station
        
        # Calculate current phase usage (excluding this station)
        phase_usage = sum(
            rate for sid, rate in current_schedule.items()
            if sid != station_id and self.get_station_phase(sid) == phase
        )
        
        # Available capacity
        limit = self.phases[phase].limit_amps
        available = limit - phase_usage
        
        # Clamp to available
        return max(0.0, min(desired_rate, available))
    
    def quantize_schedule(
        self,
        schedule: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Quantize all rates in a schedule to valid J1772 pilot signals.
        
        Args:
            schedule: Dictionary mapping station_id → rate (Amps)
        
        Returns:
            Schedule with rates quantized to valid pilot signals
        """
        quantized = {}
        
        for station_id, rate in schedule.items():
            evse = self.station_to_evse.get(station_id)
            if evse:
                quantized[station_id] = evse.quantize_rate(rate)
            else:
                # Use default quantization
                default_signals = self.config.pilot_signals
                valid = [s for s in default_signals if s <= rate]
                quantized[station_id] = max(valid) if valid else 0.0
        
        return quantized
    
    def get_phase_summary(self) -> Dict[str, Dict]:
        """
        Get summary of phase utilization.
        
        Returns:
            Dictionary with phase statistics
        """
        return {
            phase.value: {
                'limit_amps': cap.limit_amps,
                'allocated_amps': cap.allocated_amps,
                'available_amps': cap.available_amps,
                'utilization_pct': cap.utilization,
                'evse_count': len(cap.evse_ids),
            }
            for phase, cap in self.phases.items()
        }
    
    def get_total_available_capacity(self) -> float:
        """Get total available capacity across all phases."""
        return sum(phase.available_amps for phase in self.phases.values())
    
    def get_evse_ids_by_phase(self, phase: Phase) -> List[str]:
        """Get list of EVSE IDs connected to a specific phase."""
        return [eid for eid, evse in self.evses.items() if evse.phase == phase]
    
    def __repr__(self) -> str:
        return (
            f"ThreePhaseNetwork("
            f"evses={self.config.total_evses}, "
            f"capacity={self.config.total_capacity:.0f}A)"
        )


# Convenience function for creating a standard network
def create_standard_network(
    phase_limit_amps: float = 200.0,
    evse_count: int = 54
) -> ThreePhaseNetwork:
    """
    Create a standard balanced three-phase network.
    
    Args:
        phase_limit_amps: Per-phase transformer limit (Amps)
        evse_count: Total number of EVSEs (must be divisible by 3)
    
    Returns:
        Configured ThreePhaseNetwork instance
    """
    if evse_count % 3 != 0:
        raise ValueError("EVSE count must be divisible by 3 for balanced network")
    
    config = ThreePhaseNetworkConfig(
        total_evses=evse_count,
        evses_per_phase=evse_count // 3,
        phase_a_limit=phase_limit_amps,
        phase_b_limit=phase_limit_amps,
        phase_c_limit=phase_limit_amps,
    )
    
    return ThreePhaseNetwork(config)
