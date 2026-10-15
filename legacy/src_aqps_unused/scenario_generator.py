"""
Scenario generator for AQPS testing and validation.

This module generates synthetic EV charging scenarios (S1-S6) compatible
with the AQPS scheduler for runtime testing and performance evaluation.
"""

import numpy as np
from typing import List, Tuple, Optional
from .data_structures import SessionInfo, PriorityConfig


class ScenarioGenerator:
    """
    Generate synthetic EV charging scenarios for AQPS testing.
    
    Supports the six scenario types defined in the research:
    - S1: Baseline (27% priority, uniform arrivals)
    - S2: Low Priority (10% priority)
    - S3: High Priority (50% priority)
    - S4: Morning Rush (clustered AM arrivals)
    - S5: Cloudy Day (reduced PV - for future integration)
    - S6: Peak Stress (high priority + PM clustering)
    
    Attributes:
        period_minutes: Length of each scheduling period
        voltage: Network voltage
        max_rate: Maximum charging rate per EV
        min_rate: Minimum charging rate per EV
        priority_config: Configuration for priority selection
    """
    
    SCENARIOS = {
        'S1': {
            'name': 'S1: Baseline',
            'priority_pct': 0.27,
            'arrival_pattern': 'uniform',
            'description': 'Standard operational scenario'
        },
        'S2': {
            'name': 'S2: Low Priority',
            'priority_pct': 0.10,
            'arrival_pattern': 'uniform',
            'description': 'Few urgent vehicles'
        },
        'S3': {
            'name': 'S3: High Priority',
            'priority_pct': 0.50,
            'arrival_pattern': 'uniform',
            'description': 'Many urgent vehicles'
        },
        'S4': {
            'name': 'S4: Morning Rush',
            'priority_pct': 0.27,
            'arrival_pattern': 'clustered_am',
            'description': 'Clustered morning arrivals'
        },
        'S5': {
            'name': 'S5: Cloudy Day',
            'priority_pct': 0.27,
            'arrival_pattern': 'uniform',
            'description': 'Reduced PV generation'
        },
        'S6': {
            'name': 'S6: Peak Stress',
            'priority_pct': 0.50,
            'arrival_pattern': 'clustered_pm',
            'description': 'High demand + PM clustering'
        }
    }
    
    def __init__(
        self,
        period_minutes: float = 5.0,
        voltage: float = 220.0,
        max_rate: float = 32.0,
        min_rate: float = 6.0,
        priority_config: Optional[PriorityConfig] = None
    ):
        """
        Initialize the scenario generator.
        
        Args:
            period_minutes: Length of each period in minutes
            voltage: Network voltage in Volts
            max_rate: Maximum charging rate in Amps
            min_rate: Minimum charging rate in Amps
            priority_config: Priority selection configuration
        """
        self.period_minutes = period_minutes
        self.voltage = voltage
        self.max_rate = max_rate
        self.min_rate = min_rate
        self.priority_config = priority_config or PriorityConfig()
    
    def generate(
        self,
        scenario_key: str,
        n_sessions: int = 140,
        start_hour: float = 6.0,
        end_hour: float = 18.0,
        seed: Optional[int] = None
    ) -> List[SessionInfo]:
        """
        Generate a complete scenario.
        
        Args:
            scenario_key: Scenario identifier (e.g., 'S1', 'S2')
            n_sessions: Number of sessions to generate
            start_hour: Start of arrival window (hour of day)
            end_hour: End of arrival window (hour of day)
            seed: Random seed for reproducibility
        
        Returns:
            List of SessionInfo objects with is_priority flags set
        
        Examples:
            >>> generator = ScenarioGenerator()
            >>> sessions = generator.generate('S1', n_sessions=100, seed=42)
            >>> priority_count = sum(s.is_priority for s in sessions)
        """
        if scenario_key not in self.SCENARIOS:
            raise ValueError(
                f"Unknown scenario: {scenario_key}. "
                f"Valid options: {list(self.SCENARIOS.keys())}"
            )
        
        if seed is not None:
            np.random.seed(seed)
        
        scenario = self.SCENARIOS[scenario_key]
        
        # Generate base sessions
        sessions = self._generate_base_sessions(
            n_sessions,
            start_hour,
            end_hour,
            scenario['arrival_pattern']
        )
        
        # Select priority sessions
        priority_ids = self._select_priority_sessions(
            sessions,
            scenario['priority_pct']
        )
        
        # Mark priority sessions
        for session in sessions:
            session.is_priority = session.session_id in priority_ids
        
        return sessions
    
    def _generate_base_sessions(
        self,
        n_sessions: int,
        start_hour: float,
        end_hour: float,
        arrival_pattern: str
    ) -> List[SessionInfo]:
        """Generate base sessions without priority assignment."""
        sessions = []
        
        for i in range(n_sessions):
            session_id = f"EV_{i:03d}"
            station_id = f"EVSE_{i % 54:02d}"  # 54 EVSEs as in your setup
            
            # Generate arrival time
            arrival_hour = self._sample_arrival_hour(
                arrival_pattern,
                start_hour,
                end_hour
            )
            arrival_period = int(arrival_hour * 60 / self.period_minutes)
            
            # Generate parking duration (2-8 hours)
            duration_hours = np.random.uniform(2.0, 8.0)
            duration_periods = int(duration_hours * 60 / self.period_minutes)
            departure_period = arrival_period + duration_periods
            
            # Generate energy request (5-40 kWh)
            energy_requested = np.random.uniform(5.0, 40.0)
            
            # Create session
            session = SessionInfo(
                session_id=session_id,
                station_id=station_id,
                arrival_time=arrival_period,
                departure_time=departure_period,
                energy_requested=round(energy_requested, 2),
                max_rate=self.max_rate,
                min_rate=self.min_rate,
                current_charge=0.0,
                is_priority=False  # Will be set later
            )
            
            sessions.append(session)
        
        return sessions
    
    def _sample_arrival_hour(
        self,
        pattern: str,
        start_hour: float,
        end_hour: float
    ) -> float:
        """Sample arrival time based on pattern."""
        if pattern == 'uniform':
            return np.random.uniform(start_hour, end_hour)
        
        elif pattern == 'clustered_am':
            # 70% arrive 6-9 AM
            if np.random.random() < 0.7:
                return np.random.uniform(6.0, 9.0)
            else:
                return np.random.uniform(9.0, end_hour)
        
        elif pattern == 'clustered_pm':
            # 70% arrive 2-6 PM
            if np.random.random() < 0.7:
                return np.random.uniform(14.0, 18.0)
            else:
                return np.random.uniform(start_hour, 14.0)
        
        elif pattern == 'bimodal':
            # Morning and afternoon peaks
            if np.random.random() < 0.5:
                return np.clip(np.random.normal(8.0, 1.0), start_hour, end_hour)
            else:
                return np.clip(np.random.normal(16.0, 1.0), start_hour, end_hour)
        
        else:
            return np.random.uniform(start_hour, end_hour)
    
    def _select_priority_sessions(
        self,
        sessions: List[SessionInfo],
        target_priority_pct: float
    ) -> set:
        """
        Select priority sessions based on suitability scoring.
        
        Args:
            sessions: List of sessions to select from
            target_priority_pct: Target percentage of priority sessions
        
        Returns:
            Set of session IDs to mark as priority
        """
        n_target = int(len(sessions) * target_priority_pct)
        
        # Score each session
        scored = []
        for session in sessions:
            score = self._calculate_priority_score(session)
            scored.append((session.session_id, score))
        
        # Sort by score and select top N
        scored.sort(key=lambda x: x[1], reverse=True)
        priority_ids = {sid for sid, _ in scored[:n_target]}
        
        return priority_ids
    
    def _calculate_priority_score(self, session: SessionInfo) -> float:
        """Calculate priority suitability score."""
        config = self.priority_config
        score = 0.0
        
        # Prefer energy in sweet spot (10-30 kWh)
        if config.min_energy_kwh <= session.energy_requested <= config.max_energy_kwh:
            score += 10.0
        elif session.energy_requested < config.min_energy_kwh:
            score += 5.0
        else:
            score += 2.0
        
        # Prefer longer durations
        duration_hours = (session.departure_time - session.arrival_time) * self.period_minutes / 60.0
        if duration_hours >= config.min_duration_hours:
            score += 5.0
        if duration_hours >= config.high_energy_min_duration:
            score += 3.0
        
        # Add randomness
        score += np.random.uniform(0, 2)
        
        return score
    
    @classmethod
    def list_scenarios(cls) -> List[str]:
        """List available scenario keys."""
        return list(cls.SCENARIOS.keys())
    
    @classmethod
    def get_scenario_info(cls, scenario_key: str) -> dict:
        """Get information about a scenario."""
        if scenario_key not in cls.SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario_key}")
        return cls.SCENARIOS[scenario_key].copy()


def generate_scenario(
    scenario_key: str,
    n_sessions: int = 140,
    seed: Optional[int] = None
) -> List[SessionInfo]:
    """
    Convenience function to generate a scenario.
    
    Args:
        scenario_key: Scenario identifier (S1-S6)
        n_sessions: Number of sessions to generate
        seed: Random seed
    
    Returns:
        List of SessionInfo objects
    
    Examples:
        >>> from aqps.scenario_generator import generate_scenario
        >>> sessions = generate_scenario('S1', n_sessions=100, seed=42)
    """
    generator = ScenarioGenerator()
    return generator.generate(scenario_key, n_sessions, seed=seed)
