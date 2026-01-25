"""
Utility functions for Adaptive Queuing Priority Scheduler (AQPS).

This module provides helper functions for laxity calculation, energy conversions,
quantization, and other common operations used throughout the scheduler.

Phase 4 additions:
- J1772 pilot signal quantization helpers
- DataFrame export utilities
"""

from typing import Dict, List, Optional, Any
from .data_structures import SessionInfo


# Standard J1772 pilot signals (Amps)
J1772_PILOT_SIGNALS = [0.0, 8.0, 16.0, 24.0, 32.0]


def calculate_laxity(
    session: SessionInfo,
    current_time: int,
    voltage: float = 220.0,
    period_minutes: float = 5.0
) -> float:
    """
    Calculate the laxity (time slack) for a charging session.
    
    Laxity represents how much "extra time" is available beyond the minimum
    time needed to complete charging. Lower laxity means less flexibility.
    
    Formula:
        laxity = (departure_time - current_time) - (remaining_demand / max_rate_kW)
    
    Where max_rate_kW is converted from max_rate (Amps) using:
        max_rate_kW = (max_rate * voltage) / 1000
    
    Args:
        session: The charging session
        current_time: Current simulation time (period)
        voltage: Network voltage in Volts
        period_minutes: Length of each period in minutes
    
    Returns:
        Laxity in periods (can be negative if insufficient time)
    
    Examples:
        >>> session = SessionInfo(
        ...     session_id="EV1", station_id="S1",
        ...     arrival_time=0, departure_time=100,
        ...     energy_requested=20.0, max_rate=32.0,
        ...     current_charge=5.0
        ... )
        >>> laxity = calculate_laxity(session, current_time=10, voltage=220)
        >>> # laxity ≈ 90 - (15 kWh / 7.04 kW) ≈ 90 - 2.13 ≈ 87.87 periods
    """
    # Calculate remaining time in periods
    remaining_time = session.departure_time - current_time
    
    # Calculate remaining energy demand
    remaining_demand = session.remaining_demand  # kWh
    
    # Convert max rate from Amps to kW
    max_rate_kw = (session.max_rate * voltage) / 1000.0
    
    # Handle edge cases
    if max_rate_kw <= 0:
        return float('-inf')  # Cannot charge, infinite urgency
    
    if remaining_demand <= 0:
        return float('inf')  # Already charged, no urgency
    
    # Calculate minimum charging time in periods
    # Each period delivers: max_rate_kw * (period_minutes / 60) kWh
    energy_per_period = max_rate_kw * (period_minutes / 60.0)
    min_charging_periods = remaining_demand / energy_per_period
    
    # Laxity is the slack time
    laxity = remaining_time - min_charging_periods
    
    return laxity


def calculate_remaining_energy(
    session: SessionInfo
) -> float:
    """
    Calculate remaining energy demand for a session.
    
    Args:
        session: The charging session
    
    Returns:
        Remaining energy in kWh (non-negative)
    """
    return max(0.0, session.energy_requested - session.current_charge)


def energy_deliverable_at_rate(
    rate_amps: float,
    periods: int,
    voltage: float = 220.0,
    period_minutes: float = 5.0
) -> float:
    """
    Calculate energy that can be delivered at a given rate over time.
    
    Args:
        rate_amps: Charging rate in Amps
        periods: Number of periods to charge
        voltage: Network voltage in Volts
        period_minutes: Length of each period in minutes
    
    Returns:
        Energy in kWh
    
    Examples:
        >>> energy = energy_deliverable_at_rate(32.0, 12, 220, 5)
        >>> # 32A * 220V = 7.04 kW
        >>> # 12 periods * 5 min = 60 min = 1 hour
        >>> # Energy = 7.04 kWh
    """
    power_kw = (rate_amps * voltage) / 1000.0
    hours = (periods * period_minutes) / 60.0
    return power_kw * hours


def rate_needed_for_energy(
    energy_kwh: float,
    periods: int,
    voltage: float = 220.0,
    period_minutes: float = 5.0
) -> float:
    """
    Calculate the charging rate needed to deliver energy in given time.
    
    Args:
        energy_kwh: Energy to deliver in kWh
        periods: Number of periods available
        voltage: Network voltage in Volts
        period_minutes: Length of each period in minutes
    
    Returns:
        Required rate in Amps
    
    Examples:
        >>> rate = rate_needed_for_energy(10.0, 24, 220, 5)
        >>> # 24 periods * 5 min = 2 hours
        >>> # Power needed = 10 / 2 = 5 kW
        >>> # Rate = 5000 / 220 ≈ 22.73 Amps
    """
    if periods <= 0:
        return float('inf')
    
    hours = (periods * period_minutes) / 60.0
    if hours <= 0:
        return float('inf')
    
    power_kw = energy_kwh / hours
    rate_amps = (power_kw * 1000.0) / voltage
    return rate_amps


def validate_sessions(sessions: List[SessionInfo]) -> List[str]:
    """
    Validate a list of sessions and return any error messages.
    
    Args:
        sessions: List of sessions to validate
    
    Returns:
        List of error messages (empty if all valid)
    """
    errors = []
    
    for session in sessions:
        # Check for unique session IDs
        if not session.session_id:
            errors.append(f"Session has empty session_id")
        
        # Check timing
        if session.departure_time <= session.arrival_time:
            errors.append(
                f"Session {session.session_id}: departure_time must be > arrival_time"
            )
        
        # Check energy
        if session.energy_requested < 0:
            errors.append(
                f"Session {session.session_id}: energy_requested must be non-negative"
            )
        
        if session.current_charge < 0:
            errors.append(
                f"Session {session.session_id}: current_charge must be non-negative"
            )
        
        # Check rates
        if session.max_rate <= 0:
            errors.append(
                f"Session {session.session_id}: max_rate must be positive"
            )
        
        if session.min_rate < 0:
            errors.append(
                f"Session {session.session_id}: min_rate must be non-negative"
            )
        
        if session.min_rate > session.max_rate:
            errors.append(
                f"Session {session.session_id}: min_rate cannot exceed max_rate"
            )
    
    return errors


def format_schedule(
    schedule_array: Dict[str, float],
    round_digits: int = 2
) -> Dict[str, float]:
    """
    Format a schedule dictionary by rounding values.
    
    Args:
        schedule_array: Dictionary mapping station_id to charging rate
        round_digits: Number of decimal places to round to
    
    Returns:
        Formatted schedule dictionary
    """
    return {
        station_id: round(rate, round_digits)
        for station_id, rate in schedule_array.items()
    }


def calculate_total_energy_delivered(
    schedule: Dict[str, float],
    voltage: float = 220.0,
    period_minutes: float = 5.0
) -> float:
    """
    Calculate total energy that will be delivered by a schedule.
    
    Args:
        schedule: Dictionary mapping station_id to charging rate (Amps)
        voltage: Network voltage in Volts
        period_minutes: Length of period in minutes
    
    Returns:
        Total energy in kWh
    """
    total_power_kw = sum(
        (rate * voltage) / 1000.0
        for rate in schedule.values()
    )
    hours = period_minutes / 60.0
    return total_power_kw * hours


# =============================================================================
# Phase 4: Quantization Utilities
# =============================================================================

def quantize_rate_floor(
    rate: float,
    pilot_signals: List[float] = None
) -> float:
    """
    Quantize a rate to the nearest lower valid pilot signal (floor).
    
    Args:
        rate: Continuous charging rate (Amps)
        pilot_signals: List of valid pilot signals (uses J1772 if None)
    
    Returns:
        Floor-quantized rate (Amps)
    
    Examples:
        >>> quantize_rate_floor(25.5)
        24.0
        >>> quantize_rate_floor(7.5)
        0.0
        >>> quantize_rate_floor(32.0)
        32.0
    """
    if pilot_signals is None:
        pilot_signals = J1772_PILOT_SIGNALS
    
    if rate <= 0:
        return 0.0
    
    valid = [s for s in pilot_signals if s <= rate]
    return max(valid) if valid else 0.0


def quantize_rate_ceil(
    rate: float,
    pilot_signals: List[float] = None
) -> float:
    """
    Quantize a rate to the nearest higher valid pilot signal (ceiling).
    
    Args:
        rate: Continuous charging rate (Amps)
        pilot_signals: List of valid pilot signals (uses J1772 if None)
    
    Returns:
        Ceiling-quantized rate (Amps)
    
    Examples:
        >>> quantize_rate_ceil(25.5)
        32.0
        >>> quantize_rate_ceil(7.5)
        8.0
        >>> quantize_rate_ceil(32.0)
        32.0
    """
    if pilot_signals is None:
        pilot_signals = J1772_PILOT_SIGNALS
    
    if rate <= 0:
        return 0.0
    
    valid = [s for s in pilot_signals if s >= rate]
    return min(valid) if valid else max(pilot_signals)


def quantize_schedule_floor(
    schedule: Dict[str, float],
    pilot_signals: List[float] = None
) -> Dict[str, float]:
    """
    Quantize all rates in a schedule using floor quantization.
    
    Args:
        schedule: Dictionary mapping station_id to rate (Amps)
        pilot_signals: List of valid pilot signals (uses J1772 if None)
    
    Returns:
        Schedule with floor-quantized rates
    """
    return {
        station_id: quantize_rate_floor(rate, pilot_signals)
        for station_id, rate in schedule.items()
    }


def calculate_quantization_loss(
    pre_schedule: Dict[str, float],
    post_schedule: Dict[str, float]
) -> Dict[str, Any]:
    """
    Calculate the capacity loss from quantization.
    
    Args:
        pre_schedule: Schedule before quantization
        post_schedule: Schedule after quantization
    
    Returns:
        Dictionary with quantization statistics
    """
    pre_total = sum(pre_schedule.values())
    post_total = sum(post_schedule.values())
    
    per_station_loss = {}
    for station_id in pre_schedule:
        pre_rate = pre_schedule.get(station_id, 0.0)
        post_rate = post_schedule.get(station_id, 0.0)
        per_station_loss[station_id] = pre_rate - post_rate
    
    return {
        'pre_quantization_total': pre_total,
        'post_quantization_total': post_total,
        'total_loss': pre_total - post_total,
        'efficiency_pct': (post_total / pre_total * 100.0) if pre_total > 0 else 100.0,
        'per_station_loss': per_station_loss,
    }


# =============================================================================
# Phase 4: DataFrame Export Utilities
# =============================================================================

def metrics_to_dataframe_data(metrics_list: List[Any]) -> List[Dict]:
    """
    Convert a list of metrics objects to DataFrame-ready dictionaries.
    
    Args:
        metrics_list: List of metrics objects with to_dict() method
    
    Returns:
        List of dictionaries suitable for pandas DataFrame creation
    """
    result = []
    for m in metrics_list:
        if hasattr(m, 'to_dict'):
            result.append(m.to_dict())
        elif hasattr(m, '__dict__'):
            result.append(vars(m).copy())
        else:
            result.append({'value': m})
    return result


def export_to_csv_string(data: List[Dict], delimiter: str = ',') -> str:
    """
    Export list of dictionaries to CSV string format.
    
    Args:
        data: List of dictionaries with consistent keys
        delimiter: Field delimiter (default: comma)
    
    Returns:
        CSV-formatted string
    """
    if not data:
        return ""
    
    # Get headers from first item
    headers = list(data[0].keys())
    
    lines = [delimiter.join(str(h) for h in headers)]
    
    for row in data:
        values = [str(row.get(h, '')) for h in headers]
        lines.append(delimiter.join(values))
    
    return '\n'.join(lines)


def calculate_summary_statistics(values: List[float]) -> Dict[str, float]:
    """
    Calculate summary statistics for a list of values.
    
    Args:
        values: List of numeric values
    
    Returns:
        Dictionary with min, max, mean, sum statistics
    """
    if not values:
        return {
            'min': 0.0,
            'max': 0.0,
            'mean': 0.0,
            'sum': 0.0,
            'count': 0,
        }
    
    return {
        'min': min(values),
        'max': max(values),
        'mean': sum(values) / len(values),
        'sum': sum(values),
        'count': len(values),
    }
