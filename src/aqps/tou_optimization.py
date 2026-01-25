"""
Time-of-Use (TOU) Optimization for AQPS.

This module implements TOU-aware charging optimization for the AQPS scheduler.
It provides:
- Configurable TOU tariff interface (manual insertion of rates)
- Aggressive deferral logic for non-priority EVs
- Deferral window tracking to prevent congestion
- Integration with renewable energy sources

Key Features:
- Supports Australian-style TOU tariffs (peak/shoulder/off-peak)
- Aggressive deferral policy: defer whenever cheaper slot exists
- Congestion-aware deferral tracking based on EVSE availability
- PV/BESS-aware cost adjustment

Author: Research Team
Phase: 3 (TOU Optimization)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum

logger = logging.getLogger(__name__)


class TOUPeriodType(Enum):
    """TOU period classifications."""
    PEAK = "peak"
    SHOULDER = "shoulder"  # Mid-peak / shoulder
    OFF_PEAK = "off_peak"


@dataclass
class TOUPeriodDefinition:
    """
    Definition of a TOU rate period.
    
    Attributes:
        period_type: Type of TOU period (peak/shoulder/off_peak)
        start_hour: Start hour (0-23)
        end_hour: End hour (0-23, exclusive)
        price_per_kwh: Price in $/kWh
        days: Days this period applies (0=Mon, 6=Sun). None = all days.
    """
    period_type: TOUPeriodType
    start_hour: float
    end_hour: float
    price_per_kwh: float
    days: Optional[List[int]] = None  # None = all days
    
    def covers_hour(self, hour: float, day_of_week: int = 0) -> bool:
        """Check if this period covers the given hour and day."""
        # Check day
        if self.days is not None and day_of_week not in self.days:
            return False
        
        # Check hour
        if self.start_hour < self.end_hour:
            return self.start_hour <= hour < self.end_hour
        else:
            # Wraps around midnight
            return hour >= self.start_hour or hour < self.end_hour
    
    def __repr__(self) -> str:
        return (f"TOUPeriod({self.period_type.value}: "
                f"{self.start_hour:.0f}:00-{self.end_hour:.0f}:00, "
                f"${self.price_per_kwh:.3f}/kWh)")


@dataclass
class TOUTariffConfig:
    """
    Time-of-Use tariff configuration.
    
    This is the main interface for inserting your tariff details.
    You can configure peak, shoulder (optional), and off-peak rates
    along with their time windows.
    
    Attributes:
        name: Tariff name/identifier
        peak_price: Peak rate ($/kWh)
        shoulder_price: Shoulder/mid-peak rate ($/kWh), optional
        off_peak_price: Off-peak rate ($/kWh)
        peak_hours: List of (start_hour, end_hour) tuples for peak periods
        shoulder_hours: List of (start_hour, end_hour) tuples for shoulder
        supply_charge_daily: Daily supply charge ($), optional
        demand_charge_per_kw: Demand charge ($/kW), optional
    
    Examples:
        >>> # Simple two-rate tariff (peak/off-peak only)
        >>> tariff = TOUTariffConfig(
        ...     name="Simple TOU",
        ...     peak_price=0.35,
        ...     off_peak_price=0.12,
        ...     peak_hours=[(14, 20)]  # 2pm-8pm
        ... )
        
        >>> # Three-rate tariff with shoulder
        >>> tariff = TOUTariffConfig(
        ...     name="Full TOU",
        ...     peak_price=0.45,
        ...     shoulder_price=0.25,
        ...     off_peak_price=0.10,
        ...     peak_hours=[(17, 21)],      # 5pm-9pm
        ...     shoulder_hours=[(7, 17), (21, 22)]  # 7am-5pm, 9pm-10pm
        ... )
    """
    name: str = "Default TOU Tariff"
    peak_price: float = 0.35  # $/kWh - YOU INSERT YOUR VALUE
    shoulder_price: Optional[float] = None  # $/kWh - Optional
    off_peak_price: float = 0.12  # $/kWh - YOU INSERT YOUR VALUE
    peak_hours: List[Tuple[float, float]] = field(
        default_factory=lambda: [(14.0, 20.0)]  # Default: 2pm-8pm peak
    )
    shoulder_hours: List[Tuple[float, float]] = field(
        default_factory=list  # No shoulder periods by default
    )
    supply_charge_daily: float = 0.0
    demand_charge_per_kw: float = 0.0
    
    def validate(self) -> List[str]:
        """Validate tariff configuration."""
        errors = []
        
        if self.peak_price <= 0:
            errors.append("Peak price must be positive")
        if self.off_peak_price <= 0:
            errors.append("Off-peak price must be positive")
        if self.off_peak_price >= self.peak_price:
            errors.append("Off-peak price should be less than peak price")
        if self.shoulder_price is not None:
            if not (self.off_peak_price < self.shoulder_price < self.peak_price):
                errors.append("Shoulder price should be between off-peak and peak")
        
        # Validate hour ranges
        for start, end in self.peak_hours:
            if not (0 <= start < 24 and 0 <= end <= 24):
                errors.append(f"Invalid peak hour range: {start}-{end}")
        
        return errors


class TOUTariff:
    """
    Time-of-Use Tariff Manager.
    
    This class manages TOU rate periods and provides price lookups
    for the TOU optimizer. It supports:
    - Two-rate tariffs (peak/off-peak)
    - Three-rate tariffs (peak/shoulder/off-peak)
    - Custom period definitions
    
    Attributes:
        config: Tariff configuration
        periods: List of period definitions
    
    Examples:
        >>> config = TOUTariffConfig(
        ...     peak_price=0.40,
        ...     off_peak_price=0.15,
        ...     peak_hours=[(15, 21)]
        ... )
        >>> tariff = TOUTariff(config)
        >>> period, price = tariff.get_period_at_time(period=180)  # 3pm
    """
    
    def __init__(
        self,
        config: Optional[TOUTariffConfig] = None,
        period_minutes: float = 5.0
    ):
        """
        Initialize TOU tariff.
        
        Args:
            config: Tariff configuration
            period_minutes: Duration of each simulation period (minutes)
        """
        self.config = config or TOUTariffConfig()
        self.period_minutes = period_minutes
        self.periods: List[TOUPeriodDefinition] = []
        
        # Validate and build periods
        errors = self.config.validate()
        if errors:
            logger.warning(f"Tariff validation warnings: {errors}")
        
        self._build_periods()
        
        logger.info(
            f"TOUTariff initialized: {self.config.name}, "
            f"peak=${self.config.peak_price:.3f}, "
            f"off_peak=${self.config.off_peak_price:.3f}"
        )
    
    def _build_periods(self) -> None:
        """Build period definitions from config."""
        self.periods.clear()
        
        # Add peak periods
        for start, end in self.config.peak_hours:
            self.periods.append(TOUPeriodDefinition(
                period_type=TOUPeriodType.PEAK,
                start_hour=start,
                end_hour=end,
                price_per_kwh=self.config.peak_price
            ))
        
        # Add shoulder periods if defined
        if self.config.shoulder_price is not None:
            for start, end in self.config.shoulder_hours:
                self.periods.append(TOUPeriodDefinition(
                    period_type=TOUPeriodType.SHOULDER,
                    start_hour=start,
                    end_hour=end,
                    price_per_kwh=self.config.shoulder_price
                ))
    
    def period_to_hour(self, period: int, start_hour: float = 0.0) -> float:
        """
        Convert simulation period to hour of day.
        
        Args:
            period: Simulation period (timestep)
            start_hour: Hour at which simulation starts
        
        Returns:
            Hour of day (0-24)
        """
        hours_elapsed = (period * self.period_minutes) / 60.0
        hour_of_day = (start_hour + hours_elapsed) % 24.0
        return hour_of_day
    
    def get_period_type_at_time(
        self,
        period: int,
        start_hour: float = 0.0,
        day_of_week: int = 0
    ) -> TOUPeriodType:
        """
        Get TOU period type at a simulation period.
        
        Args:
            period: Simulation period (timestep)
            start_hour: Hour at which simulation starts
            day_of_week: Day of week (0=Mon, 6=Sun)
        
        Returns:
            TOU period type
        """
        hour = self.period_to_hour(period, start_hour)
        
        # Check defined periods
        for period_def in self.periods:
            if period_def.covers_hour(hour, day_of_week):
                return period_def.period_type
        
        # Default to off-peak if not in any defined period
        return TOUPeriodType.OFF_PEAK
    
    def get_price_at_period(
        self,
        period: int,
        start_hour: float = 0.0,
        day_of_week: int = 0
    ) -> float:
        """
        Get electricity price at a simulation period.
        
        Args:
            period: Simulation period
            start_hour: Hour at which simulation starts
            day_of_week: Day of week
        
        Returns:
            Price in $/kWh
        """
        period_type = self.get_period_type_at_time(period, start_hour, day_of_week)
        
        if period_type == TOUPeriodType.PEAK:
            return self.config.peak_price
        elif period_type == TOUPeriodType.SHOULDER:
            return self.config.shoulder_price or self.config.off_peak_price
        else:
            return self.config.off_peak_price
    
    def find_next_cheaper_period(
        self,
        current_period: int,
        max_lookahead: int = 288,  # 24 hours at 5-min periods
        start_hour: float = 0.0
    ) -> Optional[int]:
        """
        Find the next period with a lower price.
        
        Args:
            current_period: Current simulation period
            max_lookahead: Maximum periods to search ahead
            start_hour: Hour at which simulation starts
        
        Returns:
            Period number of next cheaper slot, or None if none found
        """
        current_price = self.get_price_at_period(current_period, start_hour)
        
        for delta in range(1, max_lookahead + 1):
            future_period = current_period + delta
            future_price = self.get_price_at_period(future_period, start_hour)
            
            if future_price < current_price:
                return future_period
        
        return None
    
    def get_cheapest_period_in_range(
        self,
        start_period: int,
        end_period: int,
        start_hour: float = 0.0
    ) -> int:
        """
        Find the cheapest period within a range.
        
        Args:
            start_period: Start of range (inclusive)
            end_period: End of range (exclusive)
            start_hour: Hour at which simulation starts
        
        Returns:
            Period with lowest price in range
        """
        best_period = start_period
        best_price = self.get_price_at_period(start_period, start_hour)
        
        for period in range(start_period + 1, end_period):
            price = self.get_price_at_period(period, start_hour)
            if price < best_price:
                best_price = price
                best_period = period
        
        return best_period
    
    def is_peak(self, period: int, start_hour: float = 0.0) -> bool:
        """Check if period is in peak TOU."""
        return self.get_period_type_at_time(period, start_hour) == TOUPeriodType.PEAK
    
    def is_off_peak(self, period: int, start_hour: float = 0.0) -> bool:
        """Check if period is in off-peak TOU."""
        return self.get_period_type_at_time(period, start_hour) == TOUPeriodType.OFF_PEAK
    
    def get_daily_profile(
        self,
        start_hour: float = 0.0,
        periods_per_day: int = 288
    ) -> List[Dict]:
        """
        Generate a full day's TOU profile.
        
        Returns:
            List of dicts with period, hour, type, and price
        """
        profile = []
        for period in range(periods_per_day):
            hour = self.period_to_hour(period, start_hour)
            period_type = self.get_period_type_at_time(period, start_hour)
            price = self.get_price_at_period(period, start_hour)
            
            profile.append({
                'period': period,
                'hour': hour,
                'period_type': period_type.value,
                'price': price
            })
        
        return profile
    
    def __repr__(self) -> str:
        return (f"TOUTariff({self.config.name}: "
                f"peak=${self.config.peak_price:.2f}, "
                f"off_peak=${self.config.off_peak_price:.2f})")


@dataclass
class DeferralDecision:
    """
    Record of a deferral decision for a session.
    
    Attributes:
        session_id: Session that may be deferred
        current_period: Current simulation period
        target_period: Target period for deferred charging
        current_price: Price at current period ($/kWh)
        target_price: Price at target period ($/kWh)
        savings_per_kwh: Cost savings per kWh
        can_defer: Whether deferral is feasible
        reason: Reason for decision
    """
    session_id: str
    current_period: int
    target_period: Optional[int]
    current_price: float
    target_price: float
    savings_per_kwh: float
    can_defer: bool
    reason: str
    
    def to_dict(self) -> Dict:
        return {
            'session_id': self.session_id,
            'current_period': self.current_period,
            'target_period': self.target_period,
            'current_price': self.current_price,
            'target_price': self.target_price,
            'savings_per_kwh': self.savings_per_kwh,
            'can_defer': self.can_defer,
            'reason': self.reason,
        }


@dataclass
class DeferralWindow:
    """
    Tracks a deferral window and its capacity.
    
    Attributes:
        target_period: Period EVs are deferring to
        max_evs: Maximum EVs that can defer to this window
        deferred_sessions: Set of session IDs deferred to this window
    """
    target_period: int
    max_evs: int
    deferred_sessions: Set[str] = field(default_factory=set)
    
    @property
    def available_slots(self) -> int:
        return max(0, self.max_evs - len(self.deferred_sessions))
    
    @property
    def is_full(self) -> bool:
        return len(self.deferred_sessions) >= self.max_evs
    
    def add_session(self, session_id: str) -> bool:
        """Add a session to this deferral window."""
        if self.is_full:
            return False
        self.deferred_sessions.add(session_id)
        return True
    
    def remove_session(self, session_id: str) -> None:
        """Remove a session from this deferral window."""
        self.deferred_sessions.discard(session_id)


class DeferralTracker:
    """
    Tracks deferral windows and prevents congestion.
    
    This class monitors how many EVs are deferring to each cheap period
    and prevents over-subscription that would cause congestion.
    
    The tracker limits deferrals based on:
    - Available EVSE count
    - Per-phase capacity limits
    - Configurable safety margin
    
    Attributes:
        windows: Dictionary of deferral windows by period
        max_evses: Total available EVSEs
        safety_margin: Factor to reduce max concurrent deferrals
    """
    
    def __init__(
        self,
        max_evses: int = 54,
        safety_margin: float = 0.8  # Allow 80% utilization
    ):
        """
        Initialize deferral tracker.
        
        Args:
            max_evses: Maximum number of EVSEs
            safety_margin: Fraction of EVSEs that can defer to one window
        """
        self.max_evses = max_evses
        self.safety_margin = safety_margin
        self.windows: Dict[int, DeferralWindow] = {}
        
        # Calculate max EVs per window
        self.max_per_window = int(max_evses * safety_margin)
        
        logger.info(
            f"DeferralTracker initialized: max {self.max_per_window} EVs per window"
        )
    
    def get_or_create_window(self, period: int) -> DeferralWindow:
        """Get or create a deferral window for a period."""
        if period not in self.windows:
            self.windows[period] = DeferralWindow(
                target_period=period,
                max_evs=self.max_per_window
            )
        return self.windows[period]
    
    def can_defer_to(self, period: int) -> bool:
        """Check if more EVs can defer to a period."""
        window = self.get_or_create_window(period)
        return not window.is_full
    
    def register_deferral(self, session_id: str, target_period: int) -> bool:
        """
        Register a session's deferral to a target period.
        
        Args:
            session_id: Session being deferred
            target_period: Target period for charging
        
        Returns:
            True if registration succeeded
        """
        window = self.get_or_create_window(target_period)
        success = window.add_session(session_id)
        
        if success:
            logger.debug(
                f"Registered deferral: {session_id} → period {target_period} "
                f"({window.available_slots} slots remaining)"
            )
        
        return success
    
    def unregister_deferral(self, session_id: str, target_period: int) -> None:
        """Remove a session's deferral registration."""
        if target_period in self.windows:
            self.windows[target_period].remove_session(session_id)
    
    def get_available_slots(self, period: int) -> int:
        """Get available deferral slots for a period."""
        window = self.get_or_create_window(period)
        return window.available_slots
    
    def get_deferral_summary(self) -> Dict[int, int]:
        """Get summary of deferrals per period."""
        return {
            period: len(window.deferred_sessions)
            for period, window in self.windows.items()
        }
    
    def clear_past_windows(self, current_period: int) -> None:
        """Clear deferral windows for past periods."""
        past_periods = [p for p in self.windows.keys() if p < current_period]
        for period in past_periods:
            del self.windows[period]
    
    def reset(self) -> None:
        """Reset all deferral tracking."""
        self.windows.clear()


class TOUOptimizer:
    """
    TOU-aware charging optimizer for AQPS.
    
    This class implements aggressive deferral logic for non-priority EVs,
    deciding when to defer charging to cheaper TOU periods.
    
    Aggressive Deferral Policy:
    - Defer whenever a cheaper slot exists before departure
    - Only constrained by departure feasibility and window congestion
    
    Attributes:
        tariff: TOU tariff instance
        deferral_tracker: Tracks deferral window congestion
        renewable: Optional renewable integration
        decisions: History of deferral decisions
    
    Examples:
        >>> tariff = TOUTariff(TOUTariffConfig(peak_price=0.40, off_peak_price=0.15))
        >>> optimizer = TOUOptimizer(tariff, max_evses=54)
        >>> decision = optimizer.should_defer(session, current_period=144)
    """
    
    def __init__(
        self,
        tariff: TOUTariff,
        max_evses: int = 54,
        deferral_safety_margin: float = 0.8,
        renewable: Optional['RenewableIntegration'] = None,
        start_hour: float = 0.0,
        voltage: float = 415.0,
        period_minutes: float = 5.0
    ):
        """
        Initialize TOU optimizer.
        
        Args:
            tariff: TOU tariff instance
            max_evses: Maximum EVSEs for deferral tracking
            deferral_safety_margin: Safety margin for deferral windows
            renewable: Optional renewable integration
            start_hour: Hour at which simulation starts
            voltage: System voltage (Volts)
            period_minutes: Duration of each period (minutes)
        """
        self.tariff = tariff
        self.deferral_tracker = DeferralTracker(max_evses, deferral_safety_margin)
        self.renewable = renewable
        self.start_hour = start_hour
        self.voltage = voltage
        self.period_minutes = period_minutes
        
        self.decisions: List[DeferralDecision] = []
        
        logger.info(
            f"TOUOptimizer initialized: aggressive deferral policy, "
            f"tariff={tariff.config.name}"
        )
    
    def can_defer_charging(
        self,
        session_id: str,
        departure_period: int,
        remaining_demand_kwh: float,
        max_rate_amps: float,
        current_period: int
    ) -> Tuple[bool, Optional[int], str]:
        """
        Check if charging can be deferred to a cheaper period.
        
        Aggressive Policy: Defer if ANY cheaper slot exists and is feasible.
        
        Args:
            session_id: Session ID
            departure_period: When session must complete
            remaining_demand_kwh: Energy still needed (kWh)
            max_rate_amps: Maximum charging rate (Amps)
            current_period: Current simulation period
        
        Returns:
            Tuple of (can_defer, target_period, reason)
        """
        # Find next cheaper period
        target_period = self.tariff.find_next_cheaper_period(
            current_period,
            max_lookahead=departure_period - current_period,
            start_hour=self.start_hour
        )
        
        if target_period is None:
            return False, None, "No cheaper period before departure"
        
        if target_period >= departure_period:
            return False, None, "Cheaper period is after departure"
        
        # Calculate if enough time after target period to complete charging
        time_available = departure_period - target_period
        max_power_kw = (max_rate_amps * self.voltage) / 1000.0
        hours_available = (time_available * self.period_minutes) / 60.0
        energy_deliverable = max_power_kw * hours_available
        
        if energy_deliverable < remaining_demand_kwh:
            return False, None, f"Insufficient time after deferral: need {remaining_demand_kwh:.1f}kWh, can deliver {energy_deliverable:.1f}kWh"
        
        # Check deferral window capacity
        if not self.deferral_tracker.can_defer_to(target_period):
            return False, None, f"Deferral window at period {target_period} is full"
        
        return True, target_period, "Deferral feasible"
    
    def should_defer(
        self,
        session_id: str,
        departure_period: int,
        remaining_demand_kwh: float,
        max_rate_amps: float,
        current_period: int,
        is_priority: bool = False
    ) -> DeferralDecision:
        """
        Decide whether to defer charging for a session.
        
        Aggressive Policy:
        - Always defer if feasible (cheaper slot exists + enough time)
        - Never defer priority EVs
        
        Args:
            session_id: Session ID
            departure_period: When session must complete
            remaining_demand_kwh: Energy still needed (kWh)
            max_rate_amps: Maximum charging rate (Amps)
            current_period: Current simulation period
            is_priority: Whether this is a priority session
        
        Returns:
            DeferralDecision with recommendation
        """
        current_price = self.tariff.get_price_at_period(
            current_period, self.start_hour
        )
        
        # Never defer priority EVs
        if is_priority:
            return DeferralDecision(
                session_id=session_id,
                current_period=current_period,
                target_period=None,
                current_price=current_price,
                target_price=current_price,
                savings_per_kwh=0.0,
                can_defer=False,
                reason="Priority EV - no deferral"
            )
        
        # Check PV/BESS - if renewables favor charging now, don't defer
        if self.renewable and self.renewable.should_prefer_charging(current_period):
            return DeferralDecision(
                session_id=session_id,
                current_period=current_period,
                target_period=None,
                current_price=current_price,
                target_price=current_price,
                savings_per_kwh=0.0,
                can_defer=False,
                reason="Renewables favor immediate charging"
            )
        
        # Check deferral feasibility
        can_defer, target_period, reason = self.can_defer_charging(
            session_id, departure_period, remaining_demand_kwh,
            max_rate_amps, current_period
        )
        
        if can_defer and target_period is not None:
            target_price = self.tariff.get_price_at_period(
                target_period, self.start_hour
            )
            savings = current_price - target_price
            
            # Register deferral
            self.deferral_tracker.register_deferral(session_id, target_period)
            
            decision = DeferralDecision(
                session_id=session_id,
                current_period=current_period,
                target_period=target_period,
                current_price=current_price,
                target_price=target_price,
                savings_per_kwh=savings,
                can_defer=True,
                reason=reason
            )
        else:
            decision = DeferralDecision(
                session_id=session_id,
                current_period=current_period,
                target_period=None,
                current_price=current_price,
                target_price=current_price,
                savings_per_kwh=0.0,
                can_defer=False,
                reason=reason
            )
        
        self.decisions.append(decision)
        return decision
    
    def get_deferral_rate(
        self,
        session_id: str,
        decision: DeferralDecision,
        min_rate: float = 0.0
    ) -> float:
        """
        Calculate charging rate based on deferral decision.
        
        Args:
            session_id: Session ID
            decision: Deferral decision
            min_rate: Minimum rate to maintain (Amps)
        
        Returns:
            Recommended charging rate (Amps)
        """
        if decision.can_defer:
            # Defer most charging, maintain minimum only
            return min_rate
        else:
            # Charge immediately - return None to indicate full rate
            return -1.0  # Signal for scheduler to use fair share
    
    def update_current_period(self, current_period: int) -> None:
        """Update tracker for new period (cleanup past windows)."""
        self.deferral_tracker.clear_past_windows(current_period)
    
    def get_statistics(self) -> Dict:
        """Get deferral statistics."""
        if not self.decisions:
            return {
                'total_decisions': 0,
                'deferred_count': 0,
                'deferred_pct': 0.0,
                'avg_savings_per_kwh': 0.0,
                'total_potential_savings': 0.0,
            }
        
        deferred = [d for d in self.decisions if d.can_defer]
        
        return {
            'total_decisions': len(self.decisions),
            'deferred_count': len(deferred),
            'deferred_pct': len(deferred) / len(self.decisions) * 100,
            'avg_savings_per_kwh': (
                sum(d.savings_per_kwh for d in deferred) / len(deferred)
                if deferred else 0.0
            ),
            'total_potential_savings': sum(d.savings_per_kwh for d in deferred),
            'deferral_window_usage': self.deferral_tracker.get_deferral_summary(),
        }
    
    def export_decisions(self) -> List[Dict]:
        """Export all deferral decisions."""
        return [d.to_dict() for d in self.decisions]
    
    def reset(self) -> None:
        """Reset optimizer state."""
        self.decisions.clear()
        self.deferral_tracker.reset()
    
    def __repr__(self) -> str:
        return f"TOUOptimizer(tariff={self.tariff.config.name})"


# =============================================================================
# Convenience Functions
# =============================================================================

def create_australian_tou_tariff(
    peak_price: float,
    off_peak_price: float,
    shoulder_price: Optional[float] = None,
    peak_start: float = 14.0,
    peak_end: float = 20.0,
    shoulder_start: Optional[float] = None,
    shoulder_end: Optional[float] = None,
    name: str = "Australian TOU"
) -> TOUTariff:
    """
    Create a typical Australian TOU tariff.
    
    Args:
        peak_price: Peak rate ($/kWh)
        off_peak_price: Off-peak rate ($/kWh)
        shoulder_price: Optional shoulder rate ($/kWh)
        peak_start: Peak period start hour (default 2pm)
        peak_end: Peak period end hour (default 8pm)
        shoulder_start: Optional shoulder start hour
        shoulder_end: Optional shoulder end hour
        name: Tariff name
    
    Returns:
        Configured TOUTariff
    """
    config = TOUTariffConfig(
        name=name,
        peak_price=peak_price,
        off_peak_price=off_peak_price,
        shoulder_price=shoulder_price,
        peak_hours=[(peak_start, peak_end)],
        shoulder_hours=[(shoulder_start, shoulder_end)] if shoulder_start else []
    )
    
    return TOUTariff(config)
