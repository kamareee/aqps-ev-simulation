"""
Renewable Energy Integration for AQPS.

This module implements PV (photovoltaic) solar and BESS (Battery Energy Storage System)
integration for the AQPS scheduler. It provides forecasted generation and storage profiles
that inform TOU optimization decisions.

Key Features:
- Forecasted PV generation profiles (list-based input)
- BESS state-of-charge and dispatch profiles
- Integration with TOU optimizer for cost-aware charging

Author: Research Team
Phase: 3 (TOU Optimization with Renewable Integration)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class BESSDispatchMode(Enum):
    """BESS dispatch operation modes."""
    IDLE = "idle"           # No charge/discharge
    CHARGING = "charging"   # Charging from grid/PV
    DISCHARGING = "discharging"  # Discharging to support load
    PEAK_SHAVING = "peak_shaving"  # Discharge during peak TOU
    PV_SMOOTHING = "pv_smoothing"  # Smooth PV intermittency


@dataclass
class PVForecastPoint:
    """
    Single point in a PV generation forecast.
    
    Attributes:
        period: Simulation period (timestep)
        generation_kw: Forecasted PV generation (kW)
        irradiance_factor: Optional irradiance multiplier (0-1)
        confidence: Forecast confidence level (0-1)
    """
    period: int
    generation_kw: float
    irradiance_factor: float = 1.0
    confidence: float = 0.9
    
    def to_dict(self) -> Dict:
        return {
            'period': self.period,
            'generation_kw': self.generation_kw,
            'irradiance_factor': self.irradiance_factor,
            'confidence': self.confidence,
        }


@dataclass
class BESSStatePoint:
    """
    Single point in a BESS state/dispatch forecast.
    
    Attributes:
        period: Simulation period (timestep)
        soc_kwh: State of charge (kWh)
        soc_pct: State of charge percentage (0-100)
        power_kw: Power flow (positive=discharge, negative=charge)
        mode: Current dispatch mode
        available_discharge_kw: Available discharge power
        available_charge_kw: Available charge power (headroom)
    """
    period: int
    soc_kwh: float
    soc_pct: float
    power_kw: float  # Positive = discharge, Negative = charge
    mode: BESSDispatchMode = BESSDispatchMode.IDLE
    available_discharge_kw: float = 0.0
    available_charge_kw: float = 0.0
    
    @property
    def is_discharging(self) -> bool:
        return self.power_kw > 0
    
    @property
    def is_charging(self) -> bool:
        return self.power_kw < 0
    
    def to_dict(self) -> Dict:
        return {
            'period': self.period,
            'soc_kwh': self.soc_kwh,
            'soc_pct': self.soc_pct,
            'power_kw': self.power_kw,
            'mode': self.mode.value,
            'available_discharge_kw': self.available_discharge_kw,
            'available_charge_kw': self.available_charge_kw,
        }


@dataclass
class PVSystemConfig:
    """
    Configuration for PV system.
    
    Attributes:
        capacity_kwp: Installed PV capacity (kWp - kilowatt peak)
        inverter_capacity_kw: Inverter capacity limit (kW)
        efficiency: System efficiency factor (0-1)
        orientation: Panel orientation (degrees from south)
        tilt: Panel tilt angle (degrees)
    """
    capacity_kwp: float = 100.0  # 100 kWp default
    inverter_capacity_kw: float = 100.0
    efficiency: float = 0.85
    orientation: float = 0.0  # South-facing
    tilt: float = 25.0  # Typical roof tilt


@dataclass
class BESSConfig:
    """
    Configuration for Battery Energy Storage System.
    
    Attributes:
        capacity_kwh: Total battery capacity (kWh)
        max_charge_kw: Maximum charging power (kW)
        max_discharge_kw: Maximum discharging power (kW)
        min_soc_pct: Minimum state of charge (%)
        max_soc_pct: Maximum state of charge (%)
        round_trip_efficiency: Charge/discharge efficiency
        initial_soc_pct: Initial state of charge (%)
    """
    capacity_kwh: float = 200.0  # 200 kWh default
    max_charge_kw: float = 50.0
    max_discharge_kw: float = 50.0
    min_soc_pct: float = 10.0  # Don't discharge below 10%
    max_soc_pct: float = 95.0  # Don't charge above 95%
    round_trip_efficiency: float = 0.90
    initial_soc_pct: float = 50.0
    
    @property
    def usable_capacity_kwh(self) -> float:
        """Calculate usable capacity considering SoC limits."""
        usable_range = (self.max_soc_pct - self.min_soc_pct) / 100.0
        return self.capacity_kwh * usable_range


class PVSystem:
    """
    PV Solar Generation System with forecasting support.
    
    This class manages PV generation forecasts and provides generation
    data to the TOU optimizer for cost-aware charging decisions.
    
    The system accepts forecasted generation profiles either as:
    - A list of power values (one per period)
    - A list of PVForecastPoint objects with metadata
    
    Attributes:
        config: PV system configuration
        forecast: List of PVForecastPoint objects
    
    Examples:
        >>> pv = PVSystem(PVSystemConfig(capacity_kwp=100))
        >>> pv.load_forecast_from_list([0, 0, 10, 30, 50, 70, 80, 70, 50, 30, 10, 0])
        >>> gen = pv.get_generation(period=5)  # Returns 70 kW
    """
    
    def __init__(self, config: Optional[PVSystemConfig] = None):
        """
        Initialize PV system.
        
        Args:
            config: PV system configuration (uses defaults if None)
        """
        self.config = config or PVSystemConfig()
        self.forecast: List[PVForecastPoint] = []
        self._forecast_by_period: Dict[int, PVForecastPoint] = {}
        
        logger.info(f"PVSystem initialized: {self.config.capacity_kwp} kWp capacity")
    
    def load_forecast_from_list(
        self,
        generation_values: List[float],
        start_period: int = 0,
        confidence: float = 0.9
    ) -> None:
        """
        Load PV forecast from a simple list of generation values.
        
        Args:
            generation_values: List of generation values (kW) per period
            start_period: Starting period for the forecast
            confidence: Confidence level for all points
        """
        self.forecast.clear()
        self._forecast_by_period.clear()
        
        for i, gen_kw in enumerate(generation_values):
            period = start_period + i
            # Cap at inverter capacity
            capped_gen = min(gen_kw, self.config.inverter_capacity_kw)
            
            point = PVForecastPoint(
                period=period,
                generation_kw=capped_gen,
                confidence=confidence
            )
            self.forecast.append(point)
            self._forecast_by_period[period] = point
        
        logger.info(f"Loaded PV forecast: {len(self.forecast)} periods")
    
    def load_forecast_from_points(self, points: List[PVForecastPoint]) -> None:
        """
        Load PV forecast from PVForecastPoint objects.
        
        Args:
            points: List of forecast points
        """
        self.forecast = sorted(points, key=lambda p: p.period)
        self._forecast_by_period = {p.period: p for p in self.forecast}
        
        logger.info(f"Loaded PV forecast: {len(self.forecast)} points")
    
    def get_generation(self, period: int) -> float:
        """
        Get PV generation for a specific period.
        
        Args:
            period: Simulation period
        
        Returns:
            Generation in kW (0 if no forecast for period)
        """
        point = self._forecast_by_period.get(period)
        if point:
            return point.generation_kw * self.config.efficiency
        return 0.0
    
    def get_forecast_range(
        self,
        start_period: int,
        end_period: int
    ) -> List[PVForecastPoint]:
        """
        Get forecast points for a period range.
        
        Args:
            start_period: Start of range (inclusive)
            end_period: End of range (exclusive)
        
        Returns:
            List of forecast points in range
        """
        return [
            p for p in self.forecast
            if start_period <= p.period < end_period
        ]
    
    def get_total_generation(
        self,
        start_period: int,
        end_period: int,
        period_minutes: float = 5.0
    ) -> float:
        """
        Calculate total energy generation over a period range.
        
        Args:
            start_period: Start of range (inclusive)
            end_period: End of range (exclusive)
            period_minutes: Duration of each period in minutes
        
        Returns:
            Total energy in kWh
        """
        total_kwh = 0.0
        hours_per_period = period_minutes / 60.0
        
        for period in range(start_period, end_period):
            gen_kw = self.get_generation(period)
            total_kwh += gen_kw * hours_per_period
        
        return total_kwh
    
    def is_generating(self, period: int, threshold_kw: float = 1.0) -> bool:
        """Check if PV is generating above threshold at given period."""
        return self.get_generation(period) >= threshold_kw
    
    def get_peak_generation_period(self) -> Optional[int]:
        """Find the period with maximum PV generation."""
        if not self.forecast:
            return None
        max_point = max(self.forecast, key=lambda p: p.generation_kw)
        return max_point.period
    
    def export_forecast(self) -> List[Dict]:
        """Export forecast as list of dictionaries."""
        return [p.to_dict() for p in self.forecast]
    
    def __repr__(self) -> str:
        return (f"PVSystem({self.config.capacity_kwp}kWp, "
                f"forecast={len(self.forecast)} periods)")


class BESSController:
    """
    Battery Energy Storage System Controller.
    
    This class manages BESS state forecasts and dispatch decisions,
    providing storage data to the TOU optimizer.
    
    The controller accepts forecasted BESS state profiles that include:
    - State of charge trajectory
    - Planned charge/discharge schedule
    - Available power headroom
    
    Attributes:
        config: BESS configuration
        forecast: List of BESSStatePoint objects
    
    Examples:
        >>> bess = BESSController(BESSConfig(capacity_kwh=200))
        >>> bess.load_forecast_from_list(soc_values, power_values)
        >>> state = bess.get_state(period=10)
    """
    
    def __init__(self, config: Optional[BESSConfig] = None):
        """
        Initialize BESS controller.
        
        Args:
            config: BESS configuration (uses defaults if None)
        """
        self.config = config or BESSConfig()
        self.forecast: List[BESSStatePoint] = []
        self._forecast_by_period: Dict[int, BESSStatePoint] = {}
        
        logger.info(
            f"BESSController initialized: {self.config.capacity_kwh} kWh, "
            f"±{self.config.max_discharge_kw}/{self.config.max_charge_kw} kW"
        )
    
    def load_forecast_from_list(
        self,
        soc_values: List[float],
        power_values: Optional[List[float]] = None,
        start_period: int = 0,
        soc_is_percentage: bool = True
    ) -> None:
        """
        Load BESS forecast from lists of SoC and power values.
        
        Args:
            soc_values: List of state-of-charge values (% or kWh)
            power_values: Optional list of power values (kW, positive=discharge)
            start_period: Starting period for the forecast
            soc_is_percentage: If True, soc_values are percentages (0-100)
        """
        self.forecast.clear()
        self._forecast_by_period.clear()
        
        if power_values is None:
            power_values = [0.0] * len(soc_values)
        
        for i, soc in enumerate(soc_values):
            period = start_period + i
            power = power_values[i] if i < len(power_values) else 0.0
            
            # Convert SoC to both kWh and %
            if soc_is_percentage:
                soc_pct = soc
                soc_kwh = (soc / 100.0) * self.config.capacity_kwh
            else:
                soc_kwh = soc
                soc_pct = (soc / self.config.capacity_kwh) * 100.0
            
            # Determine mode based on power
            if power > 0:
                mode = BESSDispatchMode.DISCHARGING
            elif power < 0:
                mode = BESSDispatchMode.CHARGING
            else:
                mode = BESSDispatchMode.IDLE
            
            # Calculate available headroom
            available_discharge = min(
                self.config.max_discharge_kw,
                (soc_pct - self.config.min_soc_pct) / 100.0 * self.config.capacity_kwh
            )
            available_charge = min(
                self.config.max_charge_kw,
                (self.config.max_soc_pct - soc_pct) / 100.0 * self.config.capacity_kwh
            )
            
            point = BESSStatePoint(
                period=period,
                soc_kwh=soc_kwh,
                soc_pct=soc_pct,
                power_kw=power,
                mode=mode,
                available_discharge_kw=max(0, available_discharge),
                available_charge_kw=max(0, available_charge)
            )
            self.forecast.append(point)
            self._forecast_by_period[period] = point
        
        logger.info(f"Loaded BESS forecast: {len(self.forecast)} periods")
    
    def load_forecast_from_points(self, points: List[BESSStatePoint]) -> None:
        """
        Load BESS forecast from BESSStatePoint objects.
        
        Args:
            points: List of state points
        """
        self.forecast = sorted(points, key=lambda p: p.period)
        self._forecast_by_period = {p.period: p for p in self.forecast}
        
        logger.info(f"Loaded BESS forecast: {len(self.forecast)} points")
    
    def get_state(self, period: int) -> Optional[BESSStatePoint]:
        """
        Get BESS state for a specific period.
        
        Args:
            period: Simulation period
        
        Returns:
            BESSStatePoint or None if no forecast
        """
        return self._forecast_by_period.get(period)
    
    def get_available_discharge(self, period: int) -> float:
        """
        Get available discharge power at a period.
        
        Args:
            period: Simulation period
        
        Returns:
            Available discharge power in kW
        """
        state = self.get_state(period)
        if state:
            return state.available_discharge_kw
        return 0.0
    
    def get_available_charge(self, period: int) -> float:
        """
        Get available charging headroom at a period.
        
        Args:
            period: Simulation period
        
        Returns:
            Available charge power in kW
        """
        state = self.get_state(period)
        if state:
            return state.available_charge_kw
        return 0.0
    
    def is_discharging(self, period: int) -> bool:
        """Check if BESS is discharging at given period."""
        state = self.get_state(period)
        return state.is_discharging if state else False
    
    def is_charging(self, period: int) -> bool:
        """Check if BESS is charging at given period."""
        state = self.get_state(period)
        return state.is_charging if state else False
    
    def get_soc(self, period: int) -> float:
        """Get state of charge percentage at a period."""
        state = self.get_state(period)
        return state.soc_pct if state else self.config.initial_soc_pct
    
    def can_support_load(self, period: int, load_kw: float) -> bool:
        """
        Check if BESS can support a given load at a period.
        
        Args:
            period: Simulation period
            load_kw: Required load in kW
        
        Returns:
            True if BESS can supply the load
        """
        available = self.get_available_discharge(period)
        return available >= load_kw
    
    def get_forecast_range(
        self,
        start_period: int,
        end_period: int
    ) -> List[BESSStatePoint]:
        """Get forecast points for a period range."""
        return [
            p for p in self.forecast
            if start_period <= p.period < end_period
        ]
    
    def export_forecast(self) -> List[Dict]:
        """Export forecast as list of dictionaries."""
        return [p.to_dict() for p in self.forecast]
    
    def __repr__(self) -> str:
        return (f"BESSController({self.config.capacity_kwh}kWh, "
                f"forecast={len(self.forecast)} periods)")


@dataclass
class RenewableIntegrationConfig:
    """
    Configuration for combined PV and BESS integration.
    
    Attributes:
        pv_config: PV system configuration
        bess_config: BESS configuration
        prefer_pv_charging: Prefer EV charging when PV is generating
        bess_peak_shaving: Use BESS to reduce peak TOU costs
        min_pv_for_preference: Minimum PV generation to trigger preference (kW)
        min_bess_soc_for_support: Minimum BESS SoC to support load (%)
    """
    pv_config: Optional[PVSystemConfig] = None
    bess_config: Optional[BESSConfig] = None
    prefer_pv_charging: bool = True
    bess_peak_shaving: bool = True
    min_pv_for_preference: float = 10.0  # kW
    min_bess_soc_for_support: float = 30.0  # %


class RenewableIntegration:
    """
    Combined PV and BESS integration manager for AQPS.
    
    This class coordinates PV generation and BESS dispatch to inform
    TOU optimization decisions in the scheduler.
    
    Integration logic:
    - When PV generation is high → prefer EV charging (lower effective cost)
    - When BESS is discharging during peak → reduced need for EV deferral
    - When BESS SoC is high → opportunity to support charging
    
    Attributes:
        config: Integration configuration
        pv: PV system instance
        bess: BESS controller instance
    """
    
    def __init__(self, config: Optional[RenewableIntegrationConfig] = None):
        """
        Initialize renewable integration manager.
        
        Args:
            config: Integration configuration
        """
        self.config = config or RenewableIntegrationConfig()
        self.pv = PVSystem(self.config.pv_config)
        self.bess = BESSController(self.config.bess_config)
        
        logger.info("RenewableIntegration initialized")
    
    def load_pv_forecast(
        self,
        values: List[float],
        start_period: int = 0
    ) -> None:
        """Load PV generation forecast from list."""
        self.pv.load_forecast_from_list(values, start_period)
    
    def load_bess_forecast(
        self,
        soc_values: List[float],
        power_values: Optional[List[float]] = None,
        start_period: int = 0
    ) -> None:
        """Load BESS state forecast from lists."""
        self.bess.load_forecast_from_list(
            soc_values, power_values, start_period
        )
    
    def should_prefer_charging(self, period: int) -> bool:
        """
        Determine if EV charging should be preferred at this period.
        
        Charging is preferred when:
        - PV generation exceeds threshold, OR
        - BESS is at high SoC and not discharging
        
        Args:
            period: Simulation period
        
        Returns:
            True if charging should be preferred
        """
        if not self.config.prefer_pv_charging:
            return False
        
        # Check PV generation
        pv_gen = self.pv.get_generation(period)
        if pv_gen >= self.config.min_pv_for_preference:
            return True
        
        # Check BESS state
        bess_soc = self.bess.get_soc(period)
        if bess_soc >= 80.0 and not self.bess.is_discharging(period):
            return True
        
        return False
    
    def get_effective_capacity_offset(self, period: int) -> float:
        """
        Calculate effective capacity offset from renewables.
        
        This represents additional capacity available from:
        - PV generation
        - BESS discharge
        
        Args:
            period: Simulation period
        
        Returns:
            Effective capacity offset in kW
        """
        offset = 0.0
        
        # Add PV generation
        offset += self.pv.get_generation(period)
        
        # Add BESS discharge if peak shaving
        if self.config.bess_peak_shaving:
            bess_state = self.bess.get_state(period)
            if bess_state and bess_state.is_discharging:
                offset += bess_state.power_kw
        
        return offset
    
    def get_renewable_status(self, period: int) -> Dict:
        """
        Get current renewable energy status.
        
        Returns:
            Dictionary with PV and BESS status
        """
        pv_gen = self.pv.get_generation(period)
        bess_state = self.bess.get_state(period)
        
        return {
            'period': period,
            'pv_generation_kw': pv_gen,
            'pv_generating': self.pv.is_generating(period),
            'bess_soc_pct': bess_state.soc_pct if bess_state else 0,
            'bess_power_kw': bess_state.power_kw if bess_state else 0,
            'bess_mode': bess_state.mode.value if bess_state else 'unknown',
            'prefer_charging': self.should_prefer_charging(period),
            'capacity_offset_kw': self.get_effective_capacity_offset(period),
        }
    
    def __repr__(self) -> str:
        return f"RenewableIntegration(pv={self.pv}, bess={self.bess})"


# =============================================================================
# Example PV and BESS Profile Generators
# =============================================================================

def generate_typical_pv_profile(
    periods: int = 288,
    peak_kw: float = 80.0,
    sunrise_period: int = 72,  # ~6am if 5-min periods
    sunset_period: int = 216   # ~6pm if 5-min periods
) -> List[float]:
    """
    Generate a typical bell-curve PV generation profile.
    
    Args:
        periods: Total number of periods in the day
        peak_kw: Peak generation (kW)
        sunrise_period: Period when generation starts
        sunset_period: Period when generation ends
    
    Returns:
        List of generation values (kW) per period
    """
    import math
    
    profile = []
    solar_duration = sunset_period - sunrise_period
    solar_midpoint = sunrise_period + solar_duration // 2
    
    for period in range(periods):
        if sunrise_period <= period <= sunset_period:
            # Bell curve centered at solar noon
            x = (period - solar_midpoint) / (solar_duration / 2)
            # Parabolic profile
            gen = peak_kw * max(0, 1 - x * x)
        else:
            gen = 0.0
        profile.append(gen)
    
    return profile


def generate_typical_bess_profile(
    periods: int = 288,
    capacity_kwh: float = 200.0,
    initial_soc_pct: float = 50.0,
    peak_periods: Tuple[int, int] = (144, 216),  # 12pm-6pm
    discharge_rate_kw: float = 30.0
) -> Tuple[List[float], List[float]]:
    """
    Generate a typical BESS SoC and power profile for peak shaving.
    
    Args:
        periods: Total number of periods
        capacity_kwh: Battery capacity (kWh)
        initial_soc_pct: Starting SoC (%)
        peak_periods: (start, end) periods for peak discharge
        discharge_rate_kw: Discharge rate during peak (kW)
    
    Returns:
        Tuple of (soc_values, power_values) lists
    """
    soc_values = []
    power_values = []
    
    current_soc = initial_soc_pct
    period_hours = 5 / 60  # 5-minute periods
    
    peak_start, peak_end = peak_periods
    
    for period in range(periods):
        if peak_start <= period < peak_end:
            # Discharge during peak
            power = discharge_rate_kw
            energy_delta = power * period_hours
            soc_delta = (energy_delta / capacity_kwh) * 100
            current_soc = max(10, current_soc - soc_delta)  # Min 10% SoC
        elif period >= peak_end:
            # Charge during off-peak (recovery)
            power = -discharge_rate_kw * 0.5  # Charge at half rate
            energy_delta = abs(power) * period_hours
            soc_delta = (energy_delta / capacity_kwh) * 100
            current_soc = min(95, current_soc + soc_delta)  # Max 95% SoC
        else:
            # Idle
            power = 0.0
        
        soc_values.append(current_soc)
        power_values.append(power)
    
    return soc_values, power_values
