"""
Phase 5: Comprehensive Simulation Studies for AQPS

This module implements the complete simulation study comparing AQPS against
LLF baseline across six scenarios (S1-S6) with publication-quality visualization.

Scenarios:
- S1: Baseline (27% priority, uniform arrivals)
- S2: Low Priority (10% priority, uniform arrivals)
- S3: High Priority (50% priority, uniform arrivals)
- S4: Morning Rush (27% priority, clustered AM arrivals)
- S5: Cloudy Day (27% priority, reduced PV generation)
- S6: Peak Stress (50% priority, PM clustering)

Metrics:
- Priority fulfillment rate (target: 100%)
- Non-priority energy delivery percentage
- Total energy cost vs uncontrolled baseline
- Computation time vs LLF baseline
- Preemption frequency
- Threshold violation conditions

Author: Research Team
Phase: 5 (Simulation Studies & Analysis)
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

# Import AQPS modules (from local directory)
from src.aqps.data_structures import (
    SessionInfo,
    AQPSConfig,
    Phase3Config,
    Phase4Config,
    J1772_PILOT_SIGNALS,
)
from src.aqps.scheduler import AdaptiveQueuingPriorityScheduler
from src.aqps.scenario_generator import generate_scenario
from src.aqps.renewable_integration import (
    generate_typical_pv_profile,
    generate_typical_bess_profile,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Suppress matplotlib font warnings
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# Create results directory
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


@dataclass
class SimulationResult:
    """Results from a single scenario simulation."""

    scenario_name: str
    algorithm: str  # 'AQPS' or 'LLF'

    # Session statistics
    total_sessions: int
    priority_sessions: int
    non_priority_sessions: int

    # Fulfillment metrics
    priority_fulfilled: int
    priority_fulfillment_rate: float
    non_priority_energy_delivered_pct: float

    # Cost metrics
    total_energy_delivered_kwh: float
    total_energy_cost: float

    # Operational metrics
    total_preemptions: int
    total_threshold_violations: int
    total_deferrals: int

    # Computational metrics
    avg_schedule_time_ms: float
    max_schedule_time_ms: float
    total_schedule_time_ms: float

    # Quantization metrics (AQPS only)
    quantization_efficiency_avg: float

    # Additional data
    capacity_utilization_avg: float


# =============================================================================
# LLF (Least-Laxity-First) Baseline Algorithm
# =============================================================================


class LeastLaxityFirstScheduler:
    """
    Simple Least-Laxity-First (LLF) baseline algorithm.

    This serves as a baseline comparison for AQPS. LLF:
    - Sorts all sessions by laxity (lowest first)
    - Allocates capacity in laxity order
    - No priority guarantees
    - No TOU optimization
    - No quantization
    """

    def __init__(
        self,
        total_capacity: float = 600.0,
        voltage: float = 415.0,
        period_minutes: float = 5.0,
    ):
        self.total_capacity = total_capacity
        self.voltage = voltage
        self.period_minutes = period_minutes
        self.schedule_times: List[float] = []

    def schedule(
        self, sessions: List[SessionInfo], current_time: int
    ) -> Dict[str, float]:
        """Generate schedule using LLF policy."""
        start = time.perf_counter()

        if not sessions:
            return {}

        # Calculate laxity for all sessions
        from src.aqps.utils import calculate_laxity

        sessions_with_laxity = []
        for session in sessions:
            laxity = calculate_laxity(
                session, current_time, self.voltage, self.period_minutes
            )
            sessions_with_laxity.append((session, laxity))

        # Sort by laxity (lowest first)
        sessions_with_laxity.sort(key=lambda x: x[1])

        # Fair-share allocation
        schedule = {}
        remaining_capacity = self.total_capacity
        n_sessions = len(sessions)

        for session, laxity in sessions_with_laxity:
            if remaining_capacity <= 0:
                schedule[session.station_id] = 0.0
                continue

            fair_share = remaining_capacity / n_sessions
            desired_rate = min(fair_share, session.max_rate)
            allocated = min(desired_rate, remaining_capacity)

            schedule[session.station_id] = allocated
            remaining_capacity -= allocated
            n_sessions -= 1

        elapsed = (time.perf_counter() - start) * 1000.0
        self.schedule_times.append(elapsed)

        return schedule

    def get_computational_statistics(self) -> Dict:
        """Get computational statistics."""
        if not self.schedule_times:
            return {
                "avg_time_ms": 0.0,
                "max_time_ms": 0.0,
                "total_time_ms": 0.0,
            }

        return {
            "avg_time_ms": np.mean(self.schedule_times),
            "max_time_ms": np.max(self.schedule_times),
            "total_time_ms": np.sum(self.schedule_times),
        }


# =============================================================================
# Simulation Runner
# =============================================================================


def run_simulation(
    sessions: List[SessionInfo],
    scheduler,
    scenario_name: str,
    algorithm_name: str,
    n_periods: int = 288,
    voltage: float = 415.0,
    period_minutes: float = 5.0,
) -> SimulationResult:
    """
    Run a complete simulation for given sessions and scheduler.

    Args:
        sessions: List of charging sessions
        scheduler: Scheduler instance (AQPS or LLF)
        scenario_name: Name of scenario
        algorithm_name: 'AQPS' or 'LLF'
        n_periods: Number of simulation periods
        voltage: System voltage
        period_minutes: Period duration

    Returns:
        SimulationResult with comprehensive metrics
    """
    logger.info(f"Running {algorithm_name} simulation for {scenario_name}")

    # Track session energy delivered
    session_energy: Dict[str, float] = {s.session_id: 0.0 for s in sessions}

    # Run simulation
    for t in range(n_periods):
        # Get active sessions
        active = [s for s in sessions if s.arrival_time <= t < s.departure_time]

        if not active:
            continue

        # Generate schedule
        schedule = scheduler.schedule(active, t)

        # Update session charges
        hours_per_period = period_minutes / 60.0
        for session in active:
            rate = schedule.get(session.station_id, 0.0)
            power_kw = (rate * voltage) / 1000.0
            energy = power_kw * hours_per_period

            session_energy[session.session_id] += energy
            session.current_charge += energy

    # Calculate fulfillment metrics
    priority_sessions = [s for s in sessions if s.is_priority]
    non_priority_sessions = [s for s in sessions if not s.is_priority]

    priority_fulfilled = sum(
        1
        for s in priority_sessions
        if session_energy[s.session_id] >= s.energy_requested * 0.99
    )

    priority_fulfillment_rate = (
        priority_fulfilled / len(priority_sessions) * 100
        if priority_sessions
        else 100.0
    )

    # Non-priority energy delivered percentage
    if non_priority_sessions:
        total_requested = sum(s.energy_requested for s in non_priority_sessions)
        total_delivered = sum(
            session_energy[s.session_id] for s in non_priority_sessions
        )
        non_priority_pct = (
            (total_delivered / total_requested * 100) if total_requested > 0 else 100.0
        )
    else:
        non_priority_pct = 100.0

    # Total energy delivered
    total_energy_kwh = sum(session_energy.values())

    # Get algorithm-specific metrics
    if algorithm_name == "AQPS":
        comp_stats = scheduler.get_computational_statistics()
        preempt_stats = scheduler.get_preemption_statistics()
        threshold_stats = scheduler.get_threshold_statistics()
        tou_stats = scheduler.get_tou_statistics()
        quant_stats = scheduler.get_quantization_statistics()

        # Get summary
        summary = scheduler.get_simulation_summary()

        result = SimulationResult(
            scenario_name=scenario_name,
            algorithm="AQPS",
            total_sessions=len(sessions),
            priority_sessions=len(priority_sessions),
            non_priority_sessions=len(non_priority_sessions),
            priority_fulfilled=priority_fulfilled,
            priority_fulfillment_rate=priority_fulfillment_rate,
            non_priority_energy_delivered_pct=non_priority_pct,
            total_energy_delivered_kwh=total_energy_kwh,
            total_energy_cost=0.0,  # Would need TOU cost calculation
            total_preemptions=preempt_stats.get("total_preemptions", 0),
            total_threshold_violations=threshold_stats.get("total_violations", 0),
            total_deferrals=tou_stats.get("total_deferrals", 0),
            avg_schedule_time_ms=comp_stats.get("avg_time_ms", 0.0),
            max_schedule_time_ms=comp_stats.get("max_time_ms", 0.0),
            total_schedule_time_ms=comp_stats.get("total_time_ms", 0.0),
            quantization_efficiency_avg=quant_stats.get("avg_efficiency", 100.0),
            capacity_utilization_avg=summary.avg_capacity_utilization,
        )
    else:  # LLF
        comp_stats = scheduler.get_computational_statistics()

        result = SimulationResult(
            scenario_name=scenario_name,
            algorithm="LLF",
            total_sessions=len(sessions),
            priority_sessions=len(priority_sessions),
            non_priority_sessions=len(non_priority_sessions),
            priority_fulfilled=priority_fulfilled,
            priority_fulfillment_rate=priority_fulfillment_rate,
            non_priority_energy_delivered_pct=non_priority_pct,
            total_energy_delivered_kwh=total_energy_kwh,
            total_energy_cost=0.0,
            total_preemptions=0,
            total_threshold_violations=0,
            total_deferrals=0,
            avg_schedule_time_ms=comp_stats.get("avg_time_ms", 0.0),
            max_schedule_time_ms=comp_stats.get("max_time_ms", 0.0),
            total_schedule_time_ms=comp_stats.get("total_time_ms", 0.0),
            quantization_efficiency_avg=100.0,
            capacity_utilization_avg=0.0,
        )

    logger.info(
        f"{algorithm_name} - Priority: {priority_fulfillment_rate:.1f}%, "
        f"Non-Priority: {non_priority_pct:.1f}%, "
        f"Avg Time: {result.avg_schedule_time_ms:.3f}ms"
    )

    return result


# =============================================================================
# Scenario Runners
# =============================================================================


def run_scenario_S1() -> Tuple[SimulationResult, SimulationResult]:
    """
    S1: Baseline (27% priority, uniform arrivals)

    Standard operational scenario with balanced priority ratio.
    """
    logger.info("=" * 60)
    logger.info("Running Scenario S1: Baseline")
    logger.info("=" * 60)

    # Generate sessions
    sessions_aqps = generate_scenario("S1", n_sessions=45, seed=42)
    sessions_llf = generate_scenario("S1", n_sessions=45, seed=42)

    # Configure AQPS
    config = AQPSConfig(
        min_priority_rate=16.0,
        total_capacity=560.0,
        period_minutes=5.0,
        voltage=415.0,
        enable_logging=False,
    )

    phase3_config = Phase3Config(
        enable_tou_optimization=True,
        enable_renewable_integration=True,
        deferral_policy="aggressive",
    )

    phase4_config = Phase4Config(
        enable_quantization=True, priority_ceil_enabled=True, enable_timing=True
    )

    aqps = AdaptiveQueuingPriorityScheduler(config, phase3_config, phase4_config)

    # Configure TOU
    aqps.configure_tou(
        peak_price=0.26668, off_peak_price=0.05623, peak_hours=[(8, 10), (16, 18)]
    )

    # Configure network
    aqps.configure_network(
        phase_a_limit=188.0, phase_b_limit=188.0, phase_c_limit=188.0
    )

    # Run AQPS simulation
    result_aqps = run_simulation(sessions_aqps, aqps, "S1: Baseline", "AQPS")

    # Configure LLF
    llf = LeastLaxityFirstScheduler(total_capacity=560.0, voltage=415.0)

    # Run LLF simulation
    result_llf = run_simulation(sessions_llf, llf, "S1: Baseline", "LLF")

    return result_aqps, result_llf


def run_scenario_S2() -> Tuple[SimulationResult, SimulationResult]:
    """
    S2: Low Priority (10% priority, uniform arrivals)

    Few urgent vehicles scenario.
    """
    logger.info("=" * 60)
    logger.info("Running Scenario S2: Low Priority")
    logger.info("=" * 60)

    # Generate sessions
    sessions_aqps = generate_scenario("S2", n_sessions=45, seed=43)
    sessions_llf = generate_scenario("S2", n_sessions=45, seed=43)

    # Configure AQPS
    config = AQPSConfig(
        min_priority_rate=16.0,
        total_capacity=560.0,
        period_minutes=5.0,
        voltage=415.0,
        enable_logging=False,
    )

    phase3_config = Phase3Config(
        enable_tou_optimization=True, deferral_policy="aggressive"
    )

    phase4_config = Phase4Config(enable_quantization=True, enable_timing=True)

    aqps = AdaptiveQueuingPriorityScheduler(config, phase3_config, phase4_config)
    aqps.configure_tou(
        peak_price=0.26668, off_peak_price=0.05623, peak_hours=[(8, 10), (16, 18)]
    )
    aqps.configure_network(188.0, 188.0, 188.0)

    result_aqps = run_simulation(sessions_aqps, aqps, "S2: Low Priority", "AQPS")

    # LLF
    llf = LeastLaxityFirstScheduler(total_capacity=560.0, voltage=415.0)
    result_llf = run_simulation(sessions_llf, llf, "S2: Low Priority", "LLF")

    return result_aqps, result_llf


def run_scenario_S3() -> Tuple[SimulationResult, SimulationResult]:
    """
    S3: High Priority (50% priority, uniform arrivals)

    Many urgent vehicles scenario - tests system limits.
    """
    logger.info("=" * 60)
    logger.info("Running Scenario S3: High Priority")
    logger.info("=" * 60)

    sessions_aqps = generate_scenario("S3", n_sessions=45, seed=44)
    sessions_llf = generate_scenario("S3", n_sessions=45, seed=44)

    config = AQPSConfig(
        min_priority_rate=16.0,
        total_capacity=560.0,
        voltage=415.0,
        enable_logging=False,
    )

    phase3_config = Phase3Config(enable_tou_optimization=True)
    phase4_config = Phase4Config(enable_quantization=True, enable_timing=True)

    aqps = AdaptiveQueuingPriorityScheduler(config, phase3_config, phase4_config)
    aqps.configure_tou(
        peak_price=0.26668, off_peak_price=0.05623, peak_hours=[(8, 10), (16, 18)]
    )
    aqps.configure_network(188.0, 188.0, 188.0)

    result_aqps = run_simulation(sessions_aqps, aqps, "S3: High Priority", "AQPS")

    llf = LeastLaxityFirstScheduler(total_capacity=560.0, voltage=415.0)
    result_llf = run_simulation(sessions_llf, llf, "S3: High Priority", "LLF")

    return result_aqps, result_llf


def run_scenario_S4() -> Tuple[SimulationResult, SimulationResult]:
    """
    S4: Morning Rush (27% priority, clustered AM arrivals)

    Clustered morning arrival pattern.
    """
    logger.info("=" * 60)
    logger.info("Running Scenario S4: Morning Rush")
    logger.info("=" * 60)

    sessions_aqps = generate_scenario("S4", n_sessions=45, seed=45)
    sessions_llf = generate_scenario("S4", n_sessions=45, seed=45)

    config = AQPSConfig(
        min_priority_rate=16.0,
        total_capacity=560.0,
        voltage=415.0,
        enable_logging=False,
    )

    phase3_config = Phase3Config(enable_tou_optimization=True)
    phase4_config = Phase4Config(enable_quantization=True, enable_timing=True)

    aqps = AdaptiveQueuingPriorityScheduler(config, phase3_config, phase4_config)
    aqps.configure_tou(
        peak_price=0.26668, off_peak_price=0.05623, peak_hours=[(8, 10), (16, 18)]
    )
    aqps.configure_network(188.0, 188.0, 188.0)

    result_aqps = run_simulation(sessions_aqps, aqps, "S4: Morning Rush", "AQPS")

    llf = LeastLaxityFirstScheduler(total_capacity=560.0, voltage=415.0)
    result_llf = run_simulation(sessions_llf, llf, "S4: Morning Rush", "LLF")

    return result_aqps, result_llf


def run_scenario_S5() -> Tuple[SimulationResult, SimulationResult]:
    """
    S5: Cloudy Day (27% priority, reduced PV generation)

    Reduced solar generation scenario.
    """
    logger.info("=" * 60)
    logger.info("Running Scenario S5: Cloudy Day")
    logger.info("=" * 60)

    sessions_aqps = generate_scenario("S5", n_sessions=45, seed=46)
    sessions_llf = generate_scenario("S5", n_sessions=45, seed=46)

    config = AQPSConfig(
        min_priority_rate=16.0,
        total_capacity=560.0,
        voltage=415.0,
        enable_logging=False,
    )

    phase3_config = Phase3Config(
        enable_tou_optimization=True, enable_renewable_integration=True
    )
    phase4_config = Phase4Config(enable_quantization=True, enable_timing=True)

    aqps = AdaptiveQueuingPriorityScheduler(config, phase3_config, phase4_config)
    aqps.configure_tou(
        peak_price=0.26668, off_peak_price=0.05623, peak_hours=[(8, 10), (16, 18)]
    )
    aqps.configure_network(188.0, 188.0, 188.0)

    # Configure renewables with REDUCED PV (50% of typical)
    aqps.configure_renewables(pv_capacity_kwp=100.0, bess_capacity_kwh=200.0)

    # Generate reduced PV profile (50% of typical)
    typical_pv = generate_typical_pv_profile(periods=288, peak_kw=80.0)
    reduced_pv = [p * 0.5 for p in typical_pv]  # Cloudy day = 50% generation
    aqps.load_pv_forecast(reduced_pv)

    # Generate BESS profile
    soc_vals, power_vals = generate_typical_bess_profile(periods=288)
    aqps.load_bess_forecast(soc_vals, power_vals)

    result_aqps = run_simulation(sessions_aqps, aqps, "S5: Cloudy Day", "AQPS")

    llf = LeastLaxityFirstScheduler(total_capacity=560.0, voltage=415.0)
    result_llf = run_simulation(sessions_llf, llf, "S5: Cloudy Day", "LLF")

    return result_aqps, result_llf


def run_scenario_S6() -> Tuple[SimulationResult, SimulationResult]:
    """
    S6: Peak Stress (50% priority, PM clustering)

    High demand with afternoon clustering - maximum stress test.
    """
    logger.info("=" * 60)
    logger.info("Running Scenario S6: Peak Stress")
    logger.info("=" * 60)

    sessions_aqps = generate_scenario("S6", n_sessions=45, seed=47)
    sessions_llf = generate_scenario("S6", n_sessions=45, seed=47)

    config = AQPSConfig(
        min_priority_rate=16.0,
        total_capacity=560.0,
        voltage=415.0,
        enable_logging=False,
    )

    phase3_config = Phase3Config(enable_tou_optimization=True)
    phase4_config = Phase4Config(enable_quantization=True, enable_timing=True)

    aqps = AdaptiveQueuingPriorityScheduler(config, phase3_config, phase4_config)
    aqps.configure_tou(
        peak_price=0.26668, off_peak_price=0.05623, peak_hours=[(8, 10), (16, 18)]
    )
    aqps.configure_network(188.0, 188.0, 188.0)

    result_aqps = run_simulation(sessions_aqps, aqps, "S6: Peak Stress", "AQPS")

    llf = LeastLaxityFirstScheduler(total_capacity=560.0, voltage=415.0)
    result_llf = run_simulation(sessions_llf, llf, "S6: Peak Stress", "LLF")

    return result_aqps, result_llf


# =============================================================================
# Visualization Functions
# =============================================================================


def generate_fulfillment_comparison(
    results: List[Tuple[SimulationResult, SimulationResult]], output_path: str
):
    """
    Generate fulfillment rate comparison plot.

    Shows priority and non-priority fulfillment for AQPS vs LLF across scenarios.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    scenarios = ["S1", "S2", "S3", "S4", "S5", "S6"]
    x = np.arange(len(scenarios))
    width = 0.35

    # Extract data
    aqps_priority = [r[0].priority_fulfillment_rate for r in results]
    llf_priority = [r[1].priority_fulfillment_rate for r in results]
    aqps_non_priority = [r[0].non_priority_energy_delivered_pct for r in results]
    llf_non_priority = [r[1].non_priority_energy_delivered_pct for r in results]

    # Priority fulfillment
    ax1.bar(
        x - width / 2, aqps_priority, width, label="AQPS", color="#2E7D32", alpha=0.8
    )
    ax1.bar(x + width / 2, llf_priority, width, label="LLF", color="#D32F2F", alpha=0.8)
    ax1.axhline(
        y=100,
        color="black",
        linestyle="--",
        linewidth=1,
        alpha=0.5,
        label="Target (100%)",
    )
    ax1.set_ylabel("Priority Fulfillment Rate (%)", fontsize=12, fontweight="bold")
    ax1.set_title("Priority EV Fulfillment", fontsize=14, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 110])

    # Non-priority energy delivery
    ax2.bar(
        x - width / 2,
        aqps_non_priority,
        width,
        label="AQPS",
        color="#1976D2",
        alpha=0.8,
    )
    ax2.bar(
        x + width / 2, llf_non_priority, width, label="LLF", color="#F57C00", alpha=0.8
    )
    ax2.set_ylabel("Energy Delivered (%)", fontsize=12, fontweight="bold")
    ax2.set_title("Non-Priority EV Energy Delivery", fontsize=14, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenarios)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 110])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved fulfillment comparison to {output_path}")


def generate_computational_comparison(
    results: List[Tuple[SimulationResult, SimulationResult]], output_path: str
):
    """
    Generate computational performance comparison.

    Shows average scheduling time for AQPS vs LLF.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    scenarios = ["S1", "S2", "S3", "S4", "S5", "S6"]
    x = np.arange(len(scenarios))
    width = 0.35

    aqps_times = [r[0].avg_schedule_time_ms for r in results]
    llf_times = [r[1].avg_schedule_time_ms for r in results]

    ax.bar(x - width / 2, aqps_times, width, label="AQPS", color="#6A1B9A", alpha=0.8)
    ax.bar(x + width / 2, llf_times, width, label="LLF", color="#00796B", alpha=0.8)

    ax.set_ylabel("Average Scheduling Time (ms)", fontsize=12, fontweight="bold")
    ax.set_title("Computational Performance Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    # Add speedup annotations
    for i, (aqps_t, llf_t) in enumerate(zip(aqps_times, llf_times)):
        if aqps_t < llf_t:
            speedup = llf_t / aqps_t
            ax.text(
                i,
                max(aqps_t, llf_t) + 0.5,
                f"{speedup:.1f}x faster",
                ha="center",
                fontsize=9,
                color="green",
                fontweight="bold",
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved computational comparison to {output_path}")


def generate_operational_metrics(
    results: List[Tuple[SimulationResult, SimulationResult]], output_path: str
):
    """
    Generate operational metrics comparison.

    Shows preemptions, threshold violations, and deferrals for AQPS.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

    scenarios = ["S1", "S2", "S3", "S4", "S5", "S6"]
    x = np.arange(len(scenarios))

    preemptions = [r[0].total_preemptions for r in results]
    violations = [r[0].total_threshold_violations for r in results]
    deferrals = [r[0].total_deferrals for r in results]

    # Preemptions
    ax1.bar(x, preemptions, color="#E64A19", alpha=0.8)
    ax1.set_ylabel("Count", fontsize=12, fontweight="bold")
    ax1.set_title("Preemption Events", fontsize=14, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios)
    ax1.grid(True, alpha=0.3, axis="y")

    # Threshold violations
    ax2.bar(x, violations, color="#C62828", alpha=0.8)
    ax2.set_ylabel("Count", fontsize=12, fontweight="bold")
    ax2.set_title("Threshold Violations", fontsize=14, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenarios)
    ax2.grid(True, alpha=0.3, axis="y")

    # TOU Deferrals
    ax3.bar(x, deferrals, color="#1565C0", alpha=0.8)
    ax3.set_ylabel("Count", fontsize=12, fontweight="bold")
    ax3.set_title("TOU Deferral Events", fontsize=14, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(scenarios)
    ax3.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved operational metrics to {output_path}")


def generate_summary_table(
    results: List[Tuple[SimulationResult, SimulationResult]], output_path: str
):
    """
    Generate comprehensive summary table as figure.
    """
    scenarios = ["S1", "S2", "S3", "S4", "S5", "S6"]

    # Prepare table data
    table_data = []
    headers = [
        "Scenario",
        "Algorithm",
        "Priority\nFulfill %",
        "Non-Priority\nEnergy %",
        "Avg Time\n(ms)",
        "Preemptions",
        "Violations",
        "Deferrals",
    ]

    for i, (r_aqps, r_llf) in enumerate(results):
        # AQPS row
        table_data.append(
            [
                scenarios[i],
                "AQPS",
                f"{r_aqps.priority_fulfillment_rate:.1f}",
                f"{r_aqps.non_priority_energy_delivered_pct:.1f}",
                f"{r_aqps.avg_schedule_time_ms:.2f}",
                f"{r_aqps.total_preemptions}",
                f"{r_aqps.total_threshold_violations}",
                f"{r_aqps.total_deferrals}",
            ]
        )

        # LLF row
        table_data.append(
            [
                "",
                "LLF",
                f"{r_llf.priority_fulfillment_rate:.1f}",
                f"{r_llf.non_priority_energy_delivered_pct:.1f}",
                f"{r_llf.avg_schedule_time_ms:.2f}",
                "-",
                "-",
                "-",
            ]
        )

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis("tight")
    ax.axis("off")

    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc="center",
        loc="center",
        colWidths=[0.12, 0.12, 0.14, 0.14, 0.12, 0.12, 0.12, 0.12],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header
    for i in range(len(headers)):
        cell = table[(0, i)]
        cell.set_facecolor("#2E7D32")
        cell.set_text_props(weight="bold", color="white")

    # Style cells
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            cell = table[(i, j)]
            if table_data[i - 1][1] == "AQPS":
                cell.set_facecolor("#E8F5E9")
            else:
                cell.set_facecolor("#FFEBEE")

    plt.title("Performance Summary Table", fontsize=16, fontweight="bold", pad=20)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved summary table to {output_path}")


def generate_scenario_detail_plot(
    result_aqps: SimulationResult, result_llf: SimulationResult, output_path: str
):
    """
    Generate detailed comparison plot for a single scenario.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    scenario = result_aqps.scenario_name

    # Subplot 1: Fulfillment rates
    metrics = ["Priority\nFulfillment", "Non-Priority\nEnergy"]
    aqps_vals = [
        result_aqps.priority_fulfillment_rate,
        result_aqps.non_priority_energy_delivered_pct,
    ]
    llf_vals = [
        result_llf.priority_fulfillment_rate,
        result_llf.non_priority_energy_delivered_pct,
    ]

    x = np.arange(len(metrics))
    width = 0.35

    ax1.bar(x - width / 2, aqps_vals, width, label="AQPS", color="#2E7D32", alpha=0.8)
    ax1.bar(x + width / 2, llf_vals, width, label="LLF", color="#D32F2F", alpha=0.8)
    ax1.set_ylabel("Percentage (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Fulfillment Metrics", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.set_ylim([0, 110])

    # Subplot 2: Computational time
    ax2.bar(
        ["AQPS", "LLF"],
        [result_aqps.avg_schedule_time_ms, result_llf.avg_schedule_time_ms],
        color=["#6A1B9A", "#00796B"],
        alpha=0.8,
    )
    ax2.set_ylabel("Time (ms)", fontsize=11, fontweight="bold")
    ax2.set_title("Average Scheduling Time", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")

    # Subplot 3: AQPS operational metrics
    ops_metrics = ["Preemptions", "Violations", "Deferrals"]
    ops_vals = [
        result_aqps.total_preemptions,
        result_aqps.total_threshold_violations,
        result_aqps.total_deferrals,
    ]

    colors = ["#E64A19", "#C62828", "#1565C0"]
    ax3.bar(ops_metrics, ops_vals, color=colors, alpha=0.8)
    ax3.set_ylabel("Count", fontsize=11, fontweight="bold")
    ax3.set_title("AQPS Operational Events", fontsize=12, fontweight="bold")
    ax3.grid(True, alpha=0.3, axis="y")

    # Subplot 4: Session breakdown
    session_types = ["Priority", "Non-Priority"]
    counts = [result_aqps.priority_sessions, result_aqps.non_priority_sessions]

    ax4.pie(
        counts,
        labels=session_types,
        autopct="%1.1f%%",
        colors=["#2E7D32", "#1976D2"],
        startangle=90,
    )
    ax4.set_title("Session Distribution", fontsize=12, fontweight="bold")

    fig.suptitle(scenario, fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved scenario detail to {output_path}")


# =============================================================================
# Main Execution
# =============================================================================


def main():
    """
    Run complete Phase 5 simulation study.
    """
    logger.info("=" * 80)
    logger.info("PHASE 5: COMPREHENSIVE SIMULATION STUDIES")
    logger.info("=" * 80)

    # Run all scenarios
    logger.info("\nRunning all scenarios...")
    results = []

    results.append(run_scenario_S1())
    results.append(run_scenario_S2())
    results.append(run_scenario_S3())
    results.append(run_scenario_S4())
    results.append(run_scenario_S5())
    results.append(run_scenario_S6())

    # Generate comparison plots
    logger.info("\n" + "=" * 80)
    logger.info("Generating publication-quality visualizations...")
    logger.info("=" * 80)

    comparison_dir = os.path.join(RESULTS_DIR, "comparison")
    os.makedirs(comparison_dir, exist_ok=True)

    generate_fulfillment_comparison(
        results, os.path.join(comparison_dir, "fulfillment_comparison.png")
    )

    generate_computational_comparison(
        results, os.path.join(comparison_dir, "computational_comparison.png")
    )

    generate_operational_metrics(
        results, os.path.join(comparison_dir, "operational_metrics.png")
    )

    generate_summary_table(results, os.path.join(comparison_dir, "summary_table.png"))

    # Generate individual scenario details
    scenario_names = [
        "S1_baseline",
        "S2_low_priority",
        "S3_high_priority",
        "S4_morning_rush",
        "S5_cloudy_day",
        "S6_peak_stress",
    ]

    for i, (name, (r_aqps, r_llf)) in enumerate(zip(scenario_names, results)):
        scenario_dir = os.path.join(RESULTS_DIR, name)
        os.makedirs(scenario_dir, exist_ok=True)

        generate_scenario_detail_plot(
            r_aqps, r_llf, os.path.join(scenario_dir, "scenario_detail.png")
        )

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("SIMULATION SUMMARY")
    logger.info("=" * 80)

    scenarios = ["S1", "S2", "S3", "S4", "S5", "S6"]
    for scenario, (r_aqps, r_llf) in zip(scenarios, results):
        logger.info(f"\n{scenario} - {r_aqps.scenario_name}")
        logger.info(
            f"  AQPS: Priority {r_aqps.priority_fulfillment_rate:.1f}%, "
            f"Non-Priority {r_aqps.non_priority_energy_delivered_pct:.1f}%, "
            f"Time {r_aqps.avg_schedule_time_ms:.2f}ms"
        )
        logger.info(
            f"  LLF:  Priority {r_llf.priority_fulfillment_rate:.1f}%, "
            f"Non-Priority {r_llf.non_priority_energy_delivered_pct:.1f}%, "
            f"Time {r_llf.avg_schedule_time_ms:.2f}ms"
        )
        logger.info(
            f"  AQPS Events: Preemptions={r_aqps.total_preemptions}, "
            f"Violations={r_aqps.total_threshold_violations}, "
            f"Deferrals={r_aqps.total_deferrals}"
        )

    logger.info("\n" + "=" * 80)
    logger.info("All results saved to: " + RESULTS_DIR)
    logger.info("=" * 80)

    return results


if __name__ == "__main__":
    results = main()
