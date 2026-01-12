"""
Scenario Comparison Example for AQPS.

This example compares scheduler performance across all 6 scenarios (S1-S6).
"""

from src.aqps import (
    AdaptiveQueuingPriorityScheduler,
    AQPSConfig,
    ScenarioGenerator
)


def run_scenario(scenario_key: str, scheduler: AdaptiveQueuingPriorityScheduler, n_sessions: int = 100):
    """Run scheduler on a scenario and collect metrics."""
    generator = ScenarioGenerator()
    sessions = generator.generate(scenario_key, n_sessions=n_sessions, seed=42)
    
    schedule = scheduler.schedule(sessions, current_time=0)
    metrics = scheduler.get_current_metrics()
    
    return {
        'scenario': scenario_key,
        'sessions': len(sessions),
        'priority_count': metrics.priority_sessions_active,
        'non_priority_count': metrics.non_priority_sessions_active,
        'total_capacity': metrics.total_allocated_capacity,
        'priority_capacity': metrics.priority_allocated_capacity,
        'utilization': metrics.capacity_utilization,
        'warnings': len(metrics.warnings),
        'has_threshold_exceeded': any('threshold exceeded' in w.lower() for w in metrics.warnings)
    }


def main():
    print("=" * 70)
    print("AQPS Scenario Comparison")
    print("=" * 70)
    
    # Create scheduler
    config = AQPSConfig(
        min_priority_rate=11.0,
        total_capacity=150.0,
        period_minutes=5.0,
        voltage=220.0,
        enable_logging=False  # Disable for cleaner output
    )
    scheduler = AdaptiveQueuingPriorityScheduler(config)
    
    # Run all scenarios
    results = []
    scenarios = ScenarioGenerator.list_scenarios()
    
    print("\nRunning scenarios...")
    for scenario_key in scenarios:
        print(f"  {scenario_key}...", end=" ")
        result = run_scenario(scenario_key, scheduler, n_sessions=100)
        results.append(result)
        print("Done")
        scheduler.reset_metrics()  # Reset for next scenario
    
    # Print comparison table
    print("\n" + "=" * 70)
    print("Results Summary")
    print("=" * 70)
    print(f"{'Scenario':<12} {'Priority':<10} {'Non-Pri':<10} {'Capacity':<12} {'Util%':<8} {'Warnings':<10}")
    print("-" * 70)
    
    for result in results:
        scenario_info = ScenarioGenerator.get_scenario_info(result['scenario'])
        print(
            f"{result['scenario']:<12} "
            f"{result['priority_count']:<10} "
            f"{result['non_priority_count']:<10} "
            f"{result['total_capacity']:<12.1f} "
            f"{result['utilization']:<8.1f} "
            f"{result['warnings']:<10}"
        )
    
    # Identify scenarios with threshold issues
    print("\n" + "=" * 70)
    print("Threshold Analysis")
    print("=" * 70)
    
    threshold_scenarios = [r for r in results if r['has_threshold_exceeded']]
    if threshold_scenarios:
        print("Scenarios where priority threshold was exceeded:")
        for result in threshold_scenarios:
            scenario_info = ScenarioGenerator.get_scenario_info(result['scenario'])
            print(f"  - {result['scenario']}: {scenario_info['description']}")
            print(f"    Priority EVs: {result['priority_count']}")
            print(f"    Min capacity needed: {result['priority_count'] * 11.0:.1f}A")
            print(f"    Available capacity: {config.total_capacity}A")
    else:
        print("No scenarios exceeded priority threshold!")
    
    # Capacity utilization analysis
    print("\n" + "=" * 70)
    print("Capacity Utilization")
    print("=" * 70)
    
    avg_util = sum(r['utilization'] for r in results) / len(results)
    max_util = max(results, key=lambda r: r['utilization'])
    min_util = min(results, key=lambda r: r['utilization'])
    
    print(f"Average utilization: {avg_util:.1f}%")
    print(f"Highest: {max_util['scenario']} at {max_util['utilization']:.1f}%")
    print(f"Lowest: {min_util['scenario']} at {min_util['utilization']:.1f}%")
    
    print("\n" + "=" * 70)
    print("Comparison completed!")
    print("=" * 70)


if __name__ == '__main__':
    main()
