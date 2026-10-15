"""
Basic Usage Example for AQPS Scheduler.

This example demonstrates how to:
1. Create a scheduler configuration
2. Generate a test scenario
3. Run the scheduler
4. Examine results and metrics
"""

from src.aqps import (
    AdaptiveQueuingPriorityScheduler,
    AQPSConfig,
    generate_scenario
)


def main():
    print("=" * 60)
    print("AQPS Basic Usage Example")
    print("=" * 60)
    
    # Step 1: Create scheduler configuration
    print("\n1. Creating scheduler configuration...")
    config = AQPSConfig(
        min_priority_rate=11.0,      # Minimum rate for priority EVs (Amps)
        total_capacity=150.0,         # Total charging capacity (Amps)
        period_minutes=5.0,           # 5-minute periods
        voltage=220.0,                # Network voltage
        enable_logging=True           # Enable detailed logging
    )
    print(f"   Config: {config.min_priority_rate}A min, {config.total_capacity}A total")
    
    # Step 2: Create scheduler
    print("\n2. Initializing AQPS scheduler...")
    scheduler = AdaptiveQueuingPriorityScheduler(config)
    print(f"   Scheduler: {scheduler}")
    
    # Step 3: Generate test scenario
    print("\n3. Generating S1 (Baseline) scenario...")
    sessions = generate_scenario('S1', n_sessions=50, seed=42)
    priority_count = sum(s.is_priority for s in sessions)
    print(f"   Generated {len(sessions)} sessions ({priority_count} priority)")
    
    # Step 4: Run scheduler for first period
    print("\n4. Running scheduler at t=0...")
    schedule = scheduler.schedule(sessions, current_time=0)
    
    # Step 5: Examine results
    print("\n5. Schedule Results:")
    print(f"   Total stations: {len(schedule)}")
    print(f"   Total allocated: {sum(schedule.values()):.1f}A")
    
    # Show first 5 allocations
    print("\n   First 5 allocations:")
    for i, (station_id, rate) in enumerate(list(schedule.items())[:5]):
        session = next(s for s in sessions if s.station_id == station_id)
        priority_tag = "[P]" if session.is_priority else "[N]"
        print(f"     {station_id}: {rate:.1f}A {priority_tag}")
    
    # Step 6: Examine metrics
    print("\n6. Metrics:")
    metrics = scheduler.get_current_metrics()
    print(f"   Priority sessions active: {metrics.priority_sessions_active}")
    print(f"   Non-priority sessions active: {metrics.non_priority_sessions_active}")
    print(f"   Priority capacity allocated: {metrics.priority_allocated_capacity:.1f}A")
    print(f"   Total capacity allocated: {metrics.total_allocated_capacity:.1f}A")
    print(f"   Capacity utilization: {metrics.capacity_utilization:.1f}%")
    
    if metrics.warnings:
        print(f"\n   Warnings: {len(metrics.warnings)}")
        for warning in metrics.warnings:
            print(f"     - {warning}")
    else:
        print("\n   No warnings generated")
    
    # Step 7: Priority vs non-priority comparison
    print("\n7. Priority vs Non-Priority Rates:")
    priority_rates = [
        schedule[s.station_id] for s in sessions if s.is_priority and s.station_id in schedule
    ]
    non_priority_rates = [
        schedule[s.station_id] for s in sessions if not s.is_priority and s.station_id in schedule
    ]
    
    if priority_rates:
        print(f"   Priority avg rate: {sum(priority_rates)/len(priority_rates):.1f}A")
        print(f"   Priority min rate: {min(priority_rates):.1f}A")
        print(f"   Priority max rate: {max(priority_rates):.1f}A")
    
    if non_priority_rates:
        print(f"   Non-priority avg rate: {sum(non_priority_rates)/len(non_priority_rates):.1f}A")
        print(f"   Non-priority min rate: {min(non_priority_rates):.1f}A")
        print(f"   Non-priority max rate: {max(non_priority_rates):.1f}A")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
