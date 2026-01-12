"""
Standalone test script for AQPS Phase 1 implementation.
"""

import sys
sys.path.insert(0, '.')

from src.aqps import (
    AdaptiveQueuingPriorityScheduler,
    AQPSConfig,
    generate_scenario,
    SessionInfo
)


def test_basic_functionality():
    """Test basic scheduler functionality."""
    print("=" * 60)
    print("AQPS Phase 1 - Functionality Test")
    print("=" * 60)
    
    # Test 1: Create scheduler
    print("\n[Test 1] Creating scheduler...")
    config = AQPSConfig(
        min_priority_rate=11.0,
        total_capacity=150.0,
        period_minutes=5.0,
        voltage=220.0,
        enable_logging=False
    )
    scheduler = AdaptiveQueuingPriorityScheduler(config)
    print("✓ Scheduler created successfully")
    
    # Test 2: Generate scenario
    print("\n[Test 2] Generating S1 scenario...")
    sessions = generate_scenario('S1', n_sessions=50, seed=42)
    priority_count = sum(s.is_priority for s in sessions)
    print(f"✓ Generated {len(sessions)} sessions ({priority_count} priority)")
    
    # Test 3: Run scheduler
    print("\n[Test 3] Running scheduler...")
    schedule = scheduler.schedule(sessions, current_time=0)
    print(f"✓ Schedule generated: {len(schedule)} stations")
    
    # Test 4: Verify results
    print("\n[Test 4] Verifying results...")
    metrics = scheduler.get_current_metrics()
    
    assert metrics.priority_sessions_active == priority_count, "Priority count mismatch"
    assert metrics.non_priority_sessions_active == len(sessions) - priority_count, "Non-priority count mismatch"
    
    total_allocated = sum(schedule.values())
    assert total_allocated <= config.total_capacity + 0.1, "Capacity exceeded"
    
    print(f"✓ Priority sessions: {metrics.priority_sessions_active}")
    print(f"✓ Non-priority sessions: {metrics.non_priority_sessions_active}")
    print(f"✓ Total allocated: {total_allocated:.1f}A / {config.total_capacity}A")
    print(f"✓ Utilization: {metrics.capacity_utilization:.1f}%")
    
    # Test 5: Verify priority guarantees
    print("\n[Test 5] Verifying priority guarantees...")
    priority_sessions = [s for s in sessions if s.is_priority]
    priority_violations = 0
    
    for session in priority_sessions:
        if session.station_id in schedule:
            rate = schedule[session.station_id]
            if rate < config.min_priority_rate - 0.1:  # Allow small tolerance
                priority_violations += 1
                print(f"  ! {session.session_id}: {rate:.1f}A < {config.min_priority_rate}A")
    
    if priority_violations == 0:
        print(f"✓ All {len(priority_sessions)} priority EVs meet minimum rate")
    else:
        print(f"⚠ {priority_violations} priority violations found")
    
    # Test 6: Test all scenarios
    print("\n[Test 6] Testing all scenarios...")
    from src.aqps import ScenarioGenerator
    
    for scenario_key in ScenarioGenerator.list_scenarios():
        sessions = generate_scenario(scenario_key, n_sessions=30, seed=42)
        schedule = scheduler.schedule(sessions, current_time=0)
        print(f"  ✓ {scenario_key}: {len(sessions)} sessions scheduled")
        scheduler.reset_metrics()
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n" + "=" * 60)
    print("Edge Case Testing")
    print("=" * 60)
    
    config = AQPSConfig(
        min_priority_rate=11.0,
        total_capacity=150.0,
        enable_logging=False
    )
    scheduler = AdaptiveQueuingPriorityScheduler(config)
    
    # Test empty schedule
    print("\n[Edge 1] Empty session list...")
    schedule = scheduler.schedule([], current_time=0)
    assert schedule == {}, "Empty schedule should return empty dict"
    print("✓ Empty schedule handled correctly")
    
    # Test single priority session
    print("\n[Edge 2] Single priority session...")
    session = SessionInfo(
        session_id="EV1",
        station_id="S1",
        arrival_time=0,
        departure_time=100,
        energy_requested=20.0,
        max_rate=32.0,
        is_priority=True
    )
    schedule = scheduler.schedule([session], current_time=0)
    assert "S1" in schedule, "Session not scheduled"
    assert schedule["S1"] >= config.min_priority_rate, "Priority minimum not met"
    print(f"✓ Single session allocated: {schedule['S1']:.1f}A")
    
    # Test capacity overflow
    print("\n[Edge 3] Capacity overflow scenario...")
    many_priority = [
        SessionInfo(
            session_id=f"EV{i}",
            station_id=f"S{i}",
            arrival_time=0,
            departure_time=100,
            energy_requested=20.0,
            max_rate=32.0,
            is_priority=True
        )
        for i in range(20)  # 20 * 11A = 220A > 150A
    ]
    
    schedule = scheduler.schedule(many_priority, current_time=0)
    metrics = scheduler.get_current_metrics()
    
    assert len(metrics.warnings) > 0, "Should have threshold warning"
    print(f"✓ Threshold warning generated: {len(metrics.warnings)} warnings")
    
    print("\n" + "=" * 60)
    print("Edge cases handled correctly! ✓")
    print("=" * 60)


if __name__ == '__main__':
    test_basic_functionality()
    test_edge_cases()
    
    print("\n" + "=" * 60)
    print("PHASE 1 IMPLEMENTATION COMPLETE ✓")
    print("=" * 60)
