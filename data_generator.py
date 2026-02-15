import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta

def generate_scenarios(selected_archetype=None):
    """
    Generates deterministic simulation scenarios for the AQPC framework.
    :param selected_archetype: 'Light', 'Medium', or None (defaults to Mix)
    """
    
    # 1. Define Fleet Archetypes (Heavy removed)
    archetypes = {
        'Light': {'cap': 40, 'p_max': 11, 'eff': 0.95},
        'Medium': {'cap': 75, 'p_max': 22, 'eff': 0.92}
    }

    # Helper function to get archetype data based on selection or mix
    def get_arc_props(index):
        if selected_archetype and selected_archetype in archetypes:
            name = selected_archetype
        else:
            # Cyclic mix between Light and Medium
            keys = list(archetypes.keys())
            name = keys[index % len(keys)]
        
        props = archetypes[name].copy()
        props['name'] = name
        return props
    
    # 2. Environmental Forecast Profiles (24 hours, 15-min steps)
    time_steps = pd.date_range("2026-02-12 00:00", periods=96, freq='15min')
    
    # ToU Tariff (Victorian Default Market Offer approx)
    def get_tou(dt):
        hour = dt.hour
        if 15 <= hour < 21: return 0.35 # Peak
        if 7 <= hour < 15 or 21 <= hour < 22: return 0.22 # Shoulder
        return 0.15 # Off-Peak
    
    tou_profile = [get_tou(t) for t in time_steps]
    
    # PV Profile (Bell Curve - 12kW peak)
    pv_peak = 12.0
    pv_profile = [pv_peak * max(0, np.sin(np.pi * (t.hour + t.minute/60 - 6) / 12)) for t in time_steps]
    
    # Base Load (5kW idle + 5kW office hours)
    base_load = [5.0 + (5.0 if 9 <= t.hour < 17 else 0.0) for t in time_steps]

    # 3. Scenario Logic
    scenarios = {}

    # --- Scenario A: Baseline ---
    ev_list_a = []
    windows = [
        {"start": 9, "end": 11, "count": 10}, 
        {"start": 15, "end": 16, "count": 6}
    ]
    
    global_id = 0
    for win in windows:
        win_times = time_steps[(time_steps.hour >= win["start"]) & (time_steps.hour < win["end"])]
        for i in range(win["count"]):
            arrival = win_times[i % len(win_times)]
            departure = arrival + timedelta(hours=4)
            arc = get_arc_props(global_id)
            
            ev_list_a.append({
                'id': f'EV_A_{global_id}',
                'arrival': arrival.isoformat(),
                'departure': departure.isoformat(),
                'req_energy': arc['cap'] * 0.7,
                'init_energy': arc['cap'] * 0.2,
                'p_max': arc['p_max'],
                'eff': arc['eff'],
                'archetype': arc['name'],
                'priority': 'High' if i < (win["count"] // 2) else 'Low'
            })
            global_id += 1
    scenarios['Scenario_A_Baseline'] = ev_list_a

    # --- Scenario B: Mid-Day Rush ---
    ev_list_b = []
    for i in range(15):
        is_rush = 5 <= i <= 14
        arrival = time_steps[48] if is_rush else time_steps[32 + i]
        departure = arrival + (timedelta(hours=2) if is_rush else timedelta(hours=6))
        arc = get_arc_props(i)
        
        ev_list_b.append({
            'id': f'EV_B_{i}',
            'arrival': arrival.isoformat(),
            'departure': departure.isoformat(),
            'req_energy': arc['cap'] * 0.6,
            'init_energy': arc['cap'] * 0.1,
            'p_max': arc['p_max'],
            'eff': arc['eff'],
            'archetype': arc['name'],
            'priority': 'High' if is_rush else 'Low'
        })
    scenarios['Scenario_B_MidDayRush'] = ev_list_b

    # --- Scenario C: Late Return Conflict ---
    ev_list_c = []
    for i in range(8):
        arrival = time_steps[56] # 2 PM
        arc = get_arc_props(i)
        # Dwell time calc based on arc specifics
        dwell_hours = (arc['cap'] * 0.6) / (arc['p_max'] * arc['eff']) * 1.2
        departure = arrival + timedelta(hours=dwell_hours)
        
        ev_list_c.append({
            'id': f'EV_C_{i}',
            'arrival': arrival.isoformat(),
            'departure': departure.isoformat(),
            'req_energy': arc['cap'] * 0.7,
            'init_energy': arc['cap'] * 0.1,
            'p_max': arc['p_max'],
            'eff': arc['eff'],
            'archetype': arc['name'],
            'priority': 'High'
        })
    scenarios['Scenario_C_LateReturn'] = ev_list_c

    # --- Final Compilation ---
    output = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'selected_archetype': selected_archetype if selected_archetype else "Mix",
            'step_delta': 0.25,
            'horizon': 96
        },
        'environment': {
            'tou_tariff': tou_profile,
            'pv_forecast': pv_profile,
            'base_load': base_load
        },
        'scenarios': scenarios
    }

    filename = f"simulation_data_{selected_archetype if selected_archetype else 'mix'}.json"
    with open(filename, 'w') as f:
        json.dump(output, f, indent=4)
    
    print(f"Scenarios for {selected_archetype if selected_archetype else 'Mix'} generated in {filename}")

if __name__ == "__main__":
    generate_scenarios()