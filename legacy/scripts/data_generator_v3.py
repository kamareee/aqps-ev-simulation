# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring

import numpy as np
import json
from datetime import datetime, timedelta


def generate_scenarios(output_file="simulation_data_v2.json"):
    """
    Generates adversarial bottleneck scenarios to expose LLF blind spots.

    Scenarios:
        S1 - Baseline: Mild conditions, both AQPS and LLF cope.
        S3 - LLF Trap: Early non-priority "decoys" with huge energy + tight
             deadlines hog chargers. Priority EVs arrive mid-day and starve
             under LLF. AQPS preempts via Layer 1.
        S6 - Peak Stress: Extreme non-priority EVs clustered just before PM
             peak block chargers when priority EVs arrive during TOU peak.

    Each scenario generated for both fleet sizes:
        - 45 EVs / 16 EVSEs / 320A transformer
        - 90 EVs / 35 EVSEs / 700A transformer
    """

    # ── 1. Time Configuration ────────────────────────────────────────────
    # 96 steps × 15 min = 24 h, starting 06:00
    start_time = datetime.strptime("2026-02-12 06:00", "%Y-%m-%d %H:%M")
    time_steps = [start_time + timedelta(minutes=15 * i) for i in range(96)]

    # Helper: hour → step index (relative to 06:00 start)
    def hour_to_idx(h):
        return int((h - 6) * 4)  # 4 steps per hour

    # ── 2. Archetypes ────────────────────────────────────────────────────
    archetypes = {
        "Light": {"cap": 40, "p_max": 11, "eff": 0.95},
        "Medium": {"cap": 75, "p_max": 22, "eff": 0.92},
    }

    # ── 3. Infrastructure Configs ────────────────────────────────────────
    infra = {
        45: {"n_evse": 16, "transformer_A": 320, "voltage": 220},
        90: {"n_evse": 35, "transformer_A": 700, "voltage": 220},
    }

    # ── 4. Environmental Forecasts ───────────────────────────────────────
    def get_tou(dt):
        h = dt.hour
        if 15 <= h < 21:
            return 0.35  # Peak
        if 7 <= h < 15 or 21 <= h < 22:
            return 0.22  # Shoulder
        return 0.15  # Off-Peak

    tou_profile = [get_tou(t) for t in time_steps]

    pv_profile = []
    for t in time_steps:
        if 7 <= t.hour < 19:
            x = (t.hour + t.minute / 60.0 - 7) / 12.0
            pv_profile.append(max(0.0, 50.0 * np.sin(x * np.pi)))
        else:
            pv_profile.append(0.0)

    base_load = [10.0 + (15.0 if 8 <= t.hour < 18 else 0.0) for t in time_steps]

    # ── 5. EV Generation Helpers ─────────────────────────────────────────
    ev_counter = 0

    def make_ev(archetype, arr_idx, dep_idx, energy_frac, is_priority):
        """Create a single EV dict."""
        nonlocal ev_counter
        arc = archetypes[archetype]
        # Convert step index to ISO datetime string (06:00 start, 15 min steps)
        arr_clipped = int(np.clip(arr_idx, 0, 95))
        dep_clipped = int(np.clip(dep_idx, 0, 95))
        arr_dt = start_time + timedelta(minutes=15 * arr_clipped)
        dep_dt = start_time + timedelta(minutes=15 * dep_clipped)
        ev = {
            "id": f"EV_{ev_counter:03d}",
            "arrival": arr_dt.isoformat(),
            "departure": dep_dt.isoformat(),
            "arrival_idx": arr_clipped,
            "departure_idx": dep_clipped,
            "req_energy": round(arc["cap"] * energy_frac, 2),
            "p_max": arc["p_max"],
            "battery_cap_kwh": arc["cap"],
            "priority": "High" if is_priority else "Low",
            "archetype": archetype,
        }
        ev_counter += 1
        return ev

    def make_batch(
        n, archetype, arr_range, dwell_range, energy_range, is_priority, rng
    ):
        """Generate a batch of EVs with given parameter ranges."""
        evs = []
        for _ in range(n):
            arr = rng.integers(arr_range[0], arr_range[1] + 1)
            dwell = rng.integers(dwell_range[0], dwell_range[1] + 1)
            dep = min(95, arr + dwell)
            efrac = rng.uniform(energy_range[0], energy_range[1])
            evs.append(make_ev(archetype, arr, dep, efrac, is_priority))
        return evs

    def finalize(ev_list):
        """Sort by arrival, add laxity metadata for analysis."""
        ev_list.sort(key=lambda x: x["arrival_idx"])
        for ev in ev_list:
            avail_steps = ev["departure_idx"] - ev["arrival_idx"]
            steps_needed = ev["req_energy"] / (ev["p_max"] * 0.25)
            ev["laxity_steps"] = round(avail_steps - steps_needed, 1)
            ev["avail_steps"] = avail_steps
        return ev_list

    # ── 6. Scenario Generators ───────────────────────────────────────────

    def gen_s1(n_evs, seed=100):
        """S1 - Baseline: Mild, uniform arrivals, moderate energy, relaxed deadlines.
        Both AQPS and LLF should handle this well.
        27% priority.
        """
        nonlocal ev_counter
        ev_counter = 0
        rng = np.random.default_rng(seed)

        n_pri = int(n_evs * 0.27)
        n_nonpri = n_evs - n_pri

        evs = []
        # Non-priority: uniform 06:00-16:00, moderate energy, long dwell
        evs += make_batch(
            n=n_nonpri,
            archetype="Light" if rng.random() > 0.5 else "Medium",
            arr_range=(hour_to_idx(6), hour_to_idx(16)),
            dwell_range=(20, 44),  # 5-11 hours
            energy_range=(0.25, 0.55),  # moderate
            is_priority=False,
            rng=rng,
        )
        # Fix: alternate archetypes properly
        for i, ev in enumerate(evs):
            arc_name = "Light" if i % 2 == 0 else "Medium"
            arc = archetypes[arc_name]
            ev["archetype"] = arc_name
            ev["p_max"] = arc["p_max"]
            ev["battery_cap_kwh"] = arc["cap"]
            ev["req_energy"] = round(arc["cap"] * rng.uniform(0.25, 0.55), 2)

        # Priority: uniform 07:00-15:00, moderate energy, generous dwell
        pri_evs = make_batch(
            n=n_pri,
            archetype="Medium",
            arr_range=(hour_to_idx(7), hour_to_idx(15)),
            dwell_range=(20, 40),  # 5-10 hours
            energy_range=(0.30, 0.55),
            is_priority=True,
            rng=rng,
        )
        # Alternate archetypes for priority too
        for i, ev in enumerate(pri_evs):
            arc_name = "Light" if i % 2 == 0 else "Medium"
            arc = archetypes[arc_name]
            ev["archetype"] = arc_name
            ev["p_max"] = arc["p_max"]
            ev["battery_cap_kwh"] = arc["cap"]
            ev["req_energy"] = round(arc["cap"] * rng.uniform(0.30, 0.55), 2)

        evs += pri_evs
        return finalize(evs)

    def gen_s3(n_evs, seed=200):
        """S3 - LLF Trap: 50% priority.

        Non-priority DECOYS:
          - Arrive early (06:00-08:00)
          - Huge energy need (70-90% of Medium battery = 52-67 kWh)
          - TIGHT deadlines (4-6 hours dwell → near-zero laxity)
          - LLF sees low laxity → locks onto them

        Priority EVs:
          - Arrive mid-day (10:00-14:00)
          - Moderate energy (40-60% of cap)
          - Moderate dwell (5-8 hours)
          - LLF refuses to cut off low-laxity decoys → priority starves
          - AQPS preempts decoys via Layer 1
        """
        nonlocal ev_counter
        ev_counter = 0
        rng = np.random.default_rng(seed)

        n_pri = int(n_evs * 0.50)
        n_nonpri = n_evs - n_pri

        evs = []

        # ── Non-Priority DECOYS: early, hungry, tight deadline ───────
        n_decoys = int(n_nonpri * 0.70)  # 70% of non-pri are decoys
        n_normal_np = n_nonpri - n_decoys  # 30% normal non-priority

        # Decoys: Medium archetype, early arrival, high energy, short dwell
        evs += make_batch(
            n=n_decoys,
            archetype="Medium",
            arr_range=(hour_to_idx(6), hour_to_idx(8)),  # 06:00-08:00
            dwell_range=(16, 24),  # 4-6 hours (tight!)
            energy_range=(0.70, 0.90),  # 52-67 kWh
            is_priority=False,
            rng=rng,
        )

        # Normal non-priority: filler, spread out, relaxed
        evs += make_batch(
            n=n_normal_np,
            archetype="Light",
            arr_range=(hour_to_idx(6), hour_to_idx(14)),
            dwell_range=(24, 40),
            energy_range=(0.25, 0.50),
            is_priority=False,
            rng=rng,
        )

        # ── Priority EVs: mid-day arrival, moderate needs ────────────
        evs += make_batch(
            n=n_pri,
            archetype="Medium",
            arr_range=(hour_to_idx(10), hour_to_idx(14)),  # 10:00-14:00
            dwell_range=(20, 32),  # 5-8 hours
            energy_range=(0.40, 0.60),  # 30-45 kWh
            is_priority=True,
            rng=rng,
        )

        return finalize(evs)

    def gen_s6(n_evs, seed=300):
        """S6 - Peak Stress: 50% priority, PM clustering.

        Non-priority BLOCKERS:
          - Cluster just BEFORE PM peak (13:00-15:00)
          - Extreme energy (75-90% of Medium = 56-67 kWh)
          - Tight deadlines (4-6 hours → depart 17:00-21:00)
          - Perfectly timed to hog chargers during TOU peak

        Priority EVs:
          - Arrive during peak (15:00-18:00)
          - Moderate-high energy (50-70% of cap)
          - Moderate dwell (4-7 hours)
          - Must charge during peak pricing window
          - LLF blocked by low-laxity non-priority → priority starves
        """
        nonlocal ev_counter
        ev_counter = 0
        rng = np.random.default_rng(seed)

        n_pri = int(n_evs * 0.50)
        n_nonpri = n_evs - n_pri

        evs = []

        # ── Non-Priority BLOCKERS: pre-peak cluster ──────────────────
        n_blockers = int(n_nonpri * 0.75)  # 75% are blockers
        n_normal_np = n_nonpri - n_blockers

        # Blockers: arrive 13:00-15:00, extreme energy, tight
        evs += make_batch(
            n=n_blockers,
            archetype="Medium",
            arr_range=(hour_to_idx(13), hour_to_idx(15)),  # just before peak
            dwell_range=(16, 24),  # 4-6 hours
            energy_range=(0.75, 0.90),  # 56-67 kWh
            is_priority=False,
            rng=rng,
        )

        # Normal non-priority: morning arrivals, relaxed
        evs += make_batch(
            n=n_normal_np,
            archetype="Light",
            arr_range=(hour_to_idx(6), hour_to_idx(12)),
            dwell_range=(24, 40),
            energy_range=(0.25, 0.45),
            is_priority=False,
            rng=rng,
        )

        # ── Priority EVs: arrive during peak ─────────────────────────
        evs += make_batch(
            n=n_pri,
            archetype="Medium",
            arr_range=(hour_to_idx(15), hour_to_idx(18)),  # peak window
            dwell_range=(16, 28),  # 4-7 hours
            energy_range=(0.50, 0.70),  # 37-52 kWh
            is_priority=True,
            rng=rng,
        )

        return finalize(evs)

    # ── 7. Generate All Variants ─────────────────────────────────────────
    generators = {
        "S1_Baseline": gen_s1,
        "S3_HighPriority": gen_s3,
        "S6_PeakStress": gen_s6,
    }

    fleet_sizes = [45, 90]

    scenarios = {}
    infra_configs = {}

    for size in fleet_sizes:
        for sname, gen_fn in generators.items():
            key = (
                f"{sname}"  # same key for both fleet sizes (split into separate files)
            )
            scenarios.setdefault(size, {})[key] = gen_fn(
                n_evs=size, seed=hash((sname, size)) % 10000
            )

            # Infrastructure for this variant
            cfg = infra[size]
            infra_configs[size] = {
                "n_evse": cfg["n_evse"],
                "transformer_capacity_A": cfg["transformer_A"],
                "transformer_capacity_kW": round(
                    cfg["transformer_A"] * cfg["voltage"] / 1000, 1
                ),
                "voltage_V": cfg["voltage"],
                "max_rate_per_evse_A": 32,
            }

    # ── 8. Summary Statistics ────────────────────────────────────────────
    def compute_summary(evs):
        pri = [e for e in evs if e["priority"] == "High"]
        npri = [e for e in evs if e["priority"] == "Low"]
        return {
            "total_evs": len(evs),
            "priority_evs": len(pri),
            "non_priority_evs": len(npri),
            "priority_pct": round(len(pri) / len(evs) * 100, 1),
            "total_energy_kwh": round(sum(e["req_energy"] for e in evs), 1),
            "priority_energy_kwh": round(sum(e["req_energy"] for e in pri), 1),
            "non_priority_energy_kwh": round(sum(e["req_energy"] for e in npri), 1),
            "avg_laxity_priority": (
                round(np.mean([e["laxity_steps"] for e in pri]), 1) if pri else None
            ),
            "avg_laxity_non_priority": (
                round(np.mean([e["laxity_steps"] for e in npri]), 1) if npri else None
            ),
            "min_laxity_non_priority": (
                round(min(e["laxity_steps"] for e in npri), 1) if npri else None
            ),
        }

    # ── 9. Output: one JSON per fleet size ───────────────────────────────
    environment = {
        "tou_tariff": tou_profile,
        "pv_forecast": pv_profile,
        "base_load": base_load,
        "bess": [0.0] * 96,
    }

    metadata = {
        "step_minutes": 15,
        "horizon": 96,
        "start_time": "2026-02-12 06:00",
        "description": "Adversarial bottleneck scenarios for AQPS vs LLF comparison",
        "fleet_sizes": fleet_sizes,
        "scenario_types": ["S1_Baseline", "S3_HighPriority", "S6_PeakStress"],
    }

    all_summaries = {}

    for size in fleet_sizes:
        sc = scenarios[size]  # dict: {S1_Baseline: [...], S3_HighPriority: [...], ...}
        summ = {k: compute_summary(v) for k, v in sc.items()}
        all_summaries[size] = summ

        file_data = {
            "metadata": metadata,
            "infrastructure": infra_configs[size],
            "environment": environment,
            "scenarios": sc,
            "summary": summ,
        }

        fname = f"simulation_data_{size}EVs.json"
        with open(fname, "w") as f:
            json.dump(file_data, f, indent=2)
        print(f"Generated {fname}")

    # Print summary table
    print(f"\n{'='*75}")
    print(
        f"{'Scenario':<20} {'EVs':>4} {'Pri':>4} {'NPri':>5} "
        f"{'TotE(kWh)':>10} {'AvgLax_P':>9} {'AvgLax_NP':>10} {'MinLax_NP':>10}"
    )
    print(f"{'-'*75}")
    for size in fleet_sizes:
        for key, s in all_summaries[size].items():
            label = f"{key}({size})"
            print(
                f"{label:<20} {s['total_evs']:>4} {s['priority_evs']:>4} "
                f"{s['non_priority_evs']:>5} {s['total_energy_kwh']:>10.1f} "
                f"{s['avg_laxity_priority'] or 0:>9.1f} "
                f"{s['avg_laxity_non_priority'] or 0:>10.1f} "
                f"{s['min_laxity_non_priority'] or 0:>10.1f}"
            )
    print(f"{'='*75}")

    return scenarios


if __name__ == "__main__":
    generate_scenarios()
