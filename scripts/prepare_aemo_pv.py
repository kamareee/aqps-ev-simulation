"""
Prepare real AEMO VIC1 rooftop-PV forecast/actual profiles for the 20 kW depot array.

Inputs : data/aemo_raw/*.zip  (from scripts/download_aemo_pv.py)
Outputs: data/aemo_vic1_daily_errors.csv   — every candidate weekday with its forecast error
         data/aemo_vic1_selected_days.json — 6 selected days, 120 x 15-min steps each

Method (all fixed before any simulation is run):
  * Region VIC1. Time base: NEM market time (AEST, UTC+10, no daylight saving).
  * Actual   = ROOFTOP_PV_ACTUAL.POWER; TYPE preference MEASUREMENT > SATELLITE > DAILY
               (first type present is used and recorded).
  * Forecast = ROOFTOP_PV_FORECAST.POWERMEAN from the latest run with
               VERSION_DATETIME <= (D-1) 12:00 AEST  (day-ahead vintage).
  * AEMO intervals are 30-min "time ending"; each value is HELD for the two 15-min steps
    inside its interval (no interpolation).
  * Scaling: kW = MW / max(VIC1 actual MW over the whole downloaded period) * 20 kW.
  * Daily error: nMAE = mean |forecast - actual| over 06:00-20:00 of day D, as % of 20 kW.
  * Day selection: weekdays ranked by nMAE; take the days at the
    10, 25, 50, 75, 90, 97.5th percentiles. The 50th-percentile day is the "reference day"
    used by E1-E3 and E5-E7.
  * Window per day: 120 steps, 06:00 day D -> 12:00 day D+1 (D+1 actual from AEMO, D+1
    forecast from the same day-ahead run).
Fails loudly if inputs are missing. There is no synthetic fallback.
"""

import argparse
import csv
import glob
import io
import json
import os
import sys
import zipfile
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np

REGION = "VIC1"
SITE_KW = 20.0
TYPE_PREF = ["MEASUREMENT", "SATELLITE", "DAILY"]
PERCENTILES = [10, 25, 50, 75, 90, 97.5]
REFERENCE_PCT = 50
N_STEPS = 120
START_HOUR = 6
CUTOFF_HOUR = 12  # day-ahead vintage: latest run issued at or before 12:00 on D-1

OUT_ERRORS = os.path.join("data", "aemo_vic1_daily_errors.csv")
OUT_DAYS = os.path.join("data", "aemo_vic1_selected_days.json")


def _parse_dt(s: str) -> datetime:
    s = s.strip().strip('"')
    for fmt in ("%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    raise ValueError(f"Unrecognised AEMO datetime: {s!r}")


def iter_mms_rows(zip_path: str):
    """Yield dict rows from an MMS 'I'/'D' formatted CSV inside a zip (streaming)."""
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.upper().endswith(".CSV")]
        if not names:
            raise RuntimeError(f"No CSV inside {zip_path}")
        for name in names:
            with zf.open(name) as raw:
                header = None
                for rec in csv.reader(io.TextIOWrapper(raw, encoding="utf-8", errors="replace")):
                    if not rec:
                        continue
                    tag = rec[0]
                    if tag == "I":
                        header = [c.strip().upper() for c in rec[4:]]
                    elif tag == "D" and header is not None:
                        yield dict(zip(header, rec[4:]))


def require(cols, row, table):
    missing = [c for c in cols if c not in row]
    if missing:
        raise RuntimeError(f"{table}: missing columns {missing}; got {sorted(row.keys())}")


def load_actual(raw_dir: str):
    files = sorted(glob.glob(os.path.join(raw_dir, "*ROOFTOP_PV_ACTUAL*.zip")))
    if not files:
        sys.exit(f"No ROOFTOP_PV_ACTUAL zips in {raw_dir}. Run scripts/download_aemo_pv.py first.")
    by_type = defaultdict(dict)  # type -> {interval_end: MW}
    checked = False
    for fp in files:
        for r in iter_mms_rows(fp):
            if not checked:
                require(["INTERVAL_DATETIME", "TYPE", "REGIONID", "POWER"], r, "ROOFTOP_PV_ACTUAL")
                checked = True
            if r["REGIONID"].strip() != REGION or r["POWER"].strip() == "":
                continue
            by_type[r["TYPE"].strip().upper()][_parse_dt(r["INTERVAL_DATETIME"])] = float(r["POWER"])
    for t in TYPE_PREF:
        if by_type.get(t):
            return t, by_type[t]
    sys.exit(f"No usable ROOFTOP_PV_ACTUAL rows for {REGION}; types found: {list(by_type)}")


def load_forecast(raw_dir: str):
    """Keep only runs issued between 10:00 and 12:00 (enough to find the latest <= 12:00)."""
    files = sorted(glob.glob(os.path.join(raw_dir, "*ROOFTOP_PV_FORECAST*.zip")))
    if not files:
        sys.exit(f"No ROOFTOP_PV_FORECAST zips in {raw_dir}. Run scripts/download_aemo_pv.py first.")
    runs = defaultdict(dict)  # version_dt -> {interval_end: MW}
    checked = False
    for fp in files:
        print(f"  reading {os.path.basename(fp)}")
        for r in iter_mms_rows(fp):
            if not checked:
                require(["VERSION_DATETIME", "REGIONID", "INTERVAL_DATETIME", "POWERMEAN"], r,
                        "ROOFTOP_PV_FORECAST")
                checked = True
            if r["REGIONID"].strip() != REGION or r["POWERMEAN"].strip() == "":
                continue
            v = _parse_dt(r["VERSION_DATETIME"])
            if not (CUTOFF_HOUR - 2 <= v.hour + v.minute / 60.0 <= CUTOFF_HOUR):
                continue
            runs[v][_parse_dt(r["INTERVAL_DATETIME"])] = float(r["POWERMEAN"])
    if not runs:
        sys.exit("No forecast runs found in the 10:00-12:00 issue window.")
    return runs


def step_interval_ends(day: datetime):
    """For each 15-min step from day 06:00 (120 steps), the AEMO 30-min interval-ending time."""
    ends = []
    t0 = day.replace(hour=START_HOUR, minute=0, second=0, microsecond=0)
    for s in range(N_STEPS):
        tau = t0 + timedelta(minutes=15 * s)
        minute_of_day = tau.hour * 60 + tau.minute
        end_min = (minute_of_day // 30) * 30 + 30
        ends.append(tau.replace(hour=0, minute=0) + timedelta(minutes=end_min))
    return ends


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join("data", "aemo_raw"))
    args = ap.parse_args()

    print("Loading ROOFTOP_PV_ACTUAL ...")
    a_type, actual = load_actual(args.raw)
    print(f"  {REGION} actual TYPE used: {a_type}; {len(actual)} intervals")
    print("Loading ROOFTOP_PV_FORECAST (day-ahead window) ...")
    runs = load_forecast(args.raw)
    run_times = sorted(runs)
    print(f"  {len(run_times)} candidate runs")

    norm_mw = max(actual.values())
    scale = SITE_KW / norm_mw

    days = sorted({k.date() for k in actual})
    rows, profiles = [], {}
    for d in days:
        D = datetime(d.year, d.month, d.day)
        if D.weekday() >= 5:
            continue
        cutoff = D - timedelta(days=1) + timedelta(hours=CUTOFF_HOUR)
        eligible = [v for v in run_times if v <= cutoff and v.date() == cutoff.date()]
        if not eligible:
            continue
        v = eligible[-1]
        ends = step_interval_ends(D)
        if not all(e in actual for e in ends) or not all(e in runs[v] for e in ends):
            continue  # incomplete coverage (archive edge, gaps) -> not a candidate
        act = np.array([actual[e] for e in ends]) * scale
        fc = np.array([runs[v][e] for e in ends]) * scale
        act = np.clip(act, 0.0, None)
        fc = np.clip(fc, 0.0, None)
        day_mask = np.arange(N_STEPS) < (20 - START_HOUR) * 4  # 06:00-20:00 of day D
        nmae = float(np.mean(np.abs(fc[day_mask] - act[day_mask])) / SITE_KW * 100.0)
        rows.append({
            "date": d.isoformat(), "version_datetime": v.isoformat(sep=" "),
            "nmae_pct": round(nmae, 3),
            "energy_forecast_kwh_dayD": round(float(fc[day_mask].sum() * 0.25), 2),
            "energy_actual_kwh_dayD": round(float(act[day_mask].sum() * 0.25), 2),
        })
        profiles[d.isoformat()] = (fc, act, v)

    if len(rows) < len(PERCENTILES):
        sys.exit(f"Only {len(rows)} complete weekdays found — check the downloaded months.")

    rows.sort(key=lambda r: r["nmae_pct"])
    n = len(rows)
    picked = []
    for p in PERCENTILES:
        idx = int(round(p / 100.0 * (n - 1)))
        while rows[idx]["date"] in {x["date"] for x in picked} and idx < n - 1:
            idx += 1
        r = dict(rows[idx])
        r["percentile"] = p
        r["rank"] = idx + 1
        picked.append(r)
        rows[idx]["selected_percentile"] = p

    os.makedirs("data", exist_ok=True)
    with open(OUT_ERRORS, "w", newline="") as f:
        cols = ["date", "version_datetime", "nmae_pct", "energy_forecast_kwh_dayD",
                "energy_actual_kwh_dayD", "selected_percentile"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})

    out = {
        "metadata": {
            "source": "AEMO NEMWeb MMSDM: ROOFTOP_PV_ACTUAL (POWER) and ROOFTOP_PV_FORECAST "
                      "(POWERMEAN), region VIC1",
            "actual_type": a_type,
            "forecast_vintage": f"latest VERSION_DATETIME <= D-1 {CUTOFF_HOUR:02d}:00 AEST",
            "time_base": "AEST (NEM market time, UTC+10, no daylight saving)",
            "interval_handling": "30-min time-ending values held for both 15-min steps",
            "scaling": f"MW / {norm_mw:.3f} MW (max VIC1 actual over period) * {SITE_KW} kW",
            "normalisation_max_mw": norm_mw,
            "error_metric": "nMAE over 06:00-20:00 of day D, % of 20 kW",
            "selection": f"weekdays ranked by nMAE; percentiles {PERCENTILES}",
            "reference_percentile": REFERENCE_PCT,
            "n_candidate_weekdays": n,
            "window": "120 x 15-min steps, 06:00 D -> 12:00 D+1",
            "limitations": "Regional aggregate (smooths single-site cloud transients, "
                           "understates site-level forecast error); POWER is an AEMO estimate, "
                           "not metered output.",
            "raw_checksums": "data/aemo_raw/checksums.csv",
        },
        "days": [],
    }
    for r in picked:
        fc, act, v = profiles[r["date"]]
        out["days"].append({
            **r,
            "reference": r["percentile"] == REFERENCE_PCT,
            "pv_forecast_kw": [round(float(x), 4) for x in fc],
            "pv_actual_kw": [round(float(x), 4) for x in act],
        })
    with open(OUT_DAYS, "w") as f:
        json.dump(out, f, indent=1)
    print(f"Saved {OUT_ERRORS} ({n} weekdays) and {OUT_DAYS}:")
    for r in picked:
        print(f"  p{r['percentile']:>5}: {r['date']}  nMAE {r['nmae_pct']:.2f}%  run {r['version_datetime']}")


if __name__ == "__main__":
    main()
