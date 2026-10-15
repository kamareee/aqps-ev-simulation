"""
Download AEMO rooftop PV actual + forecast monthly archives from NEMWeb (MMSDM).

Run locally (needs internet). Raw zips are NOT committed; their SHA-256 checksums are
(data/aemo_raw/checksums.csv), so anyone can verify the exact inputs.

Source tables (MMS Data Model, package DEMAND_FORECASTS):
  ROOFTOP_PV_ACTUAL   (INTERVAL_DATETIME [time-ending, AEST], TYPE, REGIONID, POWER [MW], QI)
  ROOFTOP_PV_FORECAST (VERSION_DATETIME, REGIONID, INTERVAL_DATETIME, POWERMEAN, POWERPOE50, ...)

Usage:
  python scripts/download_aemo_pv.py --start 2024-07 --end 2025-06
  (forecast archives are fetched from one month earlier so day-ahead runs exist for day 1)

Data: Australian Energy Market Operator (AEMO), NEMWeb MMS Data Model. Check AEMO's
copyright permissions (https://aemo.com.au/privacy-and-legal-notices/copyright-permissions)
and cite AEMO as the source.
"""

import argparse
import csv
import hashlib
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

BASE = "https://nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM"
UA = {"User-Agent": "Mozilla/5.0 (research download; aqps-ev-simulation)"}


def month_range(start: str, end: str):
    y, m = map(int, start.split("-"))
    ye, me = map(int, end.split("-"))
    while (y, m) <= (ye, me):
        yield y, m
        m += 1
        if m == 13:
            y, m = y + 1, 1


def shift_month(ym: str, delta: int) -> str:
    y, m = map(int, ym.split("-"))
    idx = y * 12 + (m - 1) + delta
    return f"{idx // 12:04d}-{idx % 12 + 1:02d}"


def candidate_urls(y: int, m: int, table: str):
    """Yield (filename, url) candidates. NEMWeb changed naming in 2024/25:
    old: PUBLIC_DVD_ROOFTOP_PV_ACTUAL_YYYYMM010000.zip
    new: PUBLIC_ARCHIVE#ROOFTOP_PV_ACTUAL#FILE01#YYYYMM010000.zip (FILE02.. if split)"""
    stamp = f"{y:04d}{m:02d}010000"
    d = f"{BASE}/{y:04d}/MMSDM_{y:04d}_{m:02d}/MMSDM_Historical_Data_SQLLoader/DATA"
    yield "old", f"PUBLIC_DVD_{table}_{stamp}.zip", f"{d}/PUBLIC_DVD_{table}_{stamp}.zip"
    for k in range(1, 10):
        fname = f"PUBLIC_ARCHIVE#{table}#FILE{k:02d}#{stamp}.zip"
        yield f"new{k}", fname, f"{d}/" + fname.replace("#", "%23")


def fetch(url: str, dest: str, retries: int = 3) -> bool:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=120) as r, open(dest + ".part", "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            os.replace(dest + ".part", dest)
            return True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False
            print(f"  HTTP {e.code} on {url} (attempt {attempt + 1})")
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"  network error {e} (attempt {attempt + 1})")
        time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"Failed to download {url}")


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-07")
    ap.add_argument("--end", default="2025-06")
    ap.add_argument("--out", default=os.path.join("data", "aemo_raw"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    jobs = []
    # actuals: start .. end+1 (day D+1 morning is needed for the 30 h window)
    for y, m in month_range(args.start, shift_month(args.end, 1)):
        jobs.append(("ROOFTOP_PV_ACTUAL", y, m))
    # forecasts: start-1 .. end (day-ahead run is issued on D-1)
    for y, m in month_range(shift_month(args.start, -1), args.end):
        jobs.append(("ROOFTOP_PV_FORECAST", y, m))

    rows = []
    for table, y, m in jobs:
        got_any = False
        for kind, fname, url in candidate_urls(y, m, table):
            dest = os.path.join(args.out, fname.replace("#", "_"))
            if os.path.exists(dest):
                ok = True
                print(f"[skip] {fname} (exists)")
            else:
                print(f"[get ] {fname}")
                ok = fetch(url, dest)
            if ok:
                got_any = True
                rows.append({
                    "table": table, "year": y, "month": m, "file": os.path.basename(dest),
                    "url": url, "bytes": os.path.getsize(dest), "sha256": sha256(dest),
                    "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                })
            elif kind.startswith("new") and kind != "new1" and got_any:
                break  # no more split files
        if not got_any:
            # Months beyond the archive edge (e.g. end+1 actuals) are allowed to be missing.
            print(f"[warn] no archive found for {table} {y}-{m:02d}")

    if not rows:
        sys.exit("No files downloaded — check network access to nemweb.com.au")

    with open(os.path.join(args.out, "checksums.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Done: {len(rows)} files; checksums -> {args.out}/checksums.csv")


if __name__ == "__main__":
    main()
