"""
Shared inputs, method set and figure style for the paper post-processing.

Every figure, table and number is computed from committed files only:
  results/<EXP>/summary/runs.csv, results/E2/summary/previews.csv,
  results/E1/raw/*.json.gz and results/E2/raw/*.json.gz (force-added traces),
  data/aemo_vic1_selected_days.json, simulation_data_<N>EVs.json, config/experiments.yaml.

Method set (decision 2026-10-16): AQPS-v1 (submitted method) is excluded from every output.
"""

import gzip
import json
import os

import matplotlib as mpl
import numpy as np
import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "paper")
FIG_DIR = os.path.join(OUT, "figures")
TAB_DIR = os.path.join(OUT, "tables")

EXCLUDED_METHODS = ["AQPS-v1"]

# Figure methods (AQPS-RT appears in tables only: same outcomes as AQPS-RT-ET, see E5a)
FIG_METHODS = ["LLF", "EDF", "MPC-only", "Offline-Optimal", "AQPS-static", "AQPS-RT-ET"]
TAB_METHODS = ["LLF", "EDF", "MPC-only", "Offline-Optimal", "AQPS-static", "AQPS-RT", "AQPS-RT-ET"]
LABEL = {"LLF": "LLF", "EDF": "EDF", "MPC-only": "MPC-only", "Offline-Optimal": "Offline-Opt.",
         "AQPS-static": "AQPS-static", "AQPS-RT": "AQPS-RT", "AQPS-RT-ET": "AQPS-RT-ET"}

# Categorical colours: validated with the dataviz validator (light surface, all pairs):
# lightness band and chroma PASS; worst CVD pair dE 6.9 (WARN) -> secondary encoding is used
# (fixed bar order, white gaps between bars, legend in every figure, hatch on the offline bound).
# Offline-Optimal is a perfect-information bound, drawn neutral grey with hatching.
COLOR = {"LLF": "#e34948", "EDF": "#eda100", "MPC-only": "#008300", "Offline-Optimal": "#7a7a7a",
         "AQPS-static": "#1baf7a", "AQPS-RT": "#2a78d6", "AQPS-RT-ET": "#2a78d6"}
HATCH = {"Offline-Optimal": "////"}

# Ordinal ramp (one hue, light -> dark) for tariff tiers and kW intensity
TIER_COLOR = {"off_peak": "#86b6ef", "shoulder": "#2a78d6", "peak": "#104281"}
TIER_LABEL = {"off_peak": "Off-peak", "shoulder": "Shoulder", "peak": "Peak"}

INK = "#1f1f1f"
INK2 = "#555555"
GRID = "#e3e3e3"

SCEN = ["S1_Baseline", "S2_Contention", "S3_PeakStress"]
SCEN_LABEL = {"S1_Baseline": "S1 Baseline", "S2_Contention": "S2 Contention", "S3_PeakStress": "S3 Peak stress"}

# Figure widths = column and text width of elsarticle [3p,twocolumn] (222 pt and 468 pt), so the
# 7-8 pt figure fonts print at their nominal size in the manuscript
PT = 1 / 72.27
W1 = 222 * PT
W2 = 468 * PT


def style():
    mpl.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix",
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.5,
        "axes.axisbelow": True, "legend.frameon": False, "hatch.linewidth": 0.5,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600,
        "figure.dpi": 150, "lines.linewidth": 1.2,
    })


def save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, name + ".pdf"), metadata={"CreationDate": None})  # vector, fonts embedded, reproducible
    fig.savefig(os.path.join(FIG_DIR, name + ".png"), dpi=300)  # preview only
    print("  figure", name)


def runs(exp: str) -> pd.DataFrame:
    d = pd.read_csv(os.path.join(RES, exp, "summary", "runs.csv"))
    return d[~d.method.isin(EXCLUDED_METHODS)].reset_index(drop=True)


def raw(exp: str, run_id: str) -> dict:
    with gzip.open(os.path.join(RES, exp, "raw", run_id + ".json.gz"), "rt", encoding="utf-8") as f:
        return json.load(f)


def cfg() -> dict:
    with open(os.path.join(ROOT, "config", "experiments.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


def fleet_data(n: int) -> dict:
    with open(os.path.join(ROOT, f"simulation_data_{n}EVs.json"), encoding="utf-8") as f:
        return json.load(f)


def pv_days() -> dict:
    with open(os.path.join(ROOT, "data", "aemo_vic1_selected_days.json"), encoding="utf-8") as f:
        return json.load(f)


def step_to_hour(t, start_hour=6, step_min=15):
    return start_hour + np.asarray(t) * step_min / 60.0


def hour_ticks(ax, h0=6, h1=36, every=3, axis="x"):
    ticks = np.arange(h0, h1 + 1e-9, every)
    labels = [f"{int(h % 24):02d}:00" for h in ticks]
    if axis == "x":
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
    return ticks


def ci95(x):
    """Mean and half-width of the two-sided 95 % t confidence interval."""
    from scipy import stats
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 2:
        return float(x.mean()), float("nan")
    return float(x.mean()), float(stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n))
