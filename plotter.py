#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_telemetry.py - Publication-quality plots for 4-GPU telemetry JSONL
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# Utilities
# ----------------------------
def read_json_stream(path: Path) -> List[Dict[str, Any]]:
    """Read JSONL (preferred) or a single JSON list file."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    rows: List[Dict[str, Any]] = []
    # Try JSONL
    try:
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
        if rows and isinstance(rows[0], dict) and "timestamp" in rows[0]:
            return rows
    except Exception:
        rows = []
    # Try JSON (list)
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        raise ValueError("Top-level JSON must be a list of objects.")
    except Exception as e:
        raise RuntimeError(f"Could not parse {path}: {e}")


def load_telemetry(files: List[Path]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (gpu_df, cpu_df) long-form dataframes."""
    raw: List[Dict[str, Any]] = []
    for p in files:
        raw.extend(read_json_stream(p))
    if not raw:
        raise RuntimeError("No telemetry rows found in the input files.")

    gpu_rows, cpu_rows = [], []
    for row in raw:
        ts = row.get("timestamp")
        for g in row.get("gpus", []):
            rec = dict(g)
            rec["timestamp"] = ts
            gpu_rows.append(rec)
        if "cpu" in row:
            cpu_rec = dict(row["cpu"])
            cpu_rec["timestamp"] = ts
            cpu_rows.append(cpu_rec)

    gpu_df = pd.DataFrame(gpu_rows)
    cpu_df = pd.DataFrame(cpu_rows) if cpu_rows else pd.DataFrame([])

    for df in (gpu_df, cpu_df):
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df.dropna(subset=["timestamp"], inplace=True)
            df.sort_values("timestamp", inplace=True)

    if "gpu_index" in gpu_df.columns:
        gpu_df["gpu_index"] = pd.to_numeric(gpu_df["gpu_index"], errors="coerce").astype("Int64")

    if not gpu_df.empty:
        t0 = gpu_df["timestamp"].min()
        gpu_df["elapsed_s"] = (gpu_df["timestamp"] - t0).dt.total_seconds()
    if not cpu_df.empty:
        t0c = cpu_df["timestamp"].min()
        cpu_df["elapsed_s"] = (cpu_df["timestamp"] - t0c).dt.total_seconds()

    return gpu_df.reset_index(drop=True), cpu_df.reset_index(drop=True)


def resample_df(df: pd.DataFrame, seconds: int, key_cols: List[str]) -> pd.DataFrame:
    """
    Resample numeric columns to fixed interval by mean.
    Handles both GPU (grouped by key_cols) and CPU (no key_cols).
    """
    if seconds <= 0 or df.empty or "timestamp" not in df.columns:
        return df

    df = df.copy()
    df.set_index("timestamp", inplace=True)
    freq = f"{seconds}s"

    if not key_cols:
        rs = df.resample(freq).mean(numeric_only=True).reset_index()
        rs["elapsed_s"] = (rs["timestamp"] - rs["timestamp"].min()).dt.total_seconds()
        return rs.sort_values("timestamp").reset_index(drop=True)

    groups = df.groupby([c for c in key_cols if c in df.columns])
    parts = []
    for _, sub in groups:
        r = sub.resample(freq).mean(numeric_only=True)
        for c in key_cols:
            if c in sub.columns:
                r[c] = sub[c].iloc[0]
        parts.append(r.reset_index())

    out = pd.concat(parts, ignore_index=True)
    out["elapsed_s"] = (out["timestamp"] - out["timestamp"].min()).dt.total_seconds()
    sort_keys = ["timestamp"] + [c for c in key_cols if c in out.columns]
    return out.sort_values(sort_keys).reset_index(drop=True)


def trapezoid_energy_wh(df: pd.DataFrame) -> float:
    if df.empty or "power_watts" not in df.columns:
        return 0.0
    s = df.dropna(subset=["timestamp", "power_watts"]).sort_values("timestamp")
    if s.empty:
        return 0.0
    t = s["timestamp"].astype("int64") / 1e9
    p = s["power_watts"].astype(float).to_numpy()
    joules = float(np.trapz(p, t))
    return joules / 3600.0


def delta_sum_energy_mj(df: pd.DataFrame) -> float:
    if "energy_mJ" not in df.columns or df["energy_mJ"].isna().all():
        return np.nan
    s = df["energy_mJ"].dropna().astype(float).sort_index()
    if s.empty:
        return np.nan
    deltas = np.diff(s.values)
    deltas = np.clip(deltas, 0.0, None)
    total_mJ = float(deltas.sum())
    joules = total_mJ / 1e3
    return joules / 3600.0


def compute_summaries(gpu_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metrics = ["power_watts", "gpu_utilization_percent", "memory_used_MB", "temperature_C"]
    rows, erows = [], []
    for gid, sub in gpu_df.groupby("gpu_index", dropna=True):
        rec = {"gpu_index": int(gid)}
        for m in metrics:
            if m in sub.columns:
                s = sub[m].astype(float)
                rec[f"{m}_mean"] = float(s.mean())
                rec[f"{m}_max"] = float(s.max())
                rec[f"{m}_min"] = float(s.min())
        wh_int = trapezoid_energy_wh(sub)
        wh_mj = delta_sum_energy_mj(sub)
        rec["energy_Wh_integrated"] = wh_int
        rec["energy_J_integrated"] = wh_int * 3600.0
        rec["energy_Wh_mJ_deltas"] = wh_mj
        rec["energy_J_mJ_deltas"] = wh_mj * 3600.0 if not np.isnan(wh_mj) else np.nan
        rows.append(rec)
        erows.append({"gpu_index": int(gid),
                      "energy_Wh_integrated": wh_int,
                      "energy_Wh_mJ_deltas": wh_mj})
    return pd.DataFrame(rows), pd.DataFrame(erows)


# ----------------------------
# Plotting
# ----------------------------
GPU_COLORS = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c", 3: "#d62728"}


def _style_axes(ax, title: str, x_label: str, y_label: str):
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.grid(True, alpha=0.25, linewidth=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _save(fig: plt.Figure, outdir: Path, fname_stem: str, formats: List[str], dpi: int):
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        fig.savefig(outdir / f"{fname_stem}.{ext}", dpi=dpi, bbox_inches="tight")


def plot_lines_per_gpu(gpu_df: pd.DataFrame, metric: str, ylabel: str, outdir: Path,
                       x_mode: str, formats: List[str], dpi: int, title_suffix: str = ""):
    if metric not in gpu_df.columns:
        return
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    xcol = "elapsed_s" if x_mode == "elapsed" else "timestamp"
    for gid, sub in gpu_df.groupby("gpu_index", dropna=True):
        sub = sub.dropna(subset=[xcol, metric]).sort_values(xcol)
        if sub.empty:
            continue
        color = GPU_COLORS.get(int(gid), None)
        label = f"GPU{int(gid)}"
        ax.plot(sub[xcol], sub[metric].astype(float), label=label, linewidth=1.6, color=color)
    _style_axes(ax, f"{metric} {title_suffix}".strip(), "Elapsed (s)" if x_mode == "elapsed" else "Time", ylabel)
    ax.legend(frameon=False, ncol=2)
    _save(fig, outdir, f"{metric}", formats, dpi)
    plt.close(fig)


def plot_cpu(cpu_df: pd.DataFrame, outdir: Path, x_mode: str, formats: List[str], dpi: int):
    if cpu_df.empty or "cpu_utilization_percent" not in cpu_df.columns:
        return
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    xcol = "elapsed_s" if x_mode == "elapsed" else "timestamp"
    sub = cpu_df.dropna(subset=[xcol, "cpu_utilization_percent"]).sort_values(xcol)
    ax.plot(sub[xcol], sub["cpu_utilization_percent"].astype(float), linewidth=1.6, label="CPU Utilization (%)")
    _style_axes(ax, "CPU Utilization", "Elapsed (s)" if x_mode == "elapsed" else "Time", "Percent")
    ax.legend(frameon=False)
    _save(fig, outdir, "cpu_util", formats, dpi)
    plt.close(fig)


def plot_energy_bars(energy_df: pd.DataFrame, outdir: Path, formats: List[str], dpi: int):
    if energy_df.empty:
        return
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ed = energy_df.sort_values("gpu_index")
    x = np.arange(len(ed))
    width = 0.35
    ax.bar(x - width/2, ed["energy_Wh_integrated"], width, label="Integrated Power (Wh)", color="#4C78A8")
    if "energy_Wh_mJ_deltas" in ed.columns and not ed["energy_Wh_mJ_deltas"].isna().all():
        ax.bar(x + width/2, ed["energy_Wh_mJ_deltas"], width, label="mJ Delta Sum (Wh)", color="#F58518")
    ax.set_xticks(x, [f"GPU{int(i)}" for i in ed["gpu_index"]])
    _style_axes(ax, "Energy per GPU", "GPU", "Watt-hours (Wh)")
    ax.legend(frameon=False)
    total_wh = float(ed["energy_Wh_integrated"].sum())
    ax.text(0.98, 0.98, f"Total (Integrated): {total_wh:.2f} Wh", ha="right", va="top",
            transform=ax.transAxes, fontsize=10)
    _save(fig, outdir, "energy_bars", formats, dpi)
    plt.close(fig)


# ----------------------------
# CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Plot GPU/CPU telemetry (research-grade).")
    ap.add_argument("--telemetry", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resample-seconds", type=int, default=0)
    ap.add_argument("--x", choices=["elapsed", "wall"], default="elapsed")
    ap.add_argument("--format", nargs="+", default=["png"])
    ap.add_argument("--dpi", type=int, default=160)
    args = ap.parse_args()

    files = [Path(p) for p in args.telemetry]
    outdir = Path(args.out)
    figs_dir, tbl_dir = outdir / "figures", outdir / "tables"
    figs_dir.mkdir(parents=True, exist_ok=True)
    tbl_dir.mkdir(parents=True, exist_ok=True)

    gpu_df, cpu_df = load_telemetry(files)

    if args.resample_seconds > 0:
        gpu_df = resample_df(gpu_df, args.resample_seconds, key_cols=["gpu_index"])
        if not cpu_df.empty:
            cpu_df = resample_df(cpu_df, args.resample_seconds, key_cols=[])

    summary_df, energy_df = compute_summaries(gpu_df)

    summary_df.to_csv(tbl_dir / "summary_per_gpu.csv", index=False)
    energy_df.to_csv(tbl_dir / "energy_per_gpu.csv", index=False)
    gpu_df.to_csv(outdir / "data_gpu_long.csv", index=False)
    if not cpu_df.empty:
        cpu_df.to_csv(outdir / "data_cpu.csv", index=False)

    plot_lines_per_gpu(gpu_df, "power_watts", "Watts", figs_dir, args.x, args.format, args.dpi)
    plot_lines_per_gpu(gpu_df, "gpu_utilization_percent", "Percent", figs_dir, args.x, args.format, args.dpi)
    if "memory_used_MB" in gpu_df.columns:
        gpu_df = gpu_df.copy()
        gpu_df["memory_used_GB"] = gpu_df["memory_used_MB"].astype(float) / 1024.0
        plot_lines_per_gpu(gpu_df, "memory_used_GB", "GB", figs_dir, args.x, args.format, args.dpi)
    plot_lines_per_gpu(gpu_df, "temperature_C", "°C", figs_dir, args.x, args.format, args.dpi)
    plot_cpu(cpu_df, figs_dir, args.x, args.format, args.dpi)
    plot_energy_bars(energy_df, figs_dir, args.format, args.dpi)

    print(f"[OK] Figures -> {figs_dir}")
    print(f"[OK] Tables  -> {tbl_dir}")


if __name__ == "__main__":
    main()
