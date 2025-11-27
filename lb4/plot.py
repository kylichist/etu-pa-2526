#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_matrix_speedup(matmul_csv: Path, lb1_csv: Path, out_png: Path):
    """
    Matrix Multiplication Speedup:
      - Compare CPU (block, threads=16) from lb1_csv vs GPU from matmul_csv.
      - The matmul_csv is expected to contain GPU timings (gpu_time_ms).
      - lb1_csv is expected to contain threaded CPU timings (columns: size, threads, method, time_s).
      - Speedup is computed as: cpu_block_time / gpu_time (both in seconds).
      - The CPU(block,threads=16) curve is the baseline (speedup == 1.0).
    """
    mat = pd.read_csv(matmul_csv)
    lb1 = pd.read_csv(lb1_csv)

    # normalize column names
    mat.columns = [c.strip() for c in mat.columns]
    lb1.columns = [c.strip() for c in lb1.columns]

    # --- process matmul (GPU) ---
    if "size" not in mat.columns:
        raise RuntimeError("matmul CSV must contain a 'size' column")
    # Accept gpu_time_ms or gpu_time_s
    if "gpu_time_ms" in mat.columns:
        mat["gpu_sec"] = mat["gpu_time_ms"].astype(float) / 1000.0
    elif "gpu_time_s" in mat.columns:
        mat["gpu_sec"] = mat["gpu_time_s"].astype(float)
    else:
        raise RuntimeError(
            "matmul CSV must contain 'gpu_time_ms' or 'gpu_time_s' column"
        )

    mat["size"] = mat["size"].astype(int)
    mat = (
        mat[["size", "gpu_sec"]].drop_duplicates(subset=["size"]).reset_index(drop=True)
    )

    # --- process lb1 (CPU block threaded) ---
    required_cols = {"size", "threads", "method", "time_s"}
    block = pd.DataFrame(columns=["size", "block_time_s"])
    if required_cols.issubset(set(lb1.columns)):
        lb1 = lb1.copy()
        lb1["method_l"] = lb1["method"].astype(str).str.lower()
        # try to filter by threads == 16; fallback to method string if threads cast fails
        try:
            lb1["threads_i"] = lb1["threads"].astype(int)
            cond = (lb1["threads_i"] == 16) & (lb1["method_l"].str.contains("thread"))
        except Exception:
            cond = lb1["method_l"].str.contains("thread")
        block = lb1[cond].copy()
        if not block.empty:
            block["size"] = block["size"].astype(int)
            # group by explicit 'size' column to avoid future pandas warnings
            block = (
                block.groupby("size", as_index=False)["time_s"]
                .mean()
                .rename(columns={"time_s": "block_time_s"})
            )
        else:
            block = pd.DataFrame(columns=["size", "block_time_s"])
    else:
        print(
            "[warn] lb1 CSV missing required columns (size, threads, method, time_s). CPU block curve will be skipped."
        )

    # --- merge and compute speedup (relative to cpu block) ---
    merged = block.merge(mat, on="size", how="inner")
    if merged.empty:
        raise RuntimeError(
            "No matching sizes between lb1 (cpu block) and matmul (gpu) CSVs to plot."
        )

    merged = merged.sort_values("size").reset_index(drop=True)
    merged["cpu_block_speedup"] = 1.0
    merged["gpu_speedup"] = merged.apply(
        lambda r: (
            (r["block_time_s"] / r["gpu_sec"])
            if (pd.notna(r.get("gpu_sec")) and r["gpu_sec"] > 0)
            else float("nan")
        ),
        axis=1,
    )

    # --- plotting ---
    plt.figure(figsize=(8, 5))
    # plot baseline (CPU block) as horizontal 1.0 value at each size for alignment
    plt.plot(
        merged["size"],
        merged["cpu_block_speedup"],
        marker="o",
        label="CPU (block, threads=16) [baseline]",
    )
    plt.plot(merged["size"], merged["gpu_speedup"], marker="o", label="GPU")

    plt.xlabel("matrix size")
    plt.ylabel("Speedup (relative to CPU block, threads=16)")
    plt.title("Matrix Multiplication Speedup: CPU(block,threads=16) vs GPU")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"[saved] {out_png}")


def plot_qsort_speedup(sort_csv: Path, out_png: Path):
    """
    Qsort Speedup: cpu vs gpu (unchanged)
      - Expects sort_csv to contain 'size', 'cpu_time_ms', 'gpu_time_ms'
    """
    s = pd.read_csv(sort_csv)
    s.columns = [c.strip() for c in s.columns]

    if (
        "size" not in s.columns
        or "cpu_time_ms" not in s.columns
        or "gpu_time_ms" not in s.columns
    ):
        raise RuntimeError(
            "sort CSV must contain columns: size, cpu_time_ms, gpu_time_ms"
        )

    s = s.copy()
    s["size"] = s["size"].astype(int)
    s["cpu_sec"] = s["cpu_time_ms"].astype(float) / 1000.0
    s["gpu_sec"] = s["gpu_time_ms"].astype(float) / 1000.0
    s = s.sort_values("size").reset_index(drop=True)

    s["cpu_speedup"] = 1.0
    s["gpu_speedup"] = s.apply(
        lambda r: (
            (r["cpu_sec"] / r["gpu_sec"])
            if (pd.notna(r.get("gpu_sec")) and r["gpu_sec"] > 0)
            else float("nan")
        ),
        axis=1,
    )

    plt.figure(figsize=(8, 5))
    plt.plot(s["size"], s["cpu_speedup"], marker="o", label="CPU (baseline)")
    plt.plot(s["size"], s["gpu_speedup"], marker="o", label="GPU")

    plt.xlabel("input size (elements)")
    plt.ylabel("Speedup (relative to CPU)")
    plt.title("Qsort Speedup: cpu vs gpu")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"[saved] {out_png}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate two speedup plots: matrix and qsort."
    )
    parser.add_argument(
        "--matmul-csv",
        type=Path,
        default=Path("results_matmul.csv"),
        help="CSV with matmul results (default: results_matmul.csv, should contain gpu_time_ms or gpu_time_s)",
    )
    parser.add_argument(
        "--lb1-csv",
        type=Path,
        default=Path("results_lb1_patched.csv"),
        help="CSV with CPU threaded results (default: results_lb1_patched.csv, must contain size,threads,method,time_s)",
    )
    parser.add_argument(
        "--sort-csv",
        type=Path,
        default=Path("results_sort.csv"),
        help="CSV with sort results (default: results_sort.csv)",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("plots"), help="Output directory for plots"
    )
    args = parser.parse_args()

    out_dir = args.out_dir
    plot_matrix_speedup(args.matmul_csv, args.lb1_csv, out_dir / "matrix_speedup.png")
    plot_qsort_speedup(args.sort_csv, out_dir / "qsort_speedup.png")


if __name__ == "__main__":
    main()
