#!/usr/bin/env python3
"""
plot.py

Reads perf_results.csv produced by main.out and generates PNG plots.

CSV format expected:
impl,phase,producers,consumers,per,capacity,total_ops,elapsed_s,throughput_ops_s,samples,mean_us,p50_us,p90_us,p99_us,p999_us,max_us
"""
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


def read_csv(path):
    df = pd.read_csv(path)
    return df


def plot_queue_latency_vs_load(df, out_prefix="queue_latency_vs_load"):
    # filter queue rows (LockFreeQueue and MutexQueue), phase overall
    qdf = df[
        (df["impl"].isin(["LockFreeQueue", "MutexQueue"])) & (df["phase"] == "overall")
    ]
    if qdf.empty:
        print("No queue overall data for latency vs load")
        return
    # avoid SettingWithCopyWarning by working on a copy
    qdf = qdf.copy()
    qdf["pc"] = qdf["producers"].astype(str) + "p_" + qdf["consumers"].astype(str) + "c"
    # We will plot p50 and mean for both impls
    for metric in ["p50_us", "p90_us", "p99_us", "mean_us"]:
        if metric not in qdf.columns:
            print(f"Metric {metric} not found in data, skipping")
            continue
        plt.figure(figsize=(10, 6))
        sns.pointplot(data=qdf, x="pc", y=metric, hue="impl", dodge=True)
        plt.title(f"Queue {metric} vs producer/consumer mix")
        plt.xlabel("producers_consumers")
        plt.ylabel(metric + " (us)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        fname = f"{out_prefix}_{metric}.png"
        plt.savefig(fname, dpi=150)
        print("Saved", fname)
        plt.close()


def plot_queue_latency_vs_per(df, out_prefix="queue_latency_vs_per"):
    qdf = df[
        (df["impl"].isin(["LockFreeQueue", "MutexQueue"])) & (df["phase"] == "overall")
    ]
    subset = qdf[(qdf["producers"] == 4) & (qdf["consumers"] == 4)]
    if subset.empty:
        print("No data for producers=4 consumers=4 to plot queue latency vs per")
        return
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=subset, x="per", y="mean_us", hue="impl", marker="o")
    plt.title("Queue mean latency vs per (4p x 4c)")
    plt.xlabel("per (operations per producer)")
    plt.ylabel("mean latency (us)")
    plt.tight_layout()
    fname = out_prefix + ".png"
    plt.savefig(fname, dpi=150)
    print("Saved", fname)
    plt.close()


def plot_queue_capacity_sweep(df, out_prefix="queue_capacity_sweep"):
    qdf = df[
        (df["impl"].isin(["LockFreeQueue", "MutexQueue"])) & (df["phase"] == "overall")
    ]
    subset = qdf[(qdf["producers"] == 4) & (qdf["consumers"] == 4)]
    if subset.empty:
        print("No data for capacity sweep (4p x 4c)")
        return
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=subset, x="capacity", y="mean_us", hue="impl", marker="o")
    plt.title("Queue mean latency vs capacity (4p x 4c)")
    plt.xlabel("capacity")
    plt.ylabel("mean latency (us)")
    # support both newer and older matplotlib API for log base argument
    try:
        plt.xscale("log", base=2)
    except TypeError:
        # older matplotlib used 'basex'
        plt.xscale("log", basex=2)
    plt.tight_layout()
    fname = out_prefix + ".png"
    plt.savefig(fname, dpi=150)
    print("Saved", fname)
    plt.close()


def plot_queue_vs_list(df, out_prefix="queue_vs_list"):
    q = df[
        (df["impl"] == "LockFreeQueue")
        & (df["phase"] == "overall")
        & (df["producers"] == 4)
        & (df["consumers"] == 4)
    ]
    m = df[
        (df["impl"] == "MutexQueue")
        & (df["phase"] == "overall")
        & (df["producers"] == 4)
        & (df["consumers"] == 4)
    ]
    fl_ins = df[
        (df["impl"] == "FineGrainedList")
        & (df["phase"] == "insert")
        & (df["producers"] == 4)
    ]
    fl_rem = df[
        (df["impl"] == "FineGrainedList")
        & (df["phase"] == "pop_front")
        & (df["producers"] == 4)
    ]
    ml_ins = df[
        (df["impl"] == "MutexList") & (df["phase"] == "insert") & (df["producers"] == 4)
    ]
    ml_rem = df[
        (df["impl"] == "MutexList")
        & (df["phase"] == "pop_front")
        & (df["producers"] == 4)
    ]

    records = []
    if not q.empty:
        records.append(
            {
                "thing": "LockFreeQueue (4p4c)",
                "mean_us": float(q.iloc[-1]["mean_us"]),
                "throughput": float(q.iloc[-1]["throughput_ops_s"]),
            }
        )
    if not m.empty:
        records.append(
            {
                "thing": "MutexQueue (4p4c)",
                "mean_us": float(m.iloc[-1]["mean_us"]),
                "throughput": float(m.iloc[-1]["throughput_ops_s"]),
            }
        )
    if not fl_ins.empty:
        records.append(
            {
                "thing": "FineGrainedList insert (4p)",
                "mean_us": float(fl_ins.iloc[-1]["mean_us"]),
                "throughput": float(fl_ins.iloc[-1]["throughput_ops_s"]),
            }
        )
    if not fl_rem.empty:
        records.append(
            {
                "thing": "FineGrainedList pop_front (4p)",
                "mean_us": float(fl_rem.iloc[-1]["mean_us"]),
                "throughput": float(fl_rem.iloc[-1]["throughput_ops_s"]),
            }
        )
    if not ml_ins.empty:
        records.append(
            {
                "thing": "MutexList insert (4p)",
                "mean_us": float(ml_ins.iloc[-1]["mean_us"]),
                "throughput": float(ml_ins.iloc[-1]["throughput_ops_s"]),
            }
        )
    if not ml_rem.empty:
        records.append(
            {
                "thing": "MutexList pop_front (4p)",
                "mean_us": float(ml_rem.iloc[-1]["mean_us"]),
                "throughput": float(ml_rem.iloc[-1]["throughput_ops_s"]),
            }
        )

    if not records:
        print("No data for Queue vs List comparison")
        return

    rdf = pd.DataFrame.from_records(records)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=rdf, x="thing", y="mean_us")
    plt.xticks(rotation=45)
    plt.ylabel("mean latency (us)")
    plt.title("Queue vs List mean latency (select tests)")
    plt.tight_layout()
    fname = out_prefix + "_latency.png"
    plt.savefig(fname, dpi=150)
    print("Saved", fname)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.barplot(data=rdf, x="thing", y="throughput")
    plt.xticks(rotation=45)
    plt.ylabel("throughput (ops/s)")
    plt.title("Queue vs List throughput (select tests)")
    plt.tight_layout()
    fname = out_prefix + "_throughput.png"
    plt.savefig(fname, dpi=150)
    print("Saved", fname)
    plt.close()


def plot_list_throughput_vs_producers(df, out_prefix="list_throughput_vs_producers"):
    ins = df[
        (df["impl"].isin(["MutexList", "FineGrainedList"])) & (df["phase"] == "insert")
    ]
    if ins.empty:
        print("No data for list throughput vs producers")
        return
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=ins, x="producers", y="throughput_ops_s", hue="impl", marker="o")
    plt.title("List insert throughput vs producers")
    plt.xlabel("producers")
    plt.ylabel("throughput (ops/s)")
    plt.tight_layout()
    fname = out_prefix + ".png"
    plt.savefig(fname, dpi=150)
    print("Saved", fname)
    plt.close()


def plot_list_latency_vs_per(df, out_prefix="list_latency_vs_per"):
    ins = df[
        (df["impl"].isin(["MutexList", "FineGrainedList"]))
        & (df["phase"] == "insert")
        & (df["producers"] == 4)
    ]
    rem = df[
        (df["impl"].isin(["MutexList", "FineGrainedList"]))
        & (df["phase"].isin(["remove", "pop_front"]))
        & (df["producers"] == 4)
    ]
    if ins.empty and rem.empty:
        print("No data for list latency vs per (4 producers)")
        return
    plt.figure(figsize=(8, 5))
    if not ins.empty:
        sns.lineplot(data=ins, x="per", y="mean_us", hue="impl", marker="o")
    if not rem.empty:
        sns.lineplot(
            data=rem, x="per", y="mean_us", hue="impl", marker="x", linestyle="--"
        )
    plt.title("List mean latency vs per (4 producers)")
    plt.xlabel("per")
    plt.ylabel("mean latency (us)")
    plt.tight_layout()
    fname = out_prefix + ".png"
    plt.savefig(fname, dpi=150)
    print("Saved", fname)
    plt.close()


def main(csv_path):
    df = read_csv(csv_path)
    # normalize column types if they exist
    for col in ["producers", "consumers", "per", "capacity", "samples"]:
        if col in df.columns:
            df[col] = df[col].astype(int)
    for col in ["mean_us", "p50_us", "throughput_ops_s"]:
        if col in df.columns:
            df[col] = df[col].astype(float)

    plot_queue_latency_vs_load(df)
    plot_queue_latency_vs_per(df)
    plot_queue_capacity_sweep(df)
    plot_queue_vs_list(df)
    plot_list_throughput_vs_producers(df)
    plot_list_latency_vs_per(df)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 plot.py perf_results.csv")
        sys.exit(1)
    csv_path = sys.argv[1]
    if not os.path.exists(csv_path):
        print("CSV file not found:", csv_path)
        sys.exit(2)
    main(csv_path)
