import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Загрузка CSV
df = pd.read_csv("results.csv")

# ------------------ Рисунок 1 ------------------
# Сравнение времени выполнения для 4 потоков
threads_fixed = 4
subset_threads = df[df["threads"] == threads_fixed]

# Classic всегда threads=0
time_classic = df[df["method"] == "Classic"]["time_s"].values[0]
time_threads = subset_threads[subset_threads["method"] == "Threads"]["time_s"].values[0]
time_async = subset_threads[subset_threads["method"] == "Async"]["time_s"].values[0]

methods = ["Classic", "Threads", "Async"]
times = [time_classic, time_threads, time_async]

plt.figure(figsize=(6, 4))
plt.bar(methods, times, color=["gray", "blue", "green"])
plt.ylabel("Time (s)")
plt.title(f"Рисунок 1: Время для {threads_fixed} потоков")
plt.savefig("figure1.png")
plt.close()

# ------------------ Рисунок 2 ------------------
# Threads: разные размеры матриц и числа потоков
plt.figure(figsize=(8, 6))
for size in sorted(df["size"].unique()):
    t_values = []
    t_counts = sorted(df[df["method"] == "Threads"]["threads"].unique())
    for t in t_counts:
        val = df[
            (df["size"] == size) & (df["threads"] == t) & (df["method"] == "Threads")
        ]["time_s"]
        t_values.append(val.values[0] if len(val) > 0 else np.nan)
    plt.plot(t_counts, t_values, marker="o", label=f"Size {size}")
plt.xlabel("Number of threads")
plt.ylabel("Time (s)")
plt.title("Рисунок 2: Threads")
plt.legend()
plt.grid(True)
plt.savefig("figure2.png")
plt.close()

# ------------------ Рисунок 3 ------------------
# Async: разные размеры матриц и числа потоков
plt.figure(figsize=(8, 6))
for size in sorted(df["size"].unique()):
    t_values = []
    t_counts = sorted(df[df["method"] == "Async"]["threads"].unique())
    for t in t_counts:
        val = df[
            (df["size"] == size) & (df["threads"] == t) & (df["method"] == "Async")
        ]["time_s"]
        t_values.append(val.values[0] if len(val) > 0 else np.nan)
    plt.plot(t_counts, t_values, marker="o", label=f"Size {size}")
plt.xlabel("Number of threads")
plt.ylabel("Time (s)")
plt.title("Рисунок 3: Async")
plt.legend()
plt.grid(True)
plt.savefig("figure3.png")
plt.close()

# ------------------ Рисунок 4 ------------------
# Ускорение: Classic / Parallel
plt.figure(figsize=(8, 6))
for size in sorted(df["size"].unique()):
    t_classic = df[(df["size"] == size) & (df["method"] == "Classic")]["time_s"].values[
        0
    ]

    threads_counts = sorted(df[df["method"] == "Threads"]["threads"].unique())
    speedup_threads = []
    speedup_async = []

    for t in threads_counts:
        t_thr = df[
            (df["size"] == size) & (df["threads"] == t) & (df["method"] == "Threads")
        ]["time_s"]
        t_as = df[
            (df["size"] == size) & (df["threads"] == t) & (df["method"] == "Async")
        ]["time_s"]
        speedup_threads.append(
            t_classic / t_thr.values[0] if len(t_thr) > 0 else np.nan
        )
        speedup_async.append(t_classic / t_as.values[0] if len(t_as) > 0 else np.nan)

    plt.plot(threads_counts, speedup_threads, marker="o", label=f"Threads Size {size}")
    plt.plot(
        threads_counts,
        speedup_async,
        marker="x",
        linestyle="--",
        label=f"Async Size {size}",
    )

plt.xlabel("Number of threads")
plt.ylabel("Speedup")
plt.title("Рисунок 4: Ускорение")
plt.legend()
plt.grid(True)
plt.savefig("figure4.png")
plt.close()

print(
    "Графики построены и сохранены: figure1.png, figure2.png, figure3.png, figure4.png"
)
