#!/usr/bin/env python3

import csv
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display on the cluster
import matplotlib.pyplot as plt


def load(path: Path):
    #return (matrices in file order, data[matrix][kernel] -> row dict)
    data = {}
    with path.open() as f:
        for r in csv.DictReader(f):
            for k in ("time_ms", "gflops"):
                r[k] = float(r[k]) if r[k] != "" else None
            data.setdefault(r["matrix"], {})[r["kernel"]] = r
    return list(data), data


def is_cpu(kernel):
    return kernel.startswith("CPU")


def is_spmm(kernel):
    return "SpMM" in kernel


def kernels_of(matrices, data, spmm):
    #GPU kernels of one kind, in order of first appearance, excluding FAILs
    seen = []
    for m in matrices:
        for k, r in data[m].items():
            if is_cpu(k) or is_spmm(k) != spmm or k in seen:
                continue
            seen.append(k)
    return seen


def usable(row):
    return row is not None and row["pass"] != "FAIL"


def grouped_bars(matrices, data, kernels, title, out, log=False):
    n = len(kernels)
    if n == 0:
        print(f"skip {out.name}: no kernels found")
        return
    width = 0.8 / n
    fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(matrices) + 2), 5))
    for i, k in enumerate(kernels):
        xs, ys = [], []
        for j, m in enumerate(matrices):
            row = data[m].get(k)
            if usable(row) and row["gflops"] is not None:
                xs.append(j + (i - (n - 1) / 2) * width)
                ys.append(row["gflops"])
            elif row is not None and row["pass"] == "FAIL":
                print(f"note: {m} / {k} FAILED correctness, left out of {out.name}")
        ax.bar(xs, ys, width, label=k)
    ax.set_xticks(range(len(matrices)))
    ax.set_xticklabels(matrices, rotation=30, ha="right")
    ax.set_ylabel("GFLOP/s")
    ax.set_title(title)
    if log:
        ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3, which="both" if log else "major")
    ax.set_axisbelow(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")


def best_speedups(matrices, data, spmm):
    #per matrix: (speedup, best kernel name) or None
    cpu_name = "CPU SpMM" if spmm else "CPU SpMV"
    res = []
    for m in matrices:
        cpu = data[m].get(cpu_name)
        best = None
        for k, r in data[m].items():
            if is_cpu(k) or is_spmm(k) != spmm or not usable(r) or not r["time_ms"]:
                continue
            if best is None or r["time_ms"] < best[1]:
                best = (k, r["time_ms"])
        if cpu and cpu["time_ms"] and best:
            res.append((cpu["time_ms"] / best[1], best[0]))
        else:
            res.append(None)
    return res


def speedup_plot(matrices, data, out):
    series = {"SpMV": best_speedups(matrices, data, False),"SpMM": best_speedups(matrices, data, True)}
    series = {k: v for k, v in series.items() if any(x is not None for x in v)}
    if not series:
        print(f"skip {out.name}: no CPU baselines found")
        return
    n = len(series)
    width = 0.8 / n
    fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(matrices) + 2), 5))
    for i, (label, vals) in enumerate(series.items()):
        xs, ys = [], []
        for j, v in enumerate(vals):
            if v is None:
                continue
            xs.append(j + (i - (n - 1) / 2) * width)
            ys.append(v[0])
            print(f"best {label} kernel for {matrices[j]}: {v[1]} ({v[0]:.1f}x over CPU)")
        bars = ax.bar(xs, ys, width, label=label)
        for b, y in zip(bars, ys):
            ax.annotate(f"{y:.0f}x" if y >= 10 else f"{y:.1f}x",(b.get_x() + b.get_width() / 2, y),ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(len(matrices)))
    ax.set_xticklabels(matrices, rotation=30, ha="right")
    ax.set_ylabel("Speedup over CPU (best GPU kernel)")
    ax.set_title("Speedup over CPU baseline")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3, which="both")
    ax.set_axisbelow(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    default_csv = Path(__file__).resolve().parent.parent / "results" / "results.csv"

    if len(sys.argv) > 1:
        csv_path = Path(sys.argv[1])
    else:
        csv_path = default_csv

    if not csv_path.exists():
        sys.exit(f"error: {csv_path} not found (run parse_results.py first)")

    out_dir = csv_path.parent

    matrices, data = load(csv_path)

    spmv_kernels = kernels_of(matrices, data, spmm=False)
    grouped_bars(
        matrices,
        data,
        spmv_kernels,
        "SpMV performance (GFLOP/s)",
        out_dir / "spmv_gflops.png"
    )

    spmm_kernels = kernels_of(matrices, data, spmm=True)
    grouped_bars(
        matrices,
        data,
        spmm_kernels,
        "SpMM performance (GFLOP/s, log scale)",
        out_dir / "spmm_gflops.png",
        log=True
    )

    speedup_plot(matrices, data, out_dir / "speedup_vs_cpu.png")


if __name__ == "__main__":
    main()
