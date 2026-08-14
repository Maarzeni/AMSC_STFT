#!/usr/bin/env python3
"""Turn the benchmark suite's text tables into tidy CSV.

This is the ONLY module that knows how the result files are laid out. Everything
downstream (plots, tables in the report) reads the CSV instead, so a change to
the C++ printing costs one fix here rather than a silent misreading everywhere.

Usage:
    python3 analysis/parse_results.py results/raw/*/ -o results/csv/

Writes results/csv/timings.csv and memory.csv, both in long ("tidy") form: one
measurement per row, machine included, so several runs can be concatenated and
plotted on the same axes.

Standard library only, on purpose: it has to run wherever the results do.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

# ── Section titles, as printed by benchmark_Suite.hpp ────────────────────────
SECTION_MARKERS = [
    ("FFT engines", "fft"),
    ("STFT: serial vs OpenMP", "stft"),
    ("Distributed STFT", "mpi"),
    ("Per-rank input footprint", "mem_analytic"),
    ("Peak resident memory per rank", "mem_rss"),
]

# "  ranks        : 2"  /  "  threads/rank : up to 2"  /  "  distribution : scatter"
HEADER_FIELD = re.compile(r"^\s{2}(\w[\w/ ]*?)\s*:\s*(.+?)\s*$")

# Sweep separators written by run_suite.sh, which carry the configuration that
# the run below them used.
RANK_BLOCK = re.compile(r"-np (\d+): pure (\d+)x(\d+)")
THREAD_BLOCK = re.compile(r"OMP_NUM_THREADS=(\d+)")
STRATEGY_BLOCK = re.compile(r"distribution:\s*(\w+)")


def is_number(tok: str) -> bool:
    try:
        float(tok)
        return True
    except ValueError:
        return False


def split_row(line: str, n_numbers: int):
    """Split a table row into (label, [numbers]).

    Labels contain spaces ("STFT serial (1 thread)"), so the split is done from
    the RIGHT: the last n tokens are the numeric columns, everything before them
    is the label. Column widths are then irrelevant, which is what keeps this
    working when a label grows longer.
    """
    tokens = line.split()
    if len(tokens) < n_numbers + 1:
        return None
    tail = tokens[-n_numbers:]
    # The speedup column is "1.218x" or "-"; everything else must be numeric.
    for tok in tail[:-1] if tail[-1].endswith("x") or tail[-1] == "-" else tail:
        if not is_number(tok):
            return None
    label = " ".join(tokens[: len(tokens) - n_numbers]).strip()
    return label, tail


def parse_speedup(tok: str):
    if tok.endswith("x"):
        try:
            return float(tok[:-1])
        except ValueError:
            return None
    return None


def variant_workers(label: str, ranks_ctx: int, threads_ctx: int):
    """Infer ranks and threads for a row from its own label.

    The label is authoritative — "MPI pure (1 thr/rank)" is one thread per rank
    whatever the run's OMP_NUM_THREADS was — and that is exactly the distinction
    that makes a hybrid row comparable with a pure one.
    """
    thr = re.search(r"\((\d+) thr", label)
    threads = int(thr.group(1)) if thr else None

    if label.startswith("serial") or "1 thread" in label:
        return 1, 1
    if label.startswith("STFT OpenMP"):
        return 1, threads or threads_ctx
    if label.startswith("MPI pure"):
        return ranks_ctx, 1
    if label.startswith("hybrid"):
        return ranks_ctx, threads or threads_ctx
    if label in ("RecursiveFFT", "IterativeFFT"):
        return 1, 1                      # single-threaded by construction
    if label == "ParallelFFT":
        return 1, threads_ctx            # built with 0 = auto-detect threads
    return None, None


def parse_file(path: Path, machine: str, timings: list, memory: list,
               source: str = "") -> None:
    section = None
    ranks_ctx, threads_ctx = None, None
    strategy = None
    workload = None

    for raw in path.read_text(errors="replace").splitlines():
        line = raw.rstrip()

        # Sweep block separators reset the configuration context.
        m = RANK_BLOCK.search(line)
        if m:
            ranks_ctx, threads_ctx = int(m.group(1)), None
            continue
        m = THREAD_BLOCK.search(line)
        if m and "════" in line:
            ranks_ctx, threads_ctx = 1, int(m.group(1))
            continue
        m = STRATEGY_BLOCK.search(line)
        if m and "════" in line:
            strategy = m.group(1)
            continue

        # Run header fields.
        m = HEADER_FIELD.match(line)
        if m:
            key, val = m.group(1).strip(), m.group(2)
            if key == "ranks":
                ranks_ctx = int(val)
            elif key == "threads/rank":
                digits = re.search(r"(\d+)", val)
                threads_ctx = int(digits.group(1)) if digits else None
            elif key == "max threads":
                # benchmark_Main prints this; it is what ParallelFFT and the
                # OpenMP rows actually used, and the only way to attribute a
                # thread-sweep run to its thread count.
                threads_ctx = int(val)
            elif key == "distribution":
                strategy = val
            elif key in ("workloads", "workload"):
                one = re.match(r"^([\d.]+) s", val)
                workload = float(one.group(1)) if one else None
            continue

        for marker, name in SECTION_MARKERS:
            if line.startswith(marker):
                section = name
                break
        else:
            if not line or line.startswith(("=", "-", "─", "═", "Variant",
                                            "Distribution @", "Metric", "Notes",
                                            "  -", "AMSC_STFT")):
                continue

            if section in ("fft", "stft", "mpi"):
                row = split_row(line, 6)
                if not row:
                    continue
                label, (size, mn, mean, med, sd, sp) = row
                ranks, threads = variant_workers(label, ranks_ctx or 1,
                                                 threads_ctx or 1)
                timings.append({
                    "machine": machine,
                    # Which sweep produced the row. Without it a rank-sweep row
                    # and a plain benchmark row are indistinguishable, and a
                    # figure meant to show one would quietly average in the other.
                    "source": source,
                    "section": section,
                    "variant": label,
                    "ranks": ranks,
                    "threads": threads,
                    # What the variant itself used, above; what the RUN was given,
                    # here. They differ exactly where it matters: a serial engine
                    # handed 8 threads still uses one, and plotting it against
                    # the run's thread count is how "this engine does not scale"
                    # becomes visible instead of collapsing to a single point.
                    "omp_threads": threads_ctx,
                    "workers": (ranks * threads) if ranks and threads else None,
                    "size": int(size),
                    "workload_s": workload,
                    "min_ms": float(mn),
                    "mean_ms": float(mean),
                    "median_ms": float(med),
                    "stddev_ms": float(sd),
                    "speedup_reported": parse_speedup(sp),
                })

            elif section == "mem_analytic":
                row = split_row(line, 5)
                if not row:
                    continue
                label, (frames, smin, smax, savg, mib) = row
                kind = label.split()[0]
                timings_workload = re.search(r"@\s*([\d.]+)", label)
                memory.append({
                    "machine": machine,
                    "kind": "analytic",
                    "strategy": kind,
                    "ranks": ranks_ctx,
                    "frames": int(frames),
                    "workload_s": float(timings_workload.group(1))
                                  if timings_workload else None,
                    "min_mib": float(smin) * 8 / 1024 / 1024,
                    "max_mib": float(smax) * 8 / 1024 / 1024,
                    "avg_mib": float(mib),
                })

            elif section == "mem_rss":
                row = split_row(line, 3)
                if not row:
                    continue
                label, (mn, mx, avg) = row
                strat = re.search(r"\((\w+)\)", label)
                memory.append({
                    "machine": machine,
                    "kind": "rss",
                    "strategy": strat.group(1) if strat else (strategy or "?"),
                    # The rank count is what makes the scatter's benefit visible:
                    # its footprint falls as 1/P while the broadcast's does not.
                    "ranks": ranks_ctx,
                    "frames": None,
                    "workload_s": workload,
                    "min_mib": float(mn),
                    "max_mib": float(mx),
                    "avg_mib": float(avg),
                })


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", type=Path,
                    help="directories holding <machine>_*.txt result files")
    ap.add_argument("-o", "--out", type=Path, default=Path("results/csv"),
                    help="output directory (default: results/csv/)")
    args = ap.parse_args()

    timings: list = []
    memory: list = []
    seen = 0

    for d in args.dirs:
        if not d.is_dir():
            print(f"skipping {d}: not a directory", file=sys.stderr)
            continue
        for path in sorted(d.glob("*.txt")):
            stem = path.stem
            if stem.endswith(("_ctest", "_env")):
                continue
            machine = stem.split("_")[0]
            source = stem[len(machine) + 1:] if "_" in stem else ""
            parse_file(path, machine, timings, memory, source)
            seen += 1

    if not timings and not memory:
        print("no measurements found — are these result directories?",
              file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("timings.csv", timings), ("memory.csv", memory)):
        if not rows:
            continue
        with (args.out / name).open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"{args.out / name}: {len(rows)} rows")

    print(f"parsed {seen} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
