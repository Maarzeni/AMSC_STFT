#!/usr/bin/env python3
"""Tag the benchmark suite's CSV output with machine/source and split it.

The benchmarks (core/benchmarks/) write tidy CSV rows directly — one row per
measurement, ranks/threads/speedup as real columns, no text table to
re-derive them from. What is left for this script is only what a single
binary invocation cannot know about itself: which MACHINE produced a file,
and which SOURCE section of run_suite.sh produced it (a fixed benchmark run
vs a rank sweep vs a thread sweep, say — the same section/variant names can
appear in more than one, and a figure meant to read one sweep would otherwise
silently mix in another). Both are carried by the file name, which
run_suite.sh already writes as `<machine>_<source>.csv`.

Usage:
    python3 analysis/parse_results.py results/raw/*/ -o results/analysis/csv/

Writes results/analysis/csv/timings.csv and memory.csv, both in long ("tidy")
form —
rows from every file and every machine given, concatenated, so several runs
can be plotted on the same axes. Which output a row belongs to is decided by
its own `kind` column (see core/benchmarks/benchmark_Suite.hpp): "timing"
rows go to timings.csv, "memory_analytic" and "memory_rss" rows to
memory.csv.

Standard library only, on purpose: it has to run wherever the results do.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# memory_*.csv rows carry the shared "kind" column but not "section"/"variant"
# (timing-only) or vice versa; each output file keeps only the columns that
# apply to it, in the order the C++ schema declares them.
TIMING_COLUMNS = ["machine", "source", "kind", "section", "variant",
                  "ranks", "threads", "omp_threads", "size", "frames",
                  "workload_s", "min_ms", "mean_ms", "median_ms",
                  "stddev_ms", "speedup"]
MEMORY_COLUMNS = ["machine", "source", "kind", "strategy", "ranks", "frames",
                  "workload_s", "min_mib", "max_mib", "avg_mib"]


def read_rows(path: Path, machine: str, source: str) -> list[dict]:
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["machine"] = machine
        r["source"] = source
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", type=Path,
                    help="directories holding <machine>_*.csv result files")
    ap.add_argument("-o", "--out", type=Path, default=Path("results/analysis/csv"),
                    help="output directory (default: results/analysis/csv/)")
    args = ap.parse_args()

    timings: list[dict] = []
    memory: list[dict] = []
    seen = 0

    for d in args.dirs:
        if not d.is_dir():
            print(f"skipping {d}: not a directory", file=sys.stderr)
            continue
        for path in sorted(d.glob("*.csv")):
            # "<machine>_<source>.csv", e.g. "cluster_scaling.csv". The
            # machine never contains an underscore (run_suite.sh sets it to
            # cluster/github/local or an explicit PREFIX), so the first one
            # is always the split point.
            machine, _, source = path.stem.partition("_")
            for row in read_rows(path, machine, source):
                (timings if row.get("kind") == "timing" else memory).append(row)
            seen += 1

    if not timings and not memory:
        print("no measurements found — are these result directories?",
              file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows, columns in (("timings.csv", timings, TIMING_COLUMNS),
                                ("memory.csv", memory, MEMORY_COLUMNS)):
        if not rows:
            continue
        with (args.out / name).open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"{args.out / name}: {len(rows)} rows")

    print(f"parsed {seen} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
