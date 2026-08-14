#!/usr/bin/env python3
"""Parallel-performance figures for the AMSC_STFT report.

Reads only results/csv/timings.csv and memory.csv — never the text tables — so a
change to the C++ printing is absorbed by parse_results.py and never reaches a
figure.

    python3 analysis/parse_results.py results/raw/*/ -o results/csv/
    python3 analysis/plot_results.py --csv results/csv/ --out results/figures/

  1  fft_algorithms  FFT algorithms as OpenMP threads grow — time AND speedup
  2  stft_openmp     STFT on OpenMP alone (1 MPI rank), speedup
  3  stft_mpi        STFT on MPI alone (1 thread per rank), speedup
  4  stft_hybrid     STFT with 2 threads per rank, speedup against rank count
  5  memory          broadcast vs scatter, mean per-rank footprint vs ranks
  6  weak_scaling    constant work per rank: time should stay flat

Each figure carries a parameter box to its right listing everything held fixed,
so a figure lifted into a slide still says what it measured.

── Speedup baselines ───────────────────────────────────────────────────────
Every curve is divided by ITS OWN one-worker time, never by another series.
Dividing a parallel engine by a different serial algorithm mixes the algorithmic
win with the parallel one and yields speedups above the ideal line — the usual
reason a scaling plot looks impossible.

Figure 4 keeps two OpenMP threads on every rank, so its ideal line is 2 x ranks,
not ranks: with a fixed thread count the honest reference has to count the
threads too, otherwise the measured curve would sail past "ideal" by design.

Flags: --machine NAME (repeatable) · --stat min|median|mean · --only a,b
       --format pdf,png

Every figure writes a .csv beside it with the numbers plotted.
Requires matplotlib only — no pandas, so it runs in the course container.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Categorical slots in validated order; colour follows the ENTITY (an algorithm,
# a workload), never its position, so dropping a series never repaints the rest.
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
        "#008300", "#4a3aa7", "#e34948"]
INK, INK_SOFT, GRID, REFERENCE = "#0b0b0b", "#52514e", "#dcdbd7", "#8c8b85"

ALGO_COLOR = {"RecursiveFFT": SLOT[0], "IterativeFFT": SLOT[1],
              "ParallelFFT": SLOT[2]}
STRATEGY_COLOR = {"broadcast": SLOT[0], "scatter": SLOT[1]}

FRAME, HOP, RATE = 1024, 512, 44100


def style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.size": 10, "axes.labelsize": 10, "axes.labelcolor": INK_SOFT,
        "axes.edgecolor": GRID, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 0.7, "grid.linestyle": "-",
        "xtick.color": INK_SOFT, "ytick.color": INK_SOFT,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 9, "legend.frameon": True,
        "legend.framealpha": 1.0, "legend.edgecolor": GRID,
        "lines.linewidth": 2.0, "lines.markersize": 6.5,
    })


# ── Data ────────────────────────────────────────────────────────────────────
def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = list(csv.DictReader(path.open()))
    for r in rows:
        for k, v in list(r.items()):
            if v == "":
                r[k] = None
            elif k in ("ranks", "threads", "omp_threads", "workers", "size",
                       "frames"):
                r[k] = int(float(v))
            elif k.endswith(("_ms", "_mib", "_s")) or k == "speedup_reported":
                r[k] = float(v)
    return rows


def seconds_of(frames: int) -> float:
    """Frames back to the audio duration that produced them."""
    return ((frames - 1) * HOP + FRAME) / RATE


def workload_label(frames: int) -> str:
    return f"{seconds_of(frames):.0f} s"


def curves_by_size(rows, machine, stat, section, keep, worker_of, source=None):
    """{frames: {workers: time}} for the matching variant, plus {frames: serial}.

    `source` restricts the rows to one sweep file. Without it the plain
    benchmark section and the rank sweep both contribute "MPI pure" rows at the
    same workload, and the figure would silently mix two different experiments.
    """
    data: dict[int, dict[int, float]] = defaultdict(dict)
    serial: dict[int, float] = {}
    for r in rows:
        if r["machine"] != machine or r["section"] != section:
            continue
        if source is not None and r.get("source") != source:
            continue
        v, size, t = r["variant"], r["size"], r[stat]
        if v.startswith("serial") or "1 thread" in v:
            if size not in serial or t < serial[size]:
                serial[size] = t
            continue
        if not keep(r):
            continue
        w = worker_of(r)
        if not w:
            continue
        prev = data[size].get(w)
        data[size][w] = t if prev is None else min(prev, t)
    return data, serial


# ── Drawing ─────────────────────────────────────────────────────────────────
def param_box(fig, params) -> None:
    """The fixed configuration, listed to the right of the axes.

    A run-on subtitle under the title is read once and forgotten; an aligned list
    beside the plot is read the way a legend is, and survives being cropped into
    a slide. Two columns keep each key next to its own value.
    """
    y = 0.90
    fig.text(0.785, 0.955, "Configuration", fontsize=9.5, color=INK,
             weight="medium", va="top")
    for key, value in params:
        if key or value:
            fig.text(0.785, y, key, fontsize=8.6, color=INK_SOFT, va="top")
            fig.text(0.90, y, str(value), fontsize=8.6, color=INK, va="top")
        y -= 0.042


def sort_key_factory(order):
    """Numeric when the label starts with a number ("5 s" before "15 s")."""
    def key(item):
        head = item[0].split()[0]
        try:
            return (0, float(head))
        except ValueError:
            return (1, order.index(item[0]) if item[0] in order else 99)
    return key


def finish(fig, out: Path, formats, header, table):
    out.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(out.with_suffix(f".{fmt}"))
    plt.close(fig)
    with out.with_suffix(".csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(table)
    print(f"  wrote {out.name}: {', '.join(formats)} + csv")


def scaling_figure(series, baselines, title, params, xlabel, out, formats,
                   header, colors=None, ideal_factor=1, with_time=False,
                   time_label="execution time [ms]"):
    """Speedup against an ideal line; optionally an execution-time panel above."""
    series = {k: v for k, v in series.items() if len(v) >= 2}
    if baselines:
        series = {k: v for k, v in series.items() if k in baselines}
    if not series:
        print(f"  skip {out.name}: no curve with two points and a baseline")
        return

    if with_time:
        fig, (ax_t, ax_s) = plt.subplots(2, 1, figsize=(9.0, 7.4), sharex=True,
                                         gridspec_kw={"hspace": 0.15})
    else:
        fig, ax_s = plt.subplots(figsize=(9.0, 4.8))
        ax_t = None
    fig.subplots_adjust(right=0.74)

    order = list(colors or {})
    all_x, table = set(), []
    for i, (label, pts) in enumerate(sorted(series.items(),
                                            key=sort_key_factory(order))):
        xs = sorted(pts)
        base = (baselines or {}).get(label) or (pts[1] if 1 in pts else None)
        if base is None:
            print(f"  skip curve {label!r} in {out.name}: no one-worker baseline")
            continue
        colour = (colors or {}).get(label, SLOT[i % len(SLOT)])
        sp = [base / pts[x] for x in xs]
        all_x.update(xs)
        ax_s.plot(xs, sp, color=colour, marker="o", markeredgecolor="white",
                  markeredgewidth=1.2, label=label, zorder=3)
        if ax_t is not None:
            ax_t.plot(xs, [pts[x] for x in xs], color=colour, marker="o",
                      markeredgecolor="white", markeredgewidth=1.2,
                      label=label, zorder=3)
        table += [[label, x, round(pts[x], 3), round(s, 3)]
                  for x, s in zip(xs, sp)]

    xs = sorted(all_x)
    ideal = [x * ideal_factor for x in xs]
    ax_s.plot(xs, ideal, color=REFERENCE, linewidth=1.3, linestyle="--",
              label="ideal", zorder=1)

    for ax in ([ax_s] if ax_t is None else [ax_t, ax_s]):
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax_s.set_yscale("log", base=2)
    ax_s.set_yticks(ideal)
    ax_s.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax_s.set_ylabel("speedup   $t_1 / t_p$")
    ax_s.set_xlabel(xlabel)
    ax_s.legend(loc="upper left")

    if ax_t is not None:
        ax_t.set_yscale("log")
        ax_t.set_ylabel(time_label)
        ax_t.set_title(title, color=INK, loc="left", fontsize=12.5, pad=12)
    else:
        ax_s.set_title(title, color=INK, loc="left", fontsize=12.5, pad=12)

    param_box(fig, params)
    finish(fig, out, formats, header, table)


# ── Figures ─────────────────────────────────────────────────────────────────
def workload_params(frames_seen):
    """Seconds label the curves; the frame counts are the parallel work items."""
    if not frames_seen:
        return []
    return [("workloads [s]",
             " / ".join(f"{seconds_of(f):.0f}" for f in sorted(frames_seen))),
            ("frames", " / ".join(str(f) for f in sorted(frames_seen)))]


def fig_fft(rows, machine, stat, out, formats):
    per_algo: dict[str, dict[int, float]] = defaultdict(dict)
    sizes = {r["size"] for r in rows
             if r["machine"] == machine and r["section"] == "fft"
             and r.get("source") == "scaling_threads"}
    if not sizes:
        print("  skip fft_algorithms: no FFT rows")
        return
    biggest = max(sizes)
    for r in rows:
        if (r["machine"] != machine or r["section"] != "fft"
                or r.get("source") != "scaling_threads"
                or r["size"] != biggest or not r.get("omp_threads")):
            continue
        t = r["omp_threads"]
        prev = per_algo[r["variant"]].get(t)
        per_algo[r["variant"]][t] = r[stat] if prev is None else min(prev, r[stat])

    params = [("machine", machine),
              ("statistic", stat[:-3]),
              ("transform size", f"{biggest} samples"),
              ("MPI ranks", 1),
              ("baseline", "each algorithm"),
              ("", "at 1 thread"),
              ("", ""),
              ("note", "Recursive and"),
              ("", "Iterative are"),
              ("", "serial: threads"),
              ("", "cannot change"),
              ("", "their time")]
    scaling_figure(per_algo, None,
                   "FFT algorithms: scaling with OpenMP threads",
                   params, "OpenMP threads", out, formats,
                   ["algorithm", "threads", stat, "speedup"],
                   colors=ALGO_COLOR, with_time=True)


def fig_stft(rows, machine, stat, out, formats, mode):
    if mode == "openmp":
        section = "stft"
        keep = lambda r: r["variant"].startswith("STFT OpenMP")   # noqa: E731
        worker = lambda r: r["threads"]                           # noqa: E731
        title, xlabel, factor = "STFT: OpenMP thread scaling", "OpenMP threads", 1
        extra = [("MPI ranks", 1),
                 ("FFT", "IterativeFFT"),
                 ("window", "Hann window"),
                 ("", ""),
                 ("note", "a single process,"),
                 ("", "no MPI, no"),
                 ("", "communication")]
    elif mode == "mpi":
        section = "mpi"
        keep = lambda r: r["variant"].startswith("MPI pure")      # noqa: E731
        worker = lambda r: r["ranks"]                             # noqa: E731
        title, xlabel, factor = "STFT: MPI rank scaling", "MPI ranks", 1
        extra = [("threads / rank", 1),
                 ("FFT", "IterativeFFT"),
                 ("window", "Hann window"),
                 ("", ""),
                 ("note", "no OpenMP: every"),
                 ("", "bit of parallelism"),
                 ("", "comes from ranks")]
    else:
        section = "mpi"
        keep = lambda r: r["variant"].startswith("hybrid")        # noqa: E731
        worker = lambda r: r["ranks"]                             # noqa: E731
        title, xlabel, factor = ("STFT: hybrid MPI + OpenMP scaling",
                                 "MPI ranks", 2)
        extra = [("threads / rank", 2),
                 ("FFT", "IterativeFFT"),
                 ("window", "Hann window"),
                 ("ideal line", "2 x ranks"),
                 ("", ""),
                 ("note", "the thread count"),
                 ("", "is fixed, so the"),
                 ("", "x axis counts"),
                 ("", "ranks only")]

    src = "scaling_threads" if mode == "openmp" else "scaling"
    data, serial = curves_by_size(rows, machine, stat, section, keep, worker,
                                  source=src)
    if not data:
        print(f"  skip {out.name}: no rows for this configuration")
        return
    series = {workload_label(s): pts for s, pts in data.items()}
    bases = {workload_label(s): t for s, t in serial.items() if s in data}
    params = ([("machine", machine), ("statistic", stat[:-3]),
               ("frame / hop", f"{FRAME} / {HOP}"),
               ("sample rate", f"{RATE / 1000:g} kHz")]
              + workload_params(set(data)) + extra)
    scaling_figure(series, bases, title, params, xlabel, out, formats,
                   ["workload", "ranks" if mode != "openmp" else "threads",
                    stat, "speedup"],
                   ideal_factor=factor)


def fig_memory(mem, machine, out, formats):
    """Mean per-rank footprint, one bar pair per rank count.

    The mean is the statistic that answers what the scatter exists to answer —
    whether the job still fits in memory — while MAX is dominated by the root
    rank, which holds the signal it read under either strategy, and MIN
    describes the single leanest rank rather than the job.
    """
    rss = defaultdict(dict)
    for r in mem:
        if r["machine"] == machine and r["kind"] == "rss" and r.get("ranks"):
            rss[r["strategy"]][r["ranks"]] = r["avg_mib"]
    if not {"bcast", "scatter"} <= set(rss):
        print(f"  skip memory: need both strategies measured for {machine}")
        return

    ranks = sorted(set(rss["bcast"]) & set(rss["scatter"]))
    if not ranks:
        print("  skip memory: strategies not measured at the same rank counts")
        return

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    fig.subplots_adjust(right=0.74)
    width, xs, table = 0.34, range(len(ranks)), []
    for i, (key, label) in enumerate((("bcast", "broadcast"),
                                      ("scatter", "scatter"))):
        vals = [rss[key][p] for p in ranks]
        offs = [x + (i - 0.5) * (width + 0.02) for x in xs]
        bars = ax.bar(offs, vals, width, label=label,
                      color=STRATEGY_COLOR[label], zorder=3)
        ax.bar_label(bars, fmt="%.0f", padding=3, fontsize=8.5, color=INK_SOFT)
        table += [[label, p, round(v, 2)] for p, v in zip(ranks, vals)]

    ax.set_xticks(list(xs))
    ax.set_xticklabels([str(p) for p in ranks])
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("mean resident memory per rank [MiB]")
    ax.margins(y=0.18)
    ax.legend(loc="upper right")
    ax.set_title("Input distribution: broadcast vs scatter",
                 color=INK, loc="left", fontsize=12.5, pad=12)

    param_box(fig, [("machine", machine),
                    ("metric", "peak RSS"),
                    ("", "(VmHWM), mean"),
                    ("", "over the ranks"),
                    ("threads / rank", 2),
                    ("", ""),
                    ("note", "broadcast keeps"),
                    ("", "the whole signal"),
                    ("", "on every rank;"),
                    ("", "scatter keeps"),
                    ("", "about 1/P of it"),
                    ("", ""),
                    ("", "the root holds"),
                    ("", "what it read under"),
                    ("", "either strategy")])
    finish(fig, out, formats, ["strategy", "ranks", "mean_mib"], table)


def fig_weak(rows, machine, stat, out, formats):
    """Constant work per rank: the time should stay flat, the efficiency at 1.

    Speedup is meaningless here — the problem is not fixed — so this figure
    plots the time itself and the weak-scaling efficiency t(1)/t(P), which is
    the fraction of the ideal flat line that survives.
    """
    pts, base = {}, None
    for r in rows:
        if (r["machine"] != machine or r.get("source") != "weak_scaling"
                or r["section"] != "mpi"):
            continue
        if r["variant"].startswith("serial") and r["ranks"] == 1:
            continue
        if not r["variant"].startswith("MPI pure") or not r["ranks"]:
            continue
        p = r["ranks"]
        pts[p] = min(pts.get(p, r[stat]), r[stat])
    if len(pts) < 2:
        print(f"  skip {out.name}: weak-scaling sweep not in the data")
        return
    base = pts[min(pts)]

    fig, (ax_t, ax_e) = plt.subplots(2, 1, figsize=(9.0, 7.4), sharex=True,
                                     gridspec_kw={"hspace": 0.15})
    fig.subplots_adjust(right=0.74)
    xs = sorted(pts)
    times = [pts[p] for p in xs]
    eff = [base / pts[p] for p in xs]

    ax_t.plot(xs, times, color=SLOT[0], marker="o", markeredgecolor="white",
              markeredgewidth=1.2, label="measured", zorder=3)
    ax_t.axhline(base, color=REFERENCE, linewidth=1.3, linestyle="--",
                 label="ideal (flat)", zorder=1)
    ax_e.plot(xs, eff, color=SLOT[0], marker="o", markeredgecolor="white",
              markeredgewidth=1.2, label="measured", zorder=3)
    ax_e.axhline(1.0, color=REFERENCE, linewidth=1.3, linestyle="--",
                 label="ideal (1.0)", zorder=1)

    for ax in (ax_t, ax_e):
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.legend(loc="best")
    ax_t.set_ylabel("execution time [ms]")
    ax_e.set_ylabel("weak-scaling efficiency")
    ax_e.set_xlabel("MPI ranks")
    ax_t.set_title("STFT: weak scaling (constant work per rank)",
                   color=INK, loc="left", fontsize=12.5, pad=12)

    param_box(fig, [("machine", machine),
                    ("statistic", stat[:-3]),
                    ("threads / rank", 1),
                    ("FFT", "IterativeFFT"),
                    ("window", "Hann window"),
                    ("frame / hop", f"{FRAME} / {HOP}"),
                    ("", ""),
                    ("note", "the audio grows"),
                    ("", "with the rank"),
                    ("", "count, so each"),
                    ("", "rank always does"),
                    ("", "the same work"),
                    ("", ""),
                    ("", "a flat line means"),
                    ("", "perfect weak"),
                    ("", "scaling; the slope"),
                    ("", "is communication")])
    finish(fig, out, formats, ["ranks", stat, "efficiency"],
           [[p, round(t, 3), round(e, 3)] for p, t, e in zip(xs, times, eff)])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=Path("results/csv"))
    ap.add_argument("--out", type=Path, default=Path("results/figures"))
    ap.add_argument("--machine", action="append", default=None)
    ap.add_argument("--stat", default="min", choices=["min", "median", "mean"])
    ap.add_argument("--only", default=None,
                    help="comma list: fft_algorithms, stft_openmp, stft_mpi, "
                         "stft_hybrid, memory, weak")
    ap.add_argument("--format", default="pdf,png")
    args = ap.parse_args()

    stat = f"{args.stat}_ms"
    formats = [f.strip() for f in args.format.split(",") if f.strip()]
    wanted = ({w.strip() for w in args.only.split(",")} if args.only else
              {"fft_algorithms", "stft_openmp", "stft_mpi", "stft_hybrid",
               "memory", "weak"})

    timings = load(args.csv / "timings.csv")
    memory = load(args.csv / "memory.csv")
    if not timings and not memory:
        print(f"no CSVs in {args.csv} — run parse_results.py first", file=sys.stderr)
        return 1

    machines = args.machine or sorted({r["machine"] for r in timings} |
                                      {r["machine"] for r in memory})
    style()
    print(f"machines: {', '.join(machines)}   statistic: {args.stat}")

    for m in machines:
        print(f"[{m}]")
        if "fft_algorithms" in wanted:
            fig_fft(timings, m, stat, args.out / f"{m}_1_fft_algorithms", formats)
        if "stft_openmp" in wanted:
            fig_stft(timings, m, stat, args.out / f"{m}_2_stft_openmp",
                     formats, "openmp")
        if "stft_mpi" in wanted:
            fig_stft(timings, m, stat, args.out / f"{m}_3_stft_mpi",
                     formats, "mpi")
        if "stft_hybrid" in wanted:
            fig_stft(timings, m, stat, args.out / f"{m}_4_stft_hybrid",
                     formats, "hybrid")
        if "memory" in wanted:
            fig_memory(memory, m, args.out / f"{m}_5_memory", formats)
        if "weak" in wanted:
            fig_weak(timings, m, stat, args.out / f"{m}_6_weak_scaling", formats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
