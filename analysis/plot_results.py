#!/usr/bin/env python3
"""Parallel-performance figures for the AMSC_STFT report.

Reads only results/results_analysis/csv/timings.csv and memory.csv — never
the raw benchmark output — so a change to the C++ output format only ever
requires a fix in parse_results.py, never here.

    python3 analysis/parse_results.py
    python3 analysis/plot_results.py

Both read from and write to their default locations under results/; pass
`--csv`/`--out` (see below) only to point at something else.

  1  fft_algorithms  FFT algorithms as OpenMP threads grow — time AND speedup
  2  stft_openmp     STFT on OpenMP alone (1 MPI rank), speedup
  3  stft_mpi        STFT on MPI alone (1 thread per rank), speedup
  4  stft_hybrid     STFT with 2 threads per rank, speedup against rank count
  5  memory          broadcast vs scatter, mean per-rank footprint vs ranks

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

Writes one PNG per figure. Requires matplotlib only — no pandas, so it runs
in the course container.
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

# The sweep whose rows figure 5 reads (run_suite.sh writes one file per
# section, and parse_results.py carries the name through as "source").
MEMORY_SOURCE = "memory_bcast_vs_scatter"


def style() -> None:
    """Sized for a projector, not for a screen at arm's length.

    Every figure here is meant to be shown in a presentation, where the reader
    is metres away: fonts and strokes that look right in an editor are
    unreadable on a wall. Larger type and thicker lines cost nothing on paper
    and are the difference between a slide that lands and one that does not.
    """
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.size": 13,
        "axes.labelsize": 13, "axes.labelcolor": INK_SOFT,
        "axes.edgecolor": GRID, "axes.linewidth": 1.1,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 0.9, "grid.linestyle": "-",
        "xtick.color": INK_SOFT, "ytick.color": INK_SOFT,
        "xtick.labelsize": 12, "ytick.labelsize": 12,
        "legend.fontsize": 12, "legend.frameon": True,
        "legend.framealpha": 1.0, "legend.edgecolor": GRID,
        "lines.linewidth": 3.0, "lines.markersize": 9,
    })


# ── Data ────────────────────────────────────────────────────────────────────
def load(path: Path) -> list[dict]:
    """Read one of parse_results.py's output CSVs into a list of row dicts.

    Numeric columns are converted to int/float per the schema; blank fields
    (a column that does not apply to this row's `kind`) become None rather
    than an empty string, so later code can test for absence with `is None`.
    """
    if not path.exists():
        return []
    rows = list(csv.DictReader(path.open()))
    for r in rows:
        for k, v in list(r.items()):
            if v == "":
                r[k] = None
            elif k in ("ranks", "threads", "omp_threads", "size", "frames"):
                r[k] = int(float(v))
            elif k.endswith(("_ms", "_mib", "_s")) or k == "speedup":
                r[k] = float(v)
    return rows


def seconds_of(frames: int) -> float:
    """Frames back to the audio duration that produced them."""
    return ((frames - 1) * HOP + FRAME) / RATE


def workload_label(frames: int) -> str:
    """Legend label for a workload, e.g. "5 s"."""
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
        v, frames, t = r["variant"], r["frames"], r[stat]
        if v.startswith("serial") or "1 thread" in v:
            if frames not in serial or t < serial[frames]:
                serial[frames] = t
            continue
        if not keep(r):
            continue
        w = worker_of(r)
        if not w:
            continue
        prev = data[frames].get(w)
        data[frames][w] = t if prev is None else min(prev, t)
    return data, serial


# ── Drawing ─────────────────────────────────────────────────────────────────
def param_box(fig, params) -> None:
    """The fixed configuration, listed to the right of the axes.

    A run-on subtitle under the title is read once and forgotten; an aligned list
    beside the plot is read the way a legend is, and survives being cropped into
    a slide. Two columns keep each key next to its own value.
    """
    y = 0.895
    fig.text(0.785, 0.965, "Configuration", fontsize=12, color=INK,
             weight="medium", va="top")
    for key, value in params:
        if key or value:
            fig.text(0.785, y, key, fontsize=11, color=INK_SOFT, va="top")
            fig.text(0.905, y, str(value), fontsize=11, color=INK, va="top")
        y -= 0.050


def sort_key_factory(order):
    """Numeric when the label starts with a number ("5 s" before "15 s")."""
    def key(item):
        head = item[0].split()[0]
        try:
            return (0, float(head))
        except ValueError:
            return (1, order.index(item[0]) if item[0] in order else 99)
    return key


def finish(fig, out: Path) -> None:
    """Save `fig` as `out`.png, close it, and confirm on stdout."""
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"  wrote {out.with_suffix('.png').name}")


def scaling_figure(series, baselines, title, params, xlabel, out,
                   colors=None, ideal_factor=1, with_time=False, metric="speedup",
                   time_label="execution time [ms]"):
    """Speedup against an ideal line; optionally an execution-time panel above."""
    series = {k: v for k, v in series.items() if len(v) >= 2}
    if baselines:
        series = {k: v for k, v in series.items() if k in baselines}
    if not series:
        print(f"  skip {out.name}: no curve with two points and a baseline")
        return

    if with_time:
        fig, (ax_t, ax_s) = plt.subplots(2, 1, figsize=(11.0, 8.4), sharex=True,
                                         gridspec_kw={"hspace": 0.15})
    else:
        fig, ax_s = plt.subplots(figsize=(11.0, 5.6))
        ax_t = None
    fig.subplots_adjust(right=0.74)

    order = list(colors or {})
    all_x = set()
    for i, (label, pts) in enumerate(sorted(series.items(),
                                            key=sort_key_factory(order))):
        xs = sorted(pts)
        base = (baselines or {}).get(label) or (pts[1] if 1 in pts else None)
        if base is None:
            print(f"  skip curve {label!r} in {out.name}: no one-worker baseline")
            continue
        colour = (colors or {}).get(label, SLOT[i % len(SLOT)])
        sp = [base / pts[x] for x in xs]
        if metric == "efficiency":
            # Speedup divided by the workers it used. Near the origin every
            # curve hugs the ideal line and they overlap into one stroke;
            # dividing by the worker count flattens the ideal to a constant
            # and lets the differences between workloads occupy the whole
            # height of the plot instead of a sliver of it.
            sp = [v / (x * ideal_factor) for v, x in zip(sp, xs)]
        all_x.update(xs)
        ax_s.plot(xs, sp, color=colour, marker="o", markeredgecolor="white",
                  markeredgewidth=1.2, label=label, zorder=3)
        if ax_t is not None:
            ax_t.plot(xs, [pts[x] for x in xs], color=colour, marker="o",
                      markeredgecolor="white", markeredgewidth=1.2,
                      label=label, zorder=3)

    xs = sorted(all_x)
    ideal = [x * ideal_factor for x in xs]
    ax_s.plot(xs, ideal, color=REFERENCE, linewidth=1.3, linestyle="--",
              label="ideal", zorder=1)

    for ax in ([ax_s] if ax_t is None else [ax_t, ax_s]):
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    if metric == "efficiency":
        # Linear, and anchored at 1: efficiency lives in [0, 1] and a log axis
        # would compress exactly the range the reader is looking at.
        ax_s.set_ylim(0, 1.08)
        ax_s.set_ylabel("parallel efficiency   $t_1 / (p \\, t_p)$")
    else:
        ax_s.set_yscale("log", base=2)
        ax_s.set_yticks(ideal)
        ax_s.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax_s.set_ylabel("speedup   $t_1 / t_p$")
    ax_s.set_xlabel(xlabel)
    # Efficiency curves start high on the left and fall away, so the bottom-left
    # is the free corner; speedup curves do the opposite.
    ax_s.legend(loc="lower left" if metric == "efficiency" else "upper left")

    if ax_t is not None:
        ax_t.set_yscale("log")
        ax_t.set_ylabel(time_label)
        ax_t.set_title(title, color=INK, loc="left", fontsize=15.5, pad=14)
    else:
        ax_s.set_title(title, color=INK, loc="left", fontsize=15.5, pad=14)

    param_box(fig, params)
    finish(fig, out)


# ── Figures ─────────────────────────────────────────────────────────────────
def workload_params(frames_seen):
    """Seconds label the curves; the frame counts are the parallel work items."""
    if not frames_seen:
        return []
    return [("workloads [s]",
             " / ".join(f"{seconds_of(f):.0f}" for f in sorted(frames_seen))),
            ("frames", " / ".join(str(f) for f in sorted(frames_seen)))]


def fig_fft(rows, machine, stat, out):
    """Figure 1: the three FFT algorithms, time and speedup vs OpenMP threads.

    Uses the largest transform size measured, since that is where threading
    has the most work to hide its own overhead behind.
    """
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
                   params, "OpenMP threads", out,
                   colors=ALGO_COLOR, with_time=True)


def fig_stft(rows, machine, stat, out, mode, metric="speedup"):
    """Figures 2-4: STFT speedup under one parallelism mode.

    `mode` selects which rows and which worker count define the curve:
    "openmp" (OpenMP threads, single rank), "mpi" (MPI ranks, one thread
    each), or "hybrid" (MPI ranks, two threads each — the ideal line is
    2 x ranks, not ranks, to account for the fixed thread count).
    """
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
    scaling_figure(series, bases, title, params, xlabel, out,
                   ideal_factor=factor, metric=metric)


def fig_memory(mem, machine, out):
    """Figure 5: what the input distribution costs each rank, two ways.

    Upper panel, the ANALYTIC input footprint: how many samples one rank has to
    hold under each strategy. It is exact arithmetic on the frame layout, not a
    measurement, and it is the quantity Distribution actually controls —
    broadcast leaves every rank holding the whole signal however many ranks are
    added, scatter leaves it about 1/P. This is the claim the feature makes, and
    the panel is on a log axis because a curve that falls by a factor of 191 has
    no other honest shape.

    Lower panel, the MEASURED peak resident memory of the whole process. It does
    NOT fall by 191, and the gap between the panels is the point rather than a
    disappointment: VmHWM includes the complete spectrogram that MPI_Gatherv
    assembles on the root rank, and the MPI runtime's own buffers, neither of
    which shrinks when ranks are added. The input bottleneck is removed; the
    output one is not, and the two panels together say exactly that.

    The mean is the statistic in the lower panel because it answers what the
    scatter exists to answer — whether the job still fits in memory — while MAX
    is dominated by the root rank, which holds what it read under either
    strategy, and MIN describes the leanest rank rather than the job.
    """
    rss = defaultdict(dict)
    analytic = defaultdict(dict)
    workloads: set[float] = set()
    for r in mem:
        # Only the dedicated bcast-vs-scatter sweep: three other sections of
        # run_suite.sh also emit memory_rss scatter rows, and without this
        # filter the last one read silently replaces the scatter series with a
        # different experiment's — the two strategies then no longer agree even
        # at one rank, where by construction they must.
        if r["machine"] != machine or r.get("source") != MEMORY_SOURCE:
            continue
        if not r.get("ranks"):
            continue
        if r["kind"] == "memory_rss":
            rss[r["strategy"]][r["ranks"]] = r["avg_mib"]
        elif r["kind"] == "memory_analytic":
            analytic[r["strategy"]][r["ranks"]] = r["avg_mib"]
            # Only the analytic rows carry the workload: memory_rss is a
            # high-water mark of the process, not of one measured signal, so
            # the C++ writer leaves its workload column empty. Reading it only
            # here is why the configuration list can name the audio at all.
            if r.get("workload_s"):
                workloads.add(float(r["workload_s"]))

    if not {"bcast", "scatter"} <= set(rss):
        print(f"  skip memory: need both strategies measured for {machine}")
        return
    ranks = sorted(set(rss["bcast"]) & set(rss["scatter"]))
    if not ranks:
        print("  skip memory: strategies not measured at the same rank counts")
        return
    have_analytic = {"bcast", "scatter"} <= set(analytic) and all(
        p in analytic["bcast"] and p in analytic["scatter"] for p in ranks)

    style()
    if have_analytic:
        fig, (ax_a, ax) = plt.subplots(2, 1, figsize=(11.0, 8.4), sharex=True,
                                       gridspec_kw={"hspace": 0.18})
    else:
        fig, ax = plt.subplots(figsize=(11.0, 5.6))
        ax_a = None
    fig.subplots_adjust(right=0.74)

    xs = range(len(ranks))
    width = 0.34

    if ax_a is not None:
        for strategy, offset in (("bcast", -0.5), ("scatter", 0.5)):
            ys = [analytic[strategy][p] for p in ranks]
            ax_a.plot(xs, ys, color=STRATEGY_COLOR[
                          "broadcast" if strategy == "bcast" else "scatter"],
                      marker="o", markeredgecolor="white", markeredgewidth=1.2,
                      linewidth=3.0, markersize=9,
                      label="broadcast" if strategy == "bcast" else "scatter")
        ax_a.set_yscale("log", base=10)
        ax_a.set_ylabel("input held per rank [MiB]")
        ax_a.legend(loc="center left")
        ax_a.set_title("Input distribution: broadcast vs scatter",
                       color=INK, loc="left", fontsize=15.5, pad=14)
        best = ranks[-1]
        ratio = analytic["bcast"][best] / analytic["scatter"][best]
        # Parked in the empty band between the two curves rather than beside
        # the last point, where it landed on the scatter line itself.
        ax_a.text(0.985, 0.30, f"{ratio:.0f}x less at {best} ranks",
                  transform=ax_a.transAxes, fontsize=11.5, color=INK_SOFT,
                  ha="right", va="center")

    for strategy, offset, name in (("bcast", -width / 2, "broadcast"),
                                   ("scatter", width / 2, "scatter")):
        ys = [rss[strategy][p] for p in ranks]
        bars = ax.bar([x + offset for x in xs], ys, width,
                      color=STRATEGY_COLOR[name], label=name, zorder=3)
        ax.bar_label(bars, fmt="%.0f", fontsize=10, color=INK_SOFT, padding=2)

    ax.set_ylabel("mean resident memory per rank [MiB]")
    ax.set_xlabel("MPI ranks")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([str(p) for p in ranks])
    ax.margins(y=0.18)
    if ax_a is None:
        ax.legend(loc="upper right")
        ax.set_title("Input distribution: broadcast vs scatter",
                     color=INK, loc="left", fontsize=15.5, pad=14)
    else:
        ax.set_title("measured peak RSS of the whole process",
                     color=INK_SOFT, loc="left", fontsize=12, pad=8)

    audio = ("/".join(f"{w:g}" for w in sorted(workloads)) if workloads
             else "unknown")
    param_box(fig, [("machine", machine),
                    ("audio [s]", audio),
                    ("", ""),
                    ("upper", "exact arithmetic:"),
                    ("", "input samples"),
                    ("", "one rank holds"),
                    ("", ""),
                    ("lower", "measured VmHWM,"),
                    ("", "mean over ranks."),
                    ("", "It also holds the"),
                    ("", "gathered output,"),
                    ("", "which no rank"),
                    ("", "count shrinks")])
    finish(fig, out)


HYBRID_SPLIT_SOURCE = "hybrid_split"

# Cores per socket and per node of the machine figure 9 describes, used only to
# annotate where a rank stops fitting in one NUMA domain. Galileo100's Xeon 8260
# is 2 x 24. A machine with a different layout needs these changed; a split that
# matches neither is simply not annotated, so a wrong value cannot invent a
# boundary that is not there.
CORES_PER_SOCKET, CORES_PER_NODE = 24, 48


def fig_hybrid_split(rows, machine, stat, out):
    """Figure 9: the rank/thread split, at a fixed number of cores.

    Every point here spends the SAME workers — the sweep divides one machine
    into P ranks of T threads with P x T held constant — so this is not a
    scaling curve and its ideal is not a diagonal. Perfect use of the cores
    would put every split on one horizontal line at that worker count, and the
    distance below it is what the decomposition costs. The x axis is threads
    per rank because that is the knob; the rank count each one implies is
    printed under it, since a reader thinking in ranks should not have to
    divide.

    Two vertical marks carry the hardware. A rank of 24 threads is exactly one
    socket of this machine; a rank of 48 spans both, so its threads reach memory
    attached to the other socket. If the curve turns between those two marks,
    the turn is NUMA, not MPI.
    """
    pts, serial = {}, None
    for r in rows:
        if r["machine"] != machine or r.get("source") != HYBRID_SPLIT_SOURCE:
            continue
        if r.get("section") != "mpi" or not r.get(stat):
            continue
        variant, t = r.get("variant", ""), float(r[stat])
        if variant.startswith("serial"):
            serial = t if serial is None else min(serial, t)
        elif variant.startswith("hybrid") and r.get("ranks") and r.get("threads"):
            threads, ranks = int(r["threads"]), int(r["ranks"])
            # One launch per split, so a repeated key means a repeated point:
            # keep the faster, matching the "min" convention of the other rows.
            if threads not in pts or t < pts[threads][0]:
                pts[threads] = (t, ranks)

    if len(pts) < 2 or not serial:
        print(f"  skip {out.name}: need a serial baseline and two splits")
        return

    xs      = sorted(pts)
    speedup = [serial / pts[t][0] for t in xs]
    workers = max(t * pts[t][1] for t in xs)

    style()
    fig, ax = plt.subplots(figsize=(11.0, 5.6))
    fig.subplots_adjust(right=0.74)

    # Evenly spaced rather than log-scaled: these are seven discrete
    # configurations, not samples of a continuous variable, and on a log axis
    # the 8- and 12-thread splits fall so close together that their labels
    # collide. Position is an index; the value is on the tick.
    pos = list(range(len(xs)))

    # Socket and node boundaries, drawn before the data so they sit behind it.
    for threads, name in ((CORES_PER_SOCKET, "one socket"),
                          (CORES_PER_NODE, "whole node")):
        if threads in pts:
            ax.axvline(pos[xs.index(threads)], color=GRID, linewidth=1.6,
                       zorder=0)
            ax.text(pos[xs.index(threads)], 0.02, f" {name}",
                    transform=ax.get_xaxis_transform(), fontsize=10,
                    color=INK_SOFT, rotation=90, va="bottom")

    ax.axhline(workers, color=REFERENCE, linestyle="--", linewidth=2.0,
               label=f"ideal ({workers} workers)", zorder=1)
    ax.plot(pos, speedup, color=SLOT[0], marker="o", markeredgecolor="white",
            markeredgewidth=1.2, linewidth=3.0, markersize=9, zorder=3)

    best = max(xs, key=lambda t: serial / pts[t][0])
    ax.plot([pos[xs.index(best)]], [serial / pts[best][0]], marker="o",
            markersize=15, markerfacecolor="none", markeredgecolor=SLOT[1],
            markeredgewidth=2.5, zorder=4,
            label=f"best: {pts[best][1]} ranks x {best} thr")

    ax.set_yscale("log", base=2)
    ax.set_xticks(pos)
    ax.set_xticklabels(["{}\n{} ranks".format(t, pts[t][1]) for t in xs])
    ax.set_xlim(-0.4, len(xs) - 0.6)
    # Headroom above the ideal line so the legend never sits on the data.
    ax.set_ylim(min(speedup) * 0.75, workers * 2.2)
    ax.minorticks_off()
    ax.set_xlabel("threads per rank")
    ax.set_ylabel("speedup  $t_1/t_p$")
    ax.set_title(f"STFT: rank / thread split at {workers} fixed cores",
                 loc="left", pad=14)
    ax.legend(loc="upper left", framealpha=0.95)

    param_box(fig, [("machine", machine), ("statistic", stat[:-3]),
                    ("frame / hop", f"{FRAME} / {HOP}"),
                    ("sample rate", f"{RATE / 1000:g} kHz"),
                    ("cores", workers),
                    ("", ""),
                    ("note", "every point uses"),
                    ("", "the same cores,"),
                    ("", "split differently")])
    finish(fig, out)


def fig_granularity(rows, machine, stat, out, distributed=False):
    """The same thread budget spent across frames vs inside each transform.

    This is the figure that justifies the engine choice: the STFT could put its
    threads either on the frame loop (many independent transforms, one thread
    each) or inside every transform (one frame at a time, all threads on its
    butterflies). Both rows use the same thread count and the same serial
    baseline, so the gap between them is the cost of the granularity decision
    and nothing else.

    The reference is horizontal, not diagonal: the worker count is fixed here
    and the workload is what varies, so "ideal" means the whole budget turned
    into speedup at every problem size.

    `distributed` switches the same comparison to its MPI form, where every
    rank splits its own share of the frames the same two ways. One function
    serves both because only the row source and the wording change — the
    experiment, the baseline and the reading are identical.
    """
    section = "mpi_granularity" if distributed else "stft_granularity"
    source  = "benchmark_mpi"   if distributed else "benchmark_openmp"
    base_section = "mpi" if distributed else "stft"
    serial_mark  = "serial (1 rank" if distributed else "1 thread"
    # One line each, not two: this legend sits inside the axes, and a box twice
    # as tall hides twice as much of the curves. What the two arms are made of
    # goes in the parameter box instead, where it costs no plot area.
    LABELS = {"OpenMP frames": "across frames",
              "sequential frames": "within each transform"}
    COLORS = {"OpenMP frames": ALGO_COLOR["IterativeFFT"],
              "sequential frames": ALGO_COLOR["ParallelFFT"]}

    curves = defaultdict(dict)      # short key -> {seconds: time}
    serial = {}                     # seconds -> serial time
    workers = None                  # the budget the ideal line stands for
    for r in rows:
        if r["machine"] != machine or r.get("source") != source:
            continue
        sec = r.get("workload_s")
        if sec is None:
            continue
        if r["section"] == base_section and serial_mark in r["variant"]:
            serial[sec] = min(serial.get(sec, r[stat]), r[stat])
        elif r["section"] == section:
            key = next((k for k in LABELS if r["variant"].startswith(k)), None)
            if key is None:
                continue
            prev = curves[key].get(sec)
            curves[key][sec] = r[stat] if prev is None else min(prev, r[stat])
            workers = (r.get("ranks") or 1) * (r.get("threads") or 1)

    if len(curves) < 2 or not serial:
        print(f"  skip {out.name}: need both strategies and a serial baseline "
              f"for {machine}")
        return

    fig, ax = plt.subplots(figsize=(11.0, 5.6))
    fig.subplots_adjust(right=0.74)
    all_x = set()
    for key in ("OpenMP frames", "sequential frames"):
        pts = {x: t for x, t in curves.get(key, {}).items() if x in serial}
        if len(pts) < 2:
            continue
        xs = sorted(pts)
        all_x.update(xs)
        ax.plot(xs, [serial[x] / pts[x] for x in xs], color=COLORS[key],
                marker="o", markeredgecolor="white", markeredgewidth=1.2,
                label=LABELS[key], zorder=3)

    if workers:
        ax.axhline(workers, color=REFERENCE, linewidth=2.4, linestyle="--",
                   label=f"ideal ({workers} workers)", zorder=1)
    ax.axhline(1.0, color=GRID, linewidth=1.1, zorder=1)

    xs = sorted(all_x)
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    # Only the measured workloads get a label: a log axis with two or three
    # points otherwise decorates the gaps between them with 2x10^1 and friends,
    # which say nothing and crowd out the numbers that do.
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("audio length [s]")
    ax.set_ylabel("speedup over serial STFT")
    # "best" rather than a fixed corner: which corner is free depends on the
    # data, and pinning it is how a legend ends up sitting on a curve.
    ax.legend(loc="best")
    ax.set_title("Thread granularity: across frames or inside the FFT"
                 + (" — per rank" if distributed else ""),
                 color=INK, loc="left", fontsize=15.5, pad=14)

    param_box(fig, [("machine", machine),
                    ("statistic", stat[:-3]),
                    ("workers", workers or "?"),
                    ("MPI ranks", "several" if distributed else 1),
                    ("window", "Hann window"),
                    ("baseline", "serial STFT,"),
                    ("", "1 thread, same"),
                    ("", "workload"),
                    ("", ""),
                    ("across frames", "OpenMP loop +"),
                    ("", "IterativeFFT"),
                    ("within transform", "sequential loop +"),
                    ("", "ParallelFFT"),
                    ("", ""),
                    ("note", "both rows spend"),
                    ("", "the same budget;"),
                    ("", "only the level"),
                    ("", "they act on"),
                    ("", "differs"),
                    ("", ""),
                    ("", "frames are whole"),
                    ("", "independent tasks;"),
                    ("", "the stages of one"),
                    ("", "transform are not")])
    finish(fig, out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=Path("results/results_analysis/csv"))
    ap.add_argument("--out", type=Path, default=Path("results/results_analysis/figures"))
    ap.add_argument("--machine", action="append", default=None)
    ap.add_argument("--stat", default="min", choices=["min", "median", "mean"])
    ap.add_argument("--metric", default="speedup",
                    choices=["speedup", "efficiency"],
                    help="what the scaling figures plot. speedup is the usual "
                         "curve against a diagonal ideal; efficiency divides it "
                         "by the worker count, which pulls apart workload curves "
                         "that overlap near the origin (default: speedup)")
    ap.add_argument("--only", default=None,
                    help="comma list: fft_algorithms, stft_openmp, stft_mpi, "
                         "stft_hybrid, memory, granularity, hybrid_split")
    args = ap.parse_args()

    stat = f"{args.stat}_ms"
    wanted = ({w.strip() for w in args.only.split(",")} if args.only else
              {"fft_algorithms", "stft_openmp", "stft_mpi", "stft_hybrid",
               "memory", "granularity", "hybrid_split"})

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
            fig_fft(timings, m, stat, args.out / f"{m}_1_fft_algorithms")
        if "stft_openmp" in wanted:
            fig_stft(timings, m, stat, args.out / f"{m}_2_stft_openmp",
                     "openmp", args.metric)
        if "stft_mpi" in wanted:
            fig_stft(timings, m, stat, args.out / f"{m}_3_stft_mpi",
                     "mpi", args.metric)
        if "stft_hybrid" in wanted:
            fig_stft(timings, m, stat, args.out / f"{m}_4_stft_hybrid",
                     "hybrid", args.metric)
        if "memory" in wanted:
            fig_memory(memory, m, args.out / f"{m}_5_memory")
        if "hybrid_split" in wanted:
            fig_hybrid_split(timings, m, stat,
                             args.out / f"{m}_9_hybrid_split")
        if "granularity" in wanted:
            fig_granularity(timings, m, stat,
                            args.out / f"{m}_7_granularity")
            fig_granularity(timings, m, stat,
                            args.out / f"{m}_8_granularity_mpi",
                            distributed=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
