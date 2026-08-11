#!/usr/bin/env bash
# =============================================================================
# Strong-scaling sweep for the distributed STFT benchmark.
#
# The rank count is fixed by mpirun when the job starts, so it cannot be varied
# from inside benchmark_MPI_Main: the sweep has to launch the binary once per
# configuration.  Everything else (frame geometry, signal length, repetitions)
# is held constant across the runs, which is what makes the times comparable.
#
# Run from the build directory:
#     ./benchmarks/run_scaling.sh
#
# Every knob is an environment variable, so a different sweep needs no edit:
#     RANKS="1 2 4 8" THREADS=2 FRAME=8192 ./benchmarks/run_scaling.sh
#
# Reading the results: -np 1 is the serial baseline, and the ratio between its
# `min` and the `min` of each larger rank count is the strong-scaling speedup.
# `min` is preferred over `mean` here because it is the least polluted by
# unrelated system activity.
# =============================================================================
set -euo pipefail

BIN=${BIN:-./benchmarks/benchmark_MPI_Main}
RANKS=${RANKS:-"1 2 4"}
THREADS=${THREADS:-1}
REPS=${REPS:-7}
WARMUP=${WARMUP:-2}
FRAME=${FRAME:-1024}
HOP=${HOP:-0}                       # 0 → frame / 2
DURATION=${DURATION:-10}            # seconds of synthetic audio
MPIRUN_OPTS=${MPIRUN_OPTS:---oversubscribe}

if [ ! -x "$BIN" ]; then
    echo "Benchmark binary not found: $BIN" >&2
    echo "Build it first with: make benchmark_MPI_Main" >&2
    exit 1
fi

echo "Strong-scaling sweep"
echo "  ranks       : $RANKS"
echo "  threads/rank: $THREADS"
echo "  frame / hop : $FRAME / $HOP  (0 = frame/2)"
echo "  duration    : ${DURATION}s   repetitions: $REPS (warmup $WARMUP)"

for np in $RANKS; do
    echo
    echo "════════════════════════════════════════════════════════════════"
    echo "  -np ${np}   OMP_NUM_THREADS=${THREADS}   →   $((np * THREADS)) total workers"
    echo "════════════════════════════════════════════════════════════════"

    OMP_NUM_THREADS="$THREADS" mpirun -np "$np" $MPIRUN_OPTS \
        "$BIN" "$REPS" "$WARMUP" "$FRAME" "$HOP" "$DURATION"
done

echo
echo "Done. Compare the 'min' column across the runs above:"
echo "  speedup(P) = min(-np 1) / min(-np P)"
