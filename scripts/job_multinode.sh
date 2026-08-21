#!/bin/bash
#SBATCH --job-name=AMSC_STFT_MN      ## Job name
#SBATCH --output=amsc_mn_%j.out      ## Standard output file
#SBATCH --error=amsc_mn_%j.err       ## Standard error file
#SBATCH --time=04:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=48         ## 4 x 48 = 192 ranks, one per core
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --partition=g100_usr_prod
#SBATCH --account=tra26_TRNPLM

## ── Why this job exists, separately from job.sh ──────────────────────────────
## Everything job.sh measures happens inside ONE node, where shared memory is
## available and message passing is pure overhead. That is a real result, but it
## judges the distributed backend in the one regime where it cannot win. This
## job answers the other question: what happens when the problem no longer fits
## in a single machine, and MPI stops being optional.
##
## Two things differ, and both are forced by multi-node MPI:
##
##  1. NO CONTAINER. The image ships its own OpenMPI, which can start ranks on
##     the node it runs on and nowhere else — it has no way to reach the other
##     three. Multi-node placement is the batch scheduler's job, so the code is
##     built here against the cluster's own MPI and launched with srun.
##
##  2. NO SHORT WORKLOADS. At 192 ranks, five seconds of audio leaves each rank
##     two frames; the measurement would be of MPI's startup, not of the STFT.
##     The workloads below keep at least ~130 frames per rank at full width.
##
## The A/B that isolates the network costs nothing extra: submit this same file
## twice with the same rank count on a different number of nodes, e.g.
##     sbatch --nodes=1 --ntasks-per-node=48 scripts/job_multinode.sh
##     sbatch --nodes=2 --ntasks-per-node=24 scripts/job_multinode.sh
## Identical 48 ranks, identical work; the difference between them IS the
## interconnect.

set -uo pipefail

cd "${SLURM_SUBMIT_DIR}" || exit 1
echo "job      : ${SLURM_JOB_ID}"
echo "nodes    : ${SLURM_JOB_NUM_NODES} (${SLURM_JOB_NODELIST})"
echo "ranks    : ${SLURM_NTASKS}"

# ── Toolchain ────────────────────────────────────────────────────────────────
# The cluster's own compiler and MPI, so that srun and the MPI runtime agree on
# how to bootstrap ranks across nodes. Module names vary between CINECA stacks,
# so each load is attempted and reported rather than assumed.
module purge 2>/dev/null || true
for m in gcc openmpi cmake; do
    module load "${m}" 2>/dev/null && echo "loaded   : ${m}" \
        || echo "WARNING  : module ${m} not loaded, relying on the default environment"
done
echo "compiler : $(command -v mpicxx || command -v g++)"
echo "launcher : $(command -v srun)"

# ── Build ────────────────────────────────────────────────────────────────────
BUILD="${SLURM_SUBMIT_DIR}/build-mn"
cmake -S "${SLURM_SUBMIT_DIR}/core" -B "${BUILD}" -DCMAKE_BUILD_TYPE=Release || exit 1
cmake --build "${BUILD}" -j 16 || exit 1
[ -x "${BUILD}/benchmarks/benchmark_MPI_Main" ] || {
    echo "ERROR: build produced no benchmark_MPI_Main" >&2; exit 1; }

# ── Measurement configuration ────────────────────────────────────────────────
# srun, not mpirun: only the scheduler knows which nodes we hold.
export MPI_LAUNCHER="srun --mpi=pmix -n"
export BUILD_DIR="${BUILD}"
export PREFIX="cineca-mn${SLURM_JOB_NUM_NODES}"
export RESULTS_DIR="${SLURM_SUBMIT_DIR}/results-mn/results_benchmark"
export TEST_RESULTS_DIR="${SLURM_SUBMIT_DIR}/results-mn/results_test"

# Rank counts chosen to straddle the node boundary: with 48 cores per node, 48
# is exactly one node, 96 is two, 192 is four. The interesting feature of the
# curve is the step between them, not the smooth part inside a node.
export SCALING_RANKS="12 24 48 96 192"

# Five and twenty minutes of audio. The short workloads of the single-node run
# are deliberately absent: they are already measured there, and at this width
# they would report startup latency rather than scaling.
export SCALING_DURATIONS="300 1200"
export MEM_DURATION=300
export WEAK_BASE=5

# One thread per rank: this job is about the distributed dimension, and the
# shared-memory questions — thread scaling, FFT engines, granularity — were
# settled on one node where they belong. Skipping them also keeps the run
# inside its walltime.
export SCALING_THREADS=1
export THREAD_SCALING=0
export AMSC_SKIP_GRANULARITY=1

bash "${SLURM_SUBMIT_DIR}/scripts/run_suite.sh"
STATUS=$?

echo ""
echo "results in ${RESULTS_DIR}"
ls -1 "${RESULTS_DIR}" 2>/dev/null | sed 's/^/  /'
exit "${STATUS}"
