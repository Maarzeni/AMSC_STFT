#!/bin/bash
#SBATCH --job-name=AMSC_STFT         ## Job name
#SBATCH --output=amsc_stft_job.out   ## Standard output file
#SBATCH --error=amsc_stft_job.err    ## Standard error file
#SBATCH --time=04:00:00              ## Maximum job duration
#SBATCH --nodes=1                    ## Single node (see MPI note below)
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=4          ## 8 x 4 = 32 cores
#SBATCH --mem=32G
#SBATCH --partition=g100_usr_prod
#SBATCH --account=tra26_TRNPLM     ## re-check the name with `saldo -b`

## ── Resources ────────────────────────────────────────────────────────────────
## Production partition on dedicated compute nodes, charged to tra26_TRNPLM.
## `saldo -b` reports the project open until 2026-09-30 with 3728 of 6000
## core-hours unspent, which is what makes g100_usr_prod accept the job: an
## expired window is what used to answer "invalid account or expired budget".
##
## 8 ranks x 4 threads = 32 cores on one node, four hours of walltime. The
## margin is deliberate: job 21842075 was killed at a one-hour limit having
## produced nothing, because the granularity row costs frames x ParallelFFT,
## and ParallelFFT degrades badly once the threads outnumber the work in a
## 1024-point transform. Long audio multiplies that by ten thousand frames.
##
## Nothing here derives from the old serial-queue setup: TOTAL_CORES and
## `mpirun -np` both read the SBATCH values above, and run_suite.sh sizes its
## sweeps from the SLURM allocation.

# ── Notes ─────────────────────────────────────────────────────────────────────
# * MPI runs INSIDE the container (single node) using the OpenMPI shipped in the
#   image. This avoids host/container MPI ABI mismatches. For MULTI-node MPI you
#   would instead launch with the host `mpirun`/`srun` + PMIx matching the image,
#   which is significantly more fragile — keep it to one node here.
# * The submission directory is bind-mounted into the container, so example
#   programs can read the WAV files from the host and write their PNG output back
#   to the host (they persist after the job ends).
# ──────────────────────────────────────────────────────────────────────────────

# Move to the directory where the job was submitted
cd "${SLURM_SUBMIT_DIR}"

echo "Job started on node: ${SLURMD_NODENAME}"

# Make the singularity/apptainer launcher available (name varies across the stack)
module load singularity 2>/dev/null || module load apptainer 2>/dev/null || true

# Verify that the container image exists
if [ ! -f "amsc_stft.sif" ]; then
    echo "ERROR: amsc_stft.sif not found in ${SLURM_SUBMIT_DIR}!"
    exit 1
fi

# Total cores reserved on the node = ranks * threads-per-rank
TOTAL_CORES=$(( SLURM_NTASKS * SLURM_CPUS_PER_TASK ))

# Container paths (binaries were compiled immutably inside the image)
SIF="amsc_stft.sif"
BUILD=/app/AMSC_STFT/core/build
BIND="--bind ${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR}"

# ── Run the benchmark suite ───────────────────────────────────────────────────
# Everything measured here lives in scripts/run_suite.sh, which the GitHub
# workflow and any manual run call too: one recipe, so the three environments
# stay comparable instead of drifting apart.
#
# The script is executed INSIDE the container, where the binaries are. It works
# out on its own that the login node's default TMPDIR is read-only, that
# OpenMPI's Cross Memory Attach path is unavailable in a container, and how many
# ranks/threads SLURM reserved.
RESULTS="${SLURM_SUBMIT_DIR}/results"
mkdir -p "${RESULTS}"

export SINGULARITYENV_RESULTS_DIR="${RESULTS}/results_benchmark"
export SINGULARITYENV_TEST_RESULTS_DIR="${RESULTS}/results_test"
export SINGULARITYENV_PREFIX="cluster"

export APPTAINERENV_RESULTS_DIR="${RESULTS}/results_benchmark"
export APPTAINERENV_TEST_RESULTS_DIR="${RESULTS}/results_test"
export APPTAINERENV_PREFIX="cluster"

# ── Measurement configuration ────────────────────────────────────────────────
# Kept in one place so the CI run and a manual `sbatch scripts/job.sh` measure
# exactly the same thing.
#
# The granularity pair runs on eight threads regardless of the allocation (see
# AMSC_GRANULARITY_THREADS in benchmark_Main.cpp): its ParallelFFT arm costs
# frames x transform, and ParallelFFT degrades once the threads outnumber the
# work in a 1024-point transform. Thirty-two threads over ten thousand frames
# is what ran job 21842075 into its wall clock.
#
# Dense variant for the final run, once the results are settled (~3 h):
#   BENCH_ARGS="15 3 1024 512"
#   SCALING_RANKS="1 2 4 6 8 10 12 14 16 20 24 28 32"
#   THREAD_LIST="1 2 4 6 8 10 12 14 16 20 24 32"
#   SCALING_DURATIONS="5 30 120"; MEM_DURATION=120; WEAK_BASE=10
for v in \
    "BENCH_ARGS=7 2 1024 512" \
    "SCALING_RANKS=1 2 4 8 16 24 32" \
    "THREAD_LIST=1 2 4 8 16 24 32" \
    "SCALING_DURATIONS=5 15" \
    "MEM_DURATION=30" \
    "WEAK_BASE=5" \
    "AMSC_GRANULARITY_THREADS=8"; do
    export "SINGULARITYENV_${v}"
    export "APPTAINERENV_${v}"
done

singularity exec ${BIND} --pwd "${SLURM_SUBMIT_DIR}" ${SIF} \
    bash "${SLURM_SUBMIT_DIR}/scripts/run_suite.sh"

# ══════════════════════════════════════════════════════════════════════════════
# Examples: WAV → spectrogram PNG
# ══════════════════════════════════════════════════════════════════════════════
# Not part of the measured suite: these read a WAV from the HOST and write their
# PNG back to the HOST, so they belong to the cluster deployment rather than to
# the benchmarks. STFT_EXAMPLES_DIR is what sends the output to the submit
# directory instead of the container's own read-only source tree.
# They do need the same two MPI workarounds the suite applies internally — a
# writable TMPDIR for the ORTE session directory, and no Cross Memory Attach
# inside the container.
echo ""
echo "==================== Examples (STFT spectrograms) ========================"
TMPWORK="${SLURM_SUBMIT_DIR}/tmp"
mkdir -p "${TMPWORK}"
export SINGULARITYENV_TMPDIR="${TMPWORK}" APPTAINERENV_TMPDIR="${TMPWORK}"
unset DISPLAY XAUTHORITY
MCA_OPTS="--mca orte_tmpdir_base ${TMPWORK} --mca btl_vader_single_copy_mechanism none"

EXAMPLES_OUT="${RESULTS}/results_examples"
mkdir -p "${EXAMPLES_OUT}"
export SINGULARITYENV_STFT_EXAMPLES_DIR="${EXAMPLES_OUT}"
export APPTAINERENV_STFT_EXAMPLES_DIR="${EXAMPLES_OUT}"

WAV="${SLURM_SUBMIT_DIR}/core/examples/data/test_audio.wav"
if [ -f "${WAV}" ]; then
    # Shared-memory example: PNG into ${EXAMPLES_OUT}
    export OMP_NUM_THREADS=${TOTAL_CORES}
    singularity exec ${BIND} ${SIF} ${BUILD}/examples/main "${WAV}" 1024 512 hann

    # Distributed example on the same WAV, same parameters, same destination
    export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
    singularity exec ${BIND} ${SIF} \
        mpirun --bind-to none -np ${SLURM_NTASKS} \
        ${MCA_OPTS} \
        ${BUILD}/examples/mpi_main "${WAV}" 1024 512 hann
else
    echo "WARNING: ${WAV} not found — skipping examples."
    echo "         Upload core/examples/data/ alongside amsc_stft.sif to enable them."
fi

echo ""
echo "Analysis completed!"
