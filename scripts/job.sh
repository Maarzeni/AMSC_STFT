#!/bin/bash
#SBATCH --job-name=AMSC_STFT         ## Job name
#SBATCH --output=amsc_stft_job.out   ## Standard output file
#SBATCH --error=amsc_stft_job.err    ## Standard error file
#SBATCH --time=00:30:00              ## Maximum job duration
#SBATCH --nodes=1                    ## Single node (see MPI note below)
#SBATCH --ntasks=2                   ## MPI ranks for the distributed benchmark
#SBATCH --cpus-per-task=2            ## OpenMP threads per rank (2 x 2 = 4 cores)
#SBATCH --mem=4G                     ## Requested memory
#SBATCH --partition=g100_all_serial  ## Default queue, usable without a budget

## ── Why the serial partition, and how to leave it ────────────────────────────
## g100_all_serial runs on the shared LOGIN nodes and needs no budget, which is
## why it works right now. Its QOS caps each user at 4 cores and ~30 GB, so the
## 2x2 layout above is the largest that will be accepted: asking for more fails
## at submission with "QOSMaxCpuPerUserLimit".
##
## Consequences worth knowing when reporting these numbers:
##   * timings come from a node shared with every interactive user;
##   * the scaling study is confined to 1..4 workers.
##
## Moving to dedicated compute nodes needs an account with an OPEN budget window.
## As of 2026-08-12 `saldo -b` reports tra26_TRNPLM ending 2026-07-31 with 3728
## of 6000 core-hours unspent: the budget is not exhausted, the project window
## simply closed, which is what makes sbatch answer "expired budget". Once the
## course PI has it extended (or a new account is granted), swap the three
## resource lines above for the block below and nothing else needs to change —
## TOTAL_CORES and `mpirun -np` both derive from these values.
##
##   #SBATCH --ntasks=8
##   #SBATCH --cpus-per-task=4          ## 8 x 4 = 32 cores
##   #SBATCH --mem=32G
##   #SBATCH --partition=g100_usr_prod
##   #SBATCH --account=tra26_TRNPLM     ## re-check the name with `saldo -b`

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

export SINGULARITYENV_RESULTS_DIR="${RESULTS}"
export SINGULARITYENV_PREFIX="cluster"
export APPTAINERENV_RESULTS_DIR="${RESULTS}"
export APPTAINERENV_PREFIX="cluster"

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
