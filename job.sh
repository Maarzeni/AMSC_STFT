#!/bin/bash
#SBATCH --job-name=AMSC_STFT         ## Job name
#SBATCH --output=amsc_stft_job.out   ## Standard output file
#SBATCH --error=amsc_stft_job.err    ## Standard error file
#SBATCH --time=00:30:00              ## Maximum job duration
#SBATCH --nodes=1                    ## Single node (see MPI note below)
#SBATCH --ntasks=4                   ## MPI ranks for the distributed benchmark
#SBATCH --cpus-per-task=4            ## OpenMP threads per rank
#SBATCH --mem=8G                     ## Requested memory
## NOTE: no --account / --partition on purpose: SLURM then charges the job to the
## user's default account and queue. Hardcoding an account that is not yours (or
## whose budget expired) makes sbatch fail with "invalid account or expired
## budget". Check your valid accounts with `saldo -b` on the login node.

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
BUILD=/app/AMSC_STFT/ft_project/build
BIND="--bind ${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR}"

# ══════════════════════════════════════════════════════════════════════════════
# 1. Unit tests (GoogleTest via ctest)
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "==================== [1/4] Running unit tests (ctest) ===================="
export OMP_NUM_THREADS=${TOTAL_CORES}
singularity exec ${BIND} --pwd ${BUILD} ${SIF} ctest --output-on-failure

# ══════════════════════════════════════════════════════════════════════════════
# 2. Serial / OpenMP benchmark (single process, all reserved cores)
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "============== [2/4] Benchmark: serial / OpenMP (benchmark_Main) =========="
export OMP_NUM_THREADS=${TOTAL_CORES}
singularity exec ${BIND} --pwd ${BUILD} ${SIF} ${BUILD}/benchmarks/benchmark_Main

# ══════════════════════════════════════════════════════════════════════════════
# 3. Distributed / hybrid benchmark (MPI ranks × OpenMP threads, single node)
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "============ [3/4] Benchmark: MPI / hybrid (benchmark_MPI_Main) ==========="
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
singularity exec ${BIND} --pwd ${BUILD} ${SIF} \
    mpirun --bind-to none -np ${SLURM_NTASKS} \
    ${BUILD}/benchmarks/benchmark_MPI_Main

# ══════════════════════════════════════════════════════════════════════════════
# 4. Examples: WAV → spectrogram PNG (output written next to the host WAV)
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "==================== [4/4] Examples (STFT spectrograms) =================="
WAV="${SLURM_SUBMIT_DIR}/ft_project/tests/data/examples_test-audio.wav"
if [ -f "${WAV}" ]; then
    # Serial/OpenMP example: <stem>_spectrogram.png next to the input WAV
    export OMP_NUM_THREADS=${TOTAL_CORES}
    singularity exec ${BIND} ${SIF} ${BUILD}/examples/main "${WAV}" 1024 512 hann

    # Distributed example on the same WAV
    export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
    singularity exec ${BIND} ${SIF} \
        mpirun --bind-to none -np ${SLURM_NTASKS} \
        ${BUILD}/examples/mpi_main "${WAV}"
else
    echo "WARNING: ${WAV} not found — skipping examples."
    echo "         Upload ft_project/tests/data/ alongside amsc_stft.sif to enable them."
fi

echo ""
echo "Analysis completed!"
