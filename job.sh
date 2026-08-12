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

## NOTE: no --account on purpose — SLURM then charges the job to the user's
## default account. Hardcoding an account that is not yours (or whose budget
## expired) makes sbatch fail with "invalid account or expired budget"; check
## yours with `saldo -b` on the login node.
##
## The resources above are sized for g100_all_serial, whose QOS caps each user at
## 4 cores and ~30 GB: asking for more fails at submission with
## "QOSMaxCpuPerUserLimit". To scale up (more ranks/threads) you need a valid
## budget and a production partition, e.g.
##   #SBATCH --account=<your_account>
##   #SBATCH --partition=g100_usr_prod

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

# ── Writable scratch ──────────────────────────────────────────────────────────
# The .sif is a read-only squashfs, and on the login nodes /scratch_local (the
# default TMPDIR) is read-only too. Two things need to write somewhere:
#   * ctest       → build/Testing/Temporary/LastTest.log, INSIDE the image;
#                   --writable-tmpfs gives it a throwaway in-memory overlay.
#   * OpenMPI     → its ORTE session directory under TMPDIR; without a writable
#                   one, orte_init fails and mpirun never starts the ranks.
# The submission directory is bind-mounted and writable, so TMPDIR points there.
TMPWORK="${SLURM_SUBMIT_DIR}/tmp"
mkdir -p "${TMPWORK}"
export SINGULARITYENV_TMPDIR="${TMPWORK}"   # TMPDIR as seen INSIDE the container
export APPTAINERENV_TMPDIR="${TMPWORK}"     # same, for the apptainer-named stack

# The login node exports DISPLAY/XAUTHORITY without a reachable X server, which
# makes container invocations print "No protocol specified". Purely cosmetic.
unset DISPLAY XAUTHORITY

# OpenMPI's shared-memory BTL (vader, OpenMPI 4.x on Ubuntu 22.04) tries to move
# large messages with Cross Memory Attach, i.e. process_vm_readv(). That syscall
# needs ptrace privileges the container does not have, so every scatter of the
# signal fails with "Read -1, expected <n>, errno = 1" (EPERM) before OpenMPI
# retries through a slower copy-in/copy-out path. The data still arrives, but
# the timings absorb the failed attempt. Disabling single-copy skips it.
MCA_OPTS="--mca orte_tmpdir_base ${TMPWORK} --mca btl_vader_single_copy_mechanism none"

# Benchmark parameters: reps, warmup, frame, hop.  Kept IDENTICAL to the ones the
# CI workflow uses on the GitHub runner, otherwise the two result tables measure
# different workloads and cannot be compared.  No 5th argument, so both binaries
# run their full duration sweep (1, 5, 10, 30 s of audio).
BENCH_ARGS="7 2 1024 512"

# Every result file is collected here; the CD workflow copies this folder back
# and publishes it as a build artifact.
RESULTS="${SLURM_SUBMIT_DIR}/results"
mkdir -p "${RESULTS}"

# Machine description, so a results file is interpretable months later without
# having to remember which node it ran on.
{
    echo "environment  : CINECA Galileo100 (SLURM job ${SLURM_JOB_ID})"
    echo "node         : ${SLURMD_NODENAME}"
    echo "partition    : ${SLURM_JOB_PARTITION}"
    echo "ranks x thr  : ${SLURM_NTASKS} x ${SLURM_CPUS_PER_TASK} = ${TOTAL_CORES} cores"
    echo "bench args   : ${BENCH_ARGS}  (reps warmup frame hop)"
    echo "date (UTC)   : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo ""
    lscpu 2>/dev/null | grep -E 'Model name|^CPU\(s\)|Thread\(s\) per core|Core\(s\) per socket|CPU MHz|L3 cache'
} | tee "${RESULTS}/cluster_env.txt"

# ══════════════════════════════════════════════════════════════════════════════
# 1. Unit tests (GoogleTest via ctest)
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "==================== [1/4] Running unit tests (ctest) ===================="
export OMP_NUM_THREADS=${TOTAL_CORES}

# --writable-tmpfs is not enough here (it is refused on this stack), so instead
# ctest is run from a writable MIRROR of the build tree. Only the generated
# CTestTestfile.cmake files are copied: each one registers its tests by ABSOLUTE
# path into the image, so the very same read-only binaries are executed, while
# Testing/Temporary/LastTest.log now lands on writable storage.
CTESTDIR="${SLURM_SUBMIT_DIR}/ctest-run"
rm -rf "${CTESTDIR}"
mkdir -p "${CTESTDIR}"
singularity exec ${BIND} --pwd ${BUILD} ${SIF} \
    sh -c "find . -name CTestTestfile.cmake -exec cp --parents -t ${CTESTDIR} {} +"

singularity exec ${BIND} --pwd ${CTESTDIR} ${SIF} \
    ctest --output-on-failure \
    2>&1 | tee "${RESULTS}/cluster_ctest.txt"

# ══════════════════════════════════════════════════════════════════════════════
# 2. Serial / OpenMP benchmark (single process, all reserved cores)
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "============== [2/4] Benchmark: serial / OpenMP (benchmark_Main) =========="
export OMP_NUM_THREADS=${TOTAL_CORES}
singularity exec ${BIND} --pwd ${BUILD} ${SIF} \
    ${BUILD}/benchmarks/benchmark_Main ${BENCH_ARGS} \
    2>&1 | tee "${RESULTS}/cluster_benchmark_openmp.txt"

# ══════════════════════════════════════════════════════════════════════════════
# 3. Distributed / hybrid benchmark (MPI ranks × OpenMP threads, single node)
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "============ [3/4] Benchmark: MPI / hybrid (benchmark_MPI_Main) ==========="
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
singularity exec ${BIND} --pwd ${BUILD} ${SIF} \
    mpirun --bind-to none -np ${SLURM_NTASKS} \
    ${MCA_OPTS} \
    ${BUILD}/benchmarks/benchmark_MPI_Main ${BENCH_ARGS} \
    2>&1 | tee "${RESULTS}/cluster_benchmark_mpi.txt"

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
        ${MCA_OPTS} \
        ${BUILD}/examples/mpi_main "${WAV}"
else
    echo "WARNING: ${WAV} not found — skipping examples."
    echo "         Upload ft_project/tests/data/ alongside amsc_stft.sif to enable them."
fi

echo ""
echo "Analysis completed!"
