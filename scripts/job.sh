#!/bin/bash
#SBATCH --job-name=AMSC_STFT         ## Job name
#SBATCH --output=amsc_stft_job.out   ## Standard output file
#SBATCH --error=amsc_stft_job.err    ## Standard error file
#SBATCH --time=00:15:00              ## Small by design — see header note below
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=g100_usr_prod

## No --account here on purpose: this file is tracked in a public repository,
## and a grant/account ID should not be. Supply yours at submission time —
## `sbatch --account=<your_account> scripts/job.sh`, or `export
## SBATCH_ACCOUNT=<your_account>` first — either overrides the missing
## directive. The CI/CD pipeline does the same, from a GitHub Actions secret.
##
## This is deliberately the SMALL entry point: it exists so the CI/CD pipeline
## has something to submit that proves the deployed container builds, passes
## its tests and runs on Galileo100, in minutes, on a couple of cores. It runs
## scripts/run_suite.sh exactly like every other environment does, so a green
## run here is the same evidence a full run would give, just faster and
## cheaper. For an actual measurement run — the whole node, the full sweep,
## the WAV → spectrogram examples — see the local, untracked job_full.sh
## (same shape as this file, sized for a real measurement instead of a
## smoke test).

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
BIND="--bind ${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR}"

# ── Run the suite ──────────────────────────────────────────────────────────
# Everything measured here lives in scripts/run_suite.sh, which the GitHub
# workflow, job_full.sh and any manual run call too: one recipe, so every
# environment stays comparable instead of drifting apart.
#
# The script is executed INSIDE the container, where the binaries are. It
# works out on its own that the login node's default TMPDIR is read-only,
# that OpenMPI's Cross Memory Attach path is unavailable in a container, and
# how many ranks/threads SLURM reserved.
RESULTS="${SLURM_SUBMIT_DIR}/results"
mkdir -p "${RESULTS}"

export SINGULARITYENV_RESULTS_DIR="${RESULTS}/results_benchmark"
export SINGULARITYENV_TEST_RESULTS_DIR="${RESULTS}/results_test"
export SINGULARITYENV_PREFIX="cluster"

# Passed straight through when set, so the pipeline can ask for tests only.
if [ -n "${TESTS_ONLY:-}" ]; then
    export SINGULARITYENV_TESTS_ONLY="${TESTS_ONLY}"
    export APPTAINERENV_TESTS_ONLY="${TESTS_ONLY}"
fi

export APPTAINERENV_RESULTS_DIR="${RESULTS}/results_benchmark"
export APPTAINERENV_TEST_RESULTS_DIR="${RESULTS}/results_test"
export APPTAINERENV_PREFIX="cluster"

singularity exec ${BIND} --pwd "${SLURM_SUBMIT_DIR}" ${SIF} \
    bash "${SLURM_SUBMIT_DIR}/scripts/run_suite.sh"

echo ""
echo "Analysis completed!"
