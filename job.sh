#!/bin/bash
#SBATCH --job-name=AMSC_STFT        ## Job name
#SBATCH --output=amsc_stft_job.out  ## Standard output file
#SBATCH --error=amsc_stft_job.err   ## Standard error file
#SBATCH --time=00:10:00             ## Maximum job duration
#SBATCH --ntasks=1                  ## Number of tasks
#SBATCH --cpus-per-task=4           ## Number of CPUs per task
#SBATCH --mem=2G                    ## Requested memory

# Move to the directory where the job was submitted
cd ${SLURM_SUBMIT_DIR}

echo "Job started on node: $SLURMD_NODENAME"

# Verify that the container image exists
if [ ! -f "amsc_stft.sif" ]; then
    echo "ERROR: amsc_stft.sif not found!"
    exit 1
fi

echo "Running tests from the immutable container..."

# Tell OpenMP to use the CPUs reserved by SLURM
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Since the code has ALREADY BEEN COMPILED inside the container in /app/...
# we tell Singularity to move into that internal folder (--pwd /app/...)
# and run the tests directly with ctest!
singularity exec --bind ${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR} \
  --pwd /app/AMSC_STFT/ft_project/build \
  amsc_stft.sif ctest --output-on-failure

# TODO: add benchmarks and examples here

echo "Analysis completed!"