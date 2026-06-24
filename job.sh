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

# Diciamo ad OpenMP di usare le CPU riservate da SLURM
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Dato che il codice è GIÀ STATO COMPILATO dentro il container in /app/...
# Noi diciamo a Singularity di posizionarsi in quella cartella interna (--pwd /app/...)
# ed eseguire direttamente i test con ctest!
singularity exec --bind ${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR} \
  --pwd /app/AMSC_STFT/ft_project/build \
  amsc_stft.sif ctest --output-on-failure

echo "Analysis completed!"