#!/bin/bash
#SBATCH --account=def-sushama-ab_cpu
#SBATCH --job-name=SL_Frl-0p8_nl-0p4_nu-0p8_rhoR-0p8_hR-1_kk-1_v3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --mem-per-cpu=60M           # memory for the entire job across all cores 3900M
#SBATCH --time=0-00:10
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=boyuan.yu@mail.mcgill.ca
#SBATCH --hint=nomultithread

set -euo pipefail

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Submit directory: $SLURM_SUBMIT_DIR"
echo "SLURM_CPUS_PER_TASK = ${SLURM_CPUS_PER_TASK}"

# -------------------------------------------------------------------
# Modules
# -------------------------------------------------------------------
module purge
module load StdEnv/2023
module load python/3.10

python two_layer_powerlaw_kp_pypde.py

echo "Job finished at: $(date)"
