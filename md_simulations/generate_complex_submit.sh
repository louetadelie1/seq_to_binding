#!/bin/bash -l
#SBATCH -o ./error_logs/job.out.%j
#SBATCH -e ./error_logs/job.err.%j
#SBATCH -D ./
#SBATCH -J run_prep
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --constraint="apu"
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4

module purge
module load gcc/13 cmake/3.30 openmpi/5.0
module load gromacs/2025.1

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export GMX_FORCE_UPDATE_DEFAULT_GPU=1
export GMX_DISABLE_GPU_TIMING=1

scripts="/ptmp/adlouet/mass_produce_md_simulations_idrome/vipergpu_scripts"

bash "${scripts}/generate_complex.sh" $1 $2
