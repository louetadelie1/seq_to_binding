#!/bin/bash -l
#SBATCH -o ./job.out.%j
#SBATCH -e ./job.err.%j
#SBATCH -D ./
#SBATCH -J protein
#SBATCH --time=24:00:00
#SBATCH --nodes=8
#SBATCH --constraint="apu"
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=23
#SBATCH --array=1-2:1%1

module purge
module load gcc/13 cmake/3.30 openmpi/5.0
module load gromacs/2025.1

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export GMX_FORCE_UPDATE_DEFAULT_GPU=1
export GMX_DISABLE_GPU_TIMING=1

# Guard: if prep (run_all_reps_together) failed, log and exit cleanly so chain continues.
# Checks every rep_*, not just rep_0 - run_all_reps_together.sh repairs up to 4 failed
# replicas by cloning a survivor, but if it bailed out (too many failures), any missing
# prod_final.tpr breaks the whole -multidir mdrun below, so we must not launch it.
shopt -s nullglob
missing_reps=()
for rep in rep_*/; do
    rep="${rep%/}"
    if [[ ! -f "${rep}/prod/prod_final.tpr" ]]; then
        missing_reps+=("$rep")
    fi
done
shopt -u nullglob

if [[ ${#missing_reps[@]} -gt 0 ]]; then
    failed_log="/ptmp/adlouet/mass_produce_md_simulations_idrome/simulations/failed_chains.log"
    echo "$(date '+%Y-%m-%d %H:%M:%S'),${chain_num:-unknown},${line_num:-unknown},${zinc_id:-unknown},${protein_name:-unknown},$(pwd),prep_failed_missing_${missing_reps[*]}" >> "$failed_log"
    echo "CHAIN FAILED: prep did not complete for replicas: ${missing_reps[*]}. Logged to $failed_log"
    exit 0
fi

srun --mpi=pmix gmx_mpi mdrun -v -ntomp $OMP_NUM_THREADS -s prod_final.tpr -multidir rep_*/prod/  -maxh 23.25 -nb gpu -noappend -cpi state
