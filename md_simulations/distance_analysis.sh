#!/bin/bash -l
#SBATCH -o ./job.out.%j
#SBATCH -e ./job.err.%j
#SBATCH -D ./
#SBATCH -J pp
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

echo "output_folder=$output_folder"
echo "protein_name=$protein_name"

# Guard: if simulation failed (no checkpoint), log and exit cleanly so chain continues
failed_log="/ptmp/adlouet/mass_produce_md_simulations_idrome/simulations/failed_chains.log"
if [[ ! -f "rep_0/prod/state.cpt" ]]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S'),${chain_num:-unknown},${line_num:-unknown},${zinc_id:-unknown},${protein_name:-unknown},$(pwd),sim_failed" >> "$failed_log"
    echo "CHAIN FAILED: simulation did not complete (rep_0/prod/state.cpt missing). Logged to $failed_log"
    exit 0
fi


printf "1 | 13\n1\nq\n" | gmx_mpi make_ndx -f rep_0/prod/prod_final.tpr

for y in rep*/; do
  cd "${y}/prod" || continue
  echo "  Concatenating parts in $y/prod ..."
  gmx_mpi trjcat -f traj_comp.part* -o concatenated.xtc -cat
  echo 'Protein_MOL0' | gmx_mpi trjconv -f concatenated.xtc  -s *tpr -pbc mol -ur compact -o pbc_mol.xtc -n ../../index.ndx
  cd ../..   # back to ph*/ directory
done

# Step 2: concatenate all replica trajectories into one file
echo "Combining all replicas for $x ..."
gmx_mpi trjcat -f rep*/prod/pbc_mol.xtc -o all_concatenated.xtc -cat
echo 'Protein_MOL0' | gmx_mpi editconf -f rep_0/prod/prod_final.tpr -o template.pdb -n index.ndx

conda activate mass_pro
PWD_VAR=$(pwd)
ZINC_VAR="${PWD_VAR##*/}"
PRO_VAR="$(basename "$(dirname "$PWD_VAR")")"
python distance_cal.py "$PWD_VAR" "$protein_name" "$output_folder"
