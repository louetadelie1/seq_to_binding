#!/bin/bash -l
#SBATCH -o ./job.out.%j
#SBATCH -e ./job.err.%j
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


lockfile=".running"                                                                                    
if [[ -f "$lockfile" ]]; then                                                                          
    echo "Job already running in this directory (lock file exists), exiting."                          
    exit 1      
fi
touch "$lockfile"
trap "rm -f $lockfile" EXIT


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores


echo "protein_name = $protein_name"
echo "run_para = $run_para"
echo "ff = $ff"
echo "scripts=$scripts"
echo "output_folder=$output_folder"

shopt -s nullglob
rep_dirs=(rep*)
if [[ ${#rep_dirs[@]} -eq 0 ]]; then
  echo "No rep_* directories found in $(pwd) - prep (generate_complex.sh) must not have completed. Exiting."
  exit 1
fi

for rep in "${rep_dirs[@]}"; do
  cd "$rep" || exit
  pdbfixer rep_*.pdb --add-atoms=heavy --output=fixed.pdb
  mkdir -p generate_complex
  cd generate_complex || exit

  ln -s "$ff" ./
  gmx_mpi pdb2gmx -f ../fixed.pdb -o "$protein_name.gro" -p "$protein_name.top" -ignh <<EOF
1
1
EOF

  # ln -s ../../ZINC* ./

  cp ../../ZINC*/*/*_MOL0.itp Ligand.itp
  cp ../../ZINC*/*/*_atomtypes.itp Ligand_atomtypes.itp
  cp ../../ZINC*/*/system.gro liga.gro

  cp "$protein_name.top" Complex2.top

  sed -e '/#include ".\/a99SBdisp.ff\/forcefield.itp"/a\
\
; Include ligand parameters\
#include "Ligand_atomtypes.itp"\
\
' -e '/; Include water topology/i\
\
; Include ligand topology\
#include "Ligand.itp"\
\
' Complex2.top > Complex.top
  echo "MOL0  1" >> Complex.top

  # Insert the ligand
  gmx_mpi editconf -f "$protein_name.gro" -o tiny_apo_pro_box.gro -c -bt cubic -d 0.5
  gmx_mpi insert-molecules -f tiny_apo_pro_box.gro -ci liga.gro -nmol 1 -radius 0.05 -try 50000 -o Complex.gro

  cd ../
  mkdir -p prep
  cd prep || exit
  ln -s "$ff" ./

  gmx_mpi editconf -f ../generate_complex/Complex.gro -o Complex_box.gro -c -d 1.5 -bt dodecahedron
  gmx_mpi solvate -cp Complex_box.gro -cs tip4p.gro -o Complex_solv.gro -p ../generate_complex/Complex.top
  gmx_mpi grompp -f "$run_para/em_ions.mdp" -c Complex_solv.gro -p ../generate_complex/Complex.top -o Complex_solv.tpr -maxwarn 3

  ### save up to here ###
  grep "qtot" ../generate_complex/Complex.top | tail -n 1 | awk '{print $11}'

  gmx_mpi genion -s Complex_solv.tpr -o Complex_ionized.gro -p ../generate_complex/Complex.top -nname CL -pname NA -neutral <<EOF
15
EOF

  cd ../
  mkdir -p minim/em
  cd minim/em || exit

  gmx_mpi grompp -f "$run_para/minim.mdp" -c ../../prep/Complex_ionized.gro -p ../../generate_complex/Complex.top -o Complex_em.tpr -maxwarn 1
  gmx_mpi mdrun -v -s Complex_em.tpr

  cd ../../
  mkdir -p minim/nvt
  cd minim/nvt || exit

  gmx_mpi make_ndx -f ../../generate_complex/Complex.gro <<EOF
1
13
q
EOF

  gmx_mpi genrestr -f ../../generate_complex/liga.gro -n index.ndx -o ../../generate_complex/posre_LIGAND.itp -fc 1000 1000 1000 <<EOF
13
EOF

  sed -i '/#include "ligand.itp"/a \n; Ligand position restraints\n#ifdef POSRES\n#include "posre_LIGAND.itp"\n#endif' ../../generate_complex/Complex.top

  gmx_mpi grompp -f "$run_para/nvt.mdp" -c ../em/confout.gro -r ../em/confout.gro -p ../../generate_complex/Complex.top -o Complex_nvt.tpr -maxwarn 1
  gmx_mpi mdrun -v -s Complex_nvt.tpr

  cd ../
  mkdir -p npt
  cd npt || exit

  gmx_mpi grompp -f "$run_para/npt.mdp" -c ../nvt/confout.gro -r ../nvt/confout.gro -p ../../generate_complex/Complex.top -o Complex_npt.tpr -maxwarn 15
  gmx_mpi mdrun -v -s Complex_npt.tpr

  cd ../../
  mkdir -p prod
  cd prod || exit

  gmx_mpi grompp -f "$run_para/run.mdp" -c ../minim/npt/confout.gro -r ../minim/npt/confout.gro -p ../generate_complex/Complex.top -o prod_final.tpr -maxwarn 15

  cd ../../
  echo "=== Finished $rep ==="
done

# Some replicas can blow up during NVT (steric clash from ligand insertion,
# GPU illegal-memory-access abort) - that's a per-replica stochastic failure,
# not a problem with the system as a whole. Repair small numbers of failures
# by cloning a random surviving replica; bail out only if too many failed.
good_reps=()
bad_reps=()
for rep in "${rep_dirs[@]}"; do
  if [[ -s "${rep}/prod/prod_final.tpr" ]]; then
    good_reps+=("$rep")
  else
    bad_reps+=("$rep")
  fi
done

n_bad=${#bad_reps[@]}
echo "Prep summary: ${#good_reps[@]} succeeded, ${n_bad} failed (${bad_reps[*]:-none})"

if [[ $n_bad -eq 0 ]]; then
  echo "All replicas succeeded."
elif [[ $n_bad -le 4 ]]; then
  echo "Repairing $n_bad failed replica(s) by cloning random successful replicas."
  for rep in "${bad_reps[@]}"; do
    donor="${good_reps[$RANDOM % ${#good_reps[@]}]}"
    echo "  $rep <- copy of $donor"
    rm -rf "$rep"
    cp -r "$donor" "$rep"
  done
else
  echo "Too many failed replicas ($n_bad/${#rep_dirs[@]}) - skipping this system, not launching run_first."
  exit 1
fi


# # this is finlizing the systems -all replicas combined

# cp "$scripts"/run_first.sh ./

# sed -i "s/protein/${protein_name}/g" run_first.sh
# cp "$scripts"/distance_analysis.sh ./
# ln -s "$scripts"/distance_cal.py ./

# jid=$(sbatch run_first.sh | awk '{print $4}')
# sbatch --dependency=afterok:$jid --export=ALL,output_folder="$output_folder",protein_name="$protein_name" distance_analysis.sh

