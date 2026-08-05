#!/bin/bash -l
#SBATCH -o ./error_logs/postprocess.out.%A_%a
#SBATCH -e ./error_logs/postprocess.err.%A_%a
#SBATCH -D ./
#SBATCH -J pp_recover
#SBATCH --time=04:00:00
#SBATCH --constraint="apu"
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-9

base="/ptmp/adlouet/mass_produce_md_simulations_idrome"
simulation_dir="${base}/simulations"
output_post_process="${base}/output_post_processed"
scripts="${base}/vipergpu_scripts"

# systems confirmed to still have all_concatenated.xtc + template.pdb intact
dirs=(
  "P6/12/44/119_160:P6_12_44_119_160"
  "A3/KM/H1/1384_1419:A3_KM_H1_1384_1419"
  "Q9/NX/E4/1_54:Q9_NX_E4_1_54"
  "Q9/BX/R6/1_32:Q9_BX_R6_1_32"
  "Q9/BY/X4/637_673:Q9_BY_X4_637_673"
  "Q9/Y6/98/38_93:Q9_Y6_98_38_93"
  "Q9/6Q/D8/245_283:Q9_6Q_D8_245_283"
  "Q1/28/49/1_152:Q1_28_49_1_152"
  "Q8/6U/38/1_59:Q8_6U_38_1_59"
  "Q8/WZ/64/869_911:Q8_WZ_64_869_911"
)

entry="${dirs[$SLURM_ARRAY_TASK_ID]}"
rel_path="${entry%%:*}"
protein_name="${entry##*:}"

sim_path="${simulation_dir}/${rel_path}"

echo "Post-processing ${protein_name} at ${sim_path}"
cd "$sim_path" || exit 1

source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate mass_pro

python "${scripts}/distance_cal.py" "$sim_path" "$protein_name" "$output_post_process"

echo "Done ${protein_name}"
