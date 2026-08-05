
module purge
module load gcc/13 cmake/3.30 openmpi/5.0
module load gromacs/2025.1

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export GMX_FORCE_UPDATE_DEFAULT_GPU=1
export GMX_DISABLE_GPU_TIMING=1

chain_num=$1
line_num=${2:-2}

echo "chain_num = $chain_num, line_num = $line_num"

base="/ptmp/adlouet/mass_produce_md_simulations_idrome"
input_pro_lig="${base}/batch_2/chain_batch_${chain_num}.csv"
simulation_dir="${base}/simulations"
scripts="${base}/vipergpu_scripts"
run_para="${base}/vipergpu_scripts/run_parameters"
ff="${base}/vipergpu_scripts/forcefields/a99SBdisp.ff"
output_post_process="${base}/output_post_processed"
this_file="${scripts}/generate_complex.sh"

tracking_file="${simulation_dir}/running_trajectories_chain_${chain_num}.txt"


if [[ ! -f "$tracking_file" ]]; then
    echo "ZINC_ID,SMILES,path" > "$tracking_file"
fi

failed_log="${simulation_dir}/failed_chains.log"
if [[ ! -f "$failed_log" ]]; then
    echo "timestamp,chain_num,line_num,zinc_id,protein_name,sim_dir,stage" > "$failed_log"
fi

#if [[ ! -f "$error_file" ]]; then
#    echo "ZINC_ID,SMILES,path" > "$error_file"
#fi


while true; do
    input_line=$(sed -n "${line_num}p" "$input_pro_lig")
    [[ -z "$input_line" ]] && echo "No data at line $line_num, all entries processed or exhausted" && exit 0
    IFS=',' read -r zinc_id smiles full_path <<< "$input_line"

    if grep -q "^${zinc_id}," "$tracking_file" 2>/dev/null; then
        echo "System $zinc_id already in tracking file, skipping to next (line $((line_num + 1)))..."
        line_num=$((line_num + 1))
        continue
    fi

    if [[ ! -f "${full_path}/traj.xtc" || ! -f "${full_path}/top.pdb" ]]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S'),${chain_num},${line_num},${zinc_id},unknown,${full_path},missing_source_data" >> "$failed_log"
        echo "${zinc_id},${smiles},${full_path}" >> "$tracking_file"
        echo "Source data missing for $zinc_id at $full_path (traj.xtc/top.pdb not found), skipping to next (line $((line_num + 1)))..."
        line_num=$((line_num + 1))
        continue
    fi

    break
done


rel_path="${full_path#/ptmp/adlouet/IDRome_v4}"
new_dir="${simulation_dir}${rel_path}"
protein_name="${rel_path//\//_}"
protein_name="${protein_name#_}"

echo "Processing: $zinc_id | $protein_name"

mkdir -p "$new_dir"
cd "$new_dir" || exit

echo "Protein" | gmx_mpi trjconv -f "${full_path}/traj.xtc" -s "${full_path}/top.pdb" -o pair.pdb -sep -skip 63

echo "${zinc_id},${smiles},${full_path}" > "${new_dir}/lig.txt"
echo "${zinc_id},${smiles},${full_path}" >> "$tracking_file"

if [[ ! -f "pair0.pdb" ]]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S'),${chain_num},${line_num},${zinc_id},${protein_name},${new_dir},trjconv_failed" >> "$failed_log"
    echo "trjconv produced no frames for $zinc_id at $full_path. Skipping to next line ($((line_num + 1))) instead of stalling the chain."
    sbatch --chdir="$scripts" "${scripts}/generate_complex_submit.sh" "$chain_num" "$((line_num + 1))"
    exit 1
fi

for i in {0..15}; do
    mv "pair${i}.pdb" "rep_${i}.pdb"
done

shopt -s nullglob
for x in *.pdb; do
    mkdir -p "${x%.*}" && mv "$x" "${x%.*}/"
done
shopt -u nullglob


current_dir=$(pwd)
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate mass_pro

echo "Processing zinc param"
python "$scripts/ligand_parameters.py" "$zinc_id" "$smiles" "$current_dir" "$protein_name"
echo "Done processing zinc param"

cp "$scripts/run_all_reps_together.sh" ./
cp "$scripts/run_first.sh" ./
cp "$scripts/distance_analysis.sh" ./
ln -sf "$scripts/distance_cal.py" ./

sed -i "s/protein/${protein_name}/g" run_first.sh

# chain: run_all_reps_together -> run_first -> distance_analysis -> next row
# afterany deps ensure the chain always continues even if a step fails
jid1=$(sbatch --export=ALL,run_para="$run_para",ff="$ff",protein_name="$protein_name",scripts="$scripts",output_folder="$output_post_process" run_all_reps_together.sh | awk '{print $4}')

jid2=$(sbatch --dependency=afterany:"$jid1" --export=ALL,chain_num="$chain_num",zinc_id="$zinc_id",line_num="$line_num",protein_name="$protein_name" run_first.sh | awk '{print $4}')

jid3=$(sbatch --dependency=afterany:"$jid2" --export=ALL,output_folder="$output_post_process",protein_name="$protein_name",chain_num="$chain_num",zinc_id="$zinc_id",line_num="$line_num" distance_analysis.sh | awk '{print $4}')

sbatch --chdir="$scripts" --dependency=afterany:"$jid3" "${scripts}/generate_complex_submit.sh" "$chain_num" "$((line_num + 1))"

## Validation: check for expected output file
#validation_script="expected_file=\"${output_post_process}/${protein_name}/distances/d_24_t_closest.pkl\"

#if [[ -f \"\$expected_file\" ]]; then
#    echo \"SUCCESS: ${protein_name} completed with output\"
#else
#    echo \"${zinc_id},${smiles},${full_path},missing_output\" >> \"${error_file}\"
#    echo \"ERROR: ${protein_name} missing expected output file\"
#fi

## Continue to next system (increment by 8 for parallel chains)
#bash ${this_file} $((line_num + 8))
#"

#sbatch --dependency=afterany:"$jid3" --time=00:05:00 --nodes=1 --cpus-per-task=1 --job-name=validate_${protein_name} --wrap="$validation_script"

# protein_name="$P1_71_81_1_37"
# run_para='/ptmp/adlouet/protein_simulations/vipercpu_scripts/run_parameters'
# ff='/ptmp/adlouet/protein_simulations/vipercpu_scripts/forcefields/a99SBdisp.ff'
# scripts='/ptmp/adlouet/protein_simulations/vipercpu_scripts'
