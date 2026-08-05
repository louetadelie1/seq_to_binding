#!/bin/bash -l
#SBATCH -o ./watch_and_launch.job.out.%j
#SBATCH -e ./watch_and_launch.job.err.%j
#SBATCH -D ./
#SBATCH -J watch_launch
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --constraint="apu"
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#
# Watcher submitted as its own SLURM job so it runs independent of any
# SSH/Claude session - survives disconnects, since SLURM jobs don't depend
# on any login session staying open.
#
# Waits for the chain-0 test job (run_first, job 10814292) to actually start
# running - proof that the prep->run_first handoff (prod_final.tpr for all
# 16 reps) worked - then launches chains 1-7.

scripts="/ptmp/adlouet/mass_produce_md_simulations_idrome/vipergpu_scripts"
sim_dir="/ptmp/adlouet/mass_produce_md_simulations_idrome/simulations/Q9/Y2/28/422_466"
failed_log="/ptmp/adlouet/mass_produce_md_simulations_idrome/simulations/failed_chains.log"
log="${scripts}/watch_and_launch.log"

echo "$(date '+%Y-%m-%d %H:%M:%S') watcher started, PID $$, watching job 10814292" >> "$log"

before_lines=$(wc -l < "$failed_log" 2>/dev/null || echo 0)

while true; do
    state=$(squeue -j 10814292 -h -o "%T" 2>/dev/null | head -1)
    if [[ -z "$state" ]]; then
        state=$(sacct -j 10814292 -X -n -o State 2>/dev/null | head -1 | xargs)
    fi

    if [[ "$state" == "RUNNING" || "$state" == "COMPLETED" ]]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') SUCCESS: run_first (10814292) reached state=$state - prep handoff worked." >> "$log"
        for i in {1..7}; do
            jid=$(sbatch --chdir="$scripts" "${scripts}/generate_complex_submit.sh" "$i" | awk '{print $4}')
            echo "$(date '+%Y-%m-%d %H:%M:%S') launched chain $i as job $jid" >> "$log"
        done
        echo "$(date '+%Y-%m-%d %H:%M:%S') all chains launched, watcher exiting." >> "$log"
        exit 0
    fi

    if [[ "$state" == "FAILED" || "$state" == "CANCELLED" || "$state" == "TIMEOUT" || "$state" == "NODE_FAIL" ]]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') FAILURE: run_first (10814292) ended in state=$state without ever running. NOT launching chains 1-7." >> "$log"
        exit 1
    fi

    now_lines=$(wc -l < "$failed_log" 2>/dev/null || echo 0)
    if [[ "$now_lines" -gt "$before_lines" ]]; then
        new_entries=$(tail -n $((now_lines - before_lines)) "$failed_log")
        before_lines=$now_lines
        if echo "$new_entries" | grep -q "Q9_Y2_28_422_466\|Q9/Y2/28/422_466"; then
            echo "$(date '+%Y-%m-%d %H:%M:%S') FAILURE: test chain Q9_Y2_28_422_466 logged a failure:" >> "$log"
            echo "$new_entries" >> "$log"
            echo "$(date '+%Y-%m-%d %H:%M:%S') NOT launching chains 1-7." >> "$log"
            exit 1
        fi
    fi

    sleep 30
done
