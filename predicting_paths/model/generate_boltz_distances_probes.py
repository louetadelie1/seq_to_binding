"""
Generate MD-independent CA-CA distance matrices for the post_processed_probes
targets, using Boltz structure prediction instead of MD -- same approach as
generate_boltz_distances.py (see that file for the full rationale), but for
the much larger probes set (~9.9k unique sequences vs ~10 for the IDPs).

Because of the scale, this script is shard-aware: run one process per GPU,
each handling a disjoint slice of the unique sequences, so N processes cover
the set in ~1/N the wall-clock time. Sequences are sorted longest-first and
dealt round-robin across shards so each shard gets a similar total workload
rather than, say, all the long sequences landing on one shard.

Run inside the boltz_env conda environment (has both `boltz` and biopython),
one invocation per GPU (see run_probes_boltz.sh for the launcher that starts
all of them):
    conda activate /ptmp/adlouet/software/boltz_env
    PYTHONNOUSERSITE=1 python generate_boltz_distances_probes.py --gpu 0 --shard 0 --num_shards 3
    PYTHONNOUSERSITE=1 python generate_boltz_distances_probes.py --gpu 1 --shard 1 --num_shards 3
    PYTHONNOUSERSITE=1 python generate_boltz_distances_probes.py --gpu 2 --shard 2 --num_shards 3

This calls `boltz predict` as a subprocess per unique sequence (not the
internal API), matching the CLI usage already validated interactively.
"""

import argparse
import os
import re
import time
import subprocess
import pickle
import collections

import numpy as np
from scipy.spatial.distance import cdist
from Bio.SeqUtils import seq1

PROBES_BASE   = "/ptmp/adlouet/camb/sequence_to_binding_paths/post_processed_probes"
# NOTE: the real prediction cache lives under boltz_stuff/ (9.8k cached
# boltz_results_* dirs) -- WORK_DIR previously pointed at a path that was
# never populated (fine_tuning_experiments/exp_4_boltz_distances, no
# "_prediction_only" suffix, doesn't exist), which would have silently
# discarded the existing cache and re-predicted everything from scratch.
WORK_DIR      = "/ptmp/adlouet/camb/sequence_to_binding_paths/boltz_stuff"
FASTA_DIR     = os.path.join(WORK_DIR, "boltz_inputs_probes")
OUT_DIR       = os.path.join(WORK_DIR, "boltz_outputs_probes")

# A separate, unrelated probe-set variant nested inside PROBES_BASE (own
# p_eq_keys.pkl sampling, no ca_dist/jacobian outputs at all) -- excluded so
# os.walk below doesn't silently sweep it in alongside the top-level probes.
EXCLUDE_SUBDIR = "post_processed_probes_skip_10_triplets"
BOLTZ_CACHE   = "/ptmp/adlouet/software/boltz_cache"
BOLTZ_BIN     = "/ptmp/adlouet/software/boltz_env/bin/boltz"

DIFFUSION_SAMPLES = 10   # ensemble size for mean/std distance estimate

os.makedirs(FASTA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)


def discover_unique_sequences(base):
    """Walk post_processed_probes, group every directory containing a
    sequence.fasta by its sequence. Returns {seq: [dirs]}.

    Probe directories without a sequence.fasta (~2100 of them) are skipped,
    same limitation as the IDP script -- their sequence would need to be
    derived from the PDB/template file instead.
    """
    seq_to_dirs = collections.defaultdict(list)
    for root, dirs, files in os.walk(base):
        if EXCLUDE_SUBDIR in dirs:
            dirs.remove(EXCLUDE_SUBDIR)
        if 'sequence.fasta' in files:
            with open(os.path.join(root, 'sequence.fasta')) as f:
                lines = [l.strip() for l in f if l.strip()]
            if len(lines) >= 2 and lines[0].startswith('>'):
                seq_to_dirs[lines[1]].append(root)
    return seq_to_dirs


def safe_name(seq_dirs):
    """Derive a filesystem-safe name for a sequence from its first directory,
    e.g. post_processed_probes/11773_6OTN -> '11773_6OTN'."""
    example = seq_dirs[0]
    rel = os.path.relpath(example, PROBES_BASE)
    top = rel.split(os.sep)[0]
    return re.sub(r'[^A-Za-z0-9_.-]', '_', top)


def write_boltz_fasta(name, seq, path):
    # Boltz fasta format: >{chain_id}|{entity_type}|{msa_path, blank = auto}
    with open(path, 'w') as f:
        f.write(f">A|protein|\n{seq}\n")


def run_boltz_predict(fasta_path, out_dir, gpu):
    """Run `boltz predict`, streaming its output live (same as before) while
    also watching for boltz's own CUDA-OOM message. Returns True if an OOM
    was seen. boltz catches OOM per-batch internally and exits 0 with
    "Number of failed examples: 1" rather than raising -- that's a
    deterministic result of sequence length vs GPU memory, not a transient
    failure, so callers should not blindly retry it (see max_retries below).
    """
    env = dict(os.environ)
    env["PYTHONNOUSERSITE"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = [
        BOLTZ_BIN, "predict", fasta_path,
        "--out_dir", out_dir,
        "--cache", BOLTZ_CACHE,
        "--diffusion_samples", str(DIFFUSION_SAMPLES),
        # Run diffusion samples one at a time instead of the default (all
        # DIFFUSION_SAMPLES batched together) -- long probe sequences
        # (~1000+ residues) were OOMing on the 24GB RTX6000s at full
        # parallelism. Sequential is slower per-sequence but avoids that
        # entirely, which matters more for this leftover, longer-tailed slice.
        "--max_parallel_samples", "1",
        "--use_msa_server",
        "--output_format", "pdb",
    ]
    print("  $", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
    oom = False
    for line in proc.stdout:
        print(line, end='', flush=True)
        if 'out of memory' in line.lower():
            oom = True
    proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    return oom


ANGSTROM_TO_NM = 0.1  # the existing MD ca_dist_matrix.pkl files are in nm (bonded
                       # CA-CA ~0.386 nm); PDB coordinates are in Angstrom, so this
                       # matches units with what exp_2 was actually trained on.


def ca_distance_matrix_from_pdb(pdb_path):
    from Bio.PDB import PDBParser
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('m', pdb_path)
    chain = next(iter(structure[0]))
    residues = [r for r in chain if 'CA' in r]
    coords = np.array([r['CA'].coord for r in residues])
    labels = [f"{r.get_resname()}{r.id[1]}" for r in residues]
    seq = ''.join(seq1(r.get_resname()) for r in residues)
    return cdist(coords, coords) * ANGSTROM_TO_NM, labels, seq


def collect_ensemble_distances(pred_dir, name, n_samples):
    """boltz predict --output_format pdb writes {name}_model_{i}.pdb per
    diffusion sample under out_dir/boltz_results_{name}/predictions/{name}/."""
    pred_subdir = os.path.join(pred_dir, f"boltz_results_{name}", "predictions", name)
    mats, labels_ref, seq_ref = [], None, None
    for i in range(n_samples):
        pdb_path = os.path.join(pred_subdir, f"{name}_model_{i}.pdb")
        if not os.path.exists(pdb_path):
            continue
        mat, labels, seq = ca_distance_matrix_from_pdb(pdb_path)
        mats.append(mat)
        if labels_ref is None:
            labels_ref, seq_ref = labels, seq
    if not mats:
        return None
    mats = np.array(mats)
    return {
        'matrix':      mats.mean(0),           # mean CA-CA distance (Angstrom) across samples
        'std_matrix':  mats.std(0),            # spread across diffusion samples
        'labels':      labels_ref,
        'seq':         seq_ref,
        'n_samples':   len(mats),
        'source':      f'boltz-1, {len(mats)} diffusion samples, ColabFold MSA server',
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, required=True, help="CUDA device index to pin this process to")
    parser.add_argument('--shard', type=int, required=True, help="This process's shard index (0-based)")
    parser.add_argument('--num_shards', type=int, required=True, help="Total number of shards")
    args = parser.parse_args()
    tag = f"[shard {args.shard}/{args.num_shards} gpu {args.gpu}]"

    seq_to_dirs = discover_unique_sequences(PROBES_BASE)
    items = sorted(seq_to_dirs.items(), key=lambda kv: safe_name(kv[1]))  # deterministic order first
    items.sort(key=lambda kv: -len(kv[0]))  # longest-first, then dealt round-robin -> balanced shards
    my_items = [it for i, it in enumerate(items) if i % args.num_shards == args.shard]

    print(f"{tag} {len(items)} unique sequences total, "
          f"{len(my_items)} assigned to this shard "
          f"({sum(len(v) for _, v in my_items)} target dirs)", flush=True)

    for seq, dirs in my_items:
        name = safe_name(dirs)
        fasta_path = os.path.join(FASTA_DIR, f"{name}.fasta")
        result_dir = os.path.join(OUT_DIR, f"boltz_results_{name}")

        already_all_written = all(
            os.path.exists(os.path.join(d, 'ca_dist_matrix_boltz.pkl')) for d in dirs
        )
        if already_all_written:
            print(f"{tag}   {name}: ca_dist_matrix_boltz.pkl already present in all "
                  f"{len(dirs)} target dirs, skipping", flush=True)
            continue

        print(f"\n{tag} === {name}  (len={len(seq)}, {len(dirs)} target dirs) ===", flush=True)

        try:
            pred_subdir = os.path.join(result_dir, "predictions", name)
            already_done = (
                os.path.isdir(pred_subdir)
                and os.path.exists(os.path.join(pred_subdir, f"{name}_model_0.pdb"))
            )
            if already_done:
                print(f"{tag}   already predicted, reusing {result_dir}", flush=True)
                ensemble = collect_ensemble_distances(OUT_DIR, name, DIFFUSION_SAMPLES)
            else:
                # boltz predict can exit 0 while silently producing nothing
                # (GPU OOM mid-run, or a truncated download from the shared
                # ColabFold MSA server under concurrent load) -- neither
                # raises, so success is checked afterwards by looking for
                # actual output, and retried here rather than relying on
                # subprocess exit code.
                ensemble = None
                max_retries = 3
                for attempt in range(1, max_retries + 1):
                    if os.path.exists(result_dir):
                        import shutil
                        shutil.rmtree(result_dir)
                    write_boltz_fasta(name, seq, fasta_path)
                    oom = run_boltz_predict(fasta_path, OUT_DIR, args.gpu)
                    ensemble = collect_ensemble_distances(OUT_DIR, name, DIFFUSION_SAMPLES)
                    if ensemble is not None:
                        break
                    if oom:
                        # Deterministic given (sequence length, GPU memory) --
                        # retrying the identical input on the identical GPU
                        # will OOM again, so don't waste the other 2 attempts.
                        print(f"{tag}   attempt {attempt}/{max_retries} OOM'd for {name} "
                              f"(len={len(seq)}) -- not retrying, too long for this GPU", flush=True)
                        break
                    print(f"{tag}   attempt {attempt}/{max_retries} produced no output for {name}"
                          f"{', retrying' if attempt < max_retries else ''}", flush=True)
                    time.sleep(10)

            if ensemble is None:
                print(f"{tag}   WARNING: no predicted structures found for {name}, skipping", flush=True)
                continue

            if ensemble['seq'] != seq:
                print(f"{tag}   WARNING: Boltz output sequence differs from input for {name} "
                      f"(len {len(ensemble['seq'])} vs {len(seq)})", flush=True)

            for d in dirs:
                out_path = os.path.join(d, 'ca_dist_matrix_boltz.pkl')
                with open(out_path, 'wb') as f:
                    pickle.dump(ensemble, f)

            print(f"{tag}   wrote ca_dist_matrix_boltz.pkl to {len(dirs)} directories "
                  f"(mean std across pairs: {ensemble['std_matrix'].mean():.2f} A)", flush=True)

        except Exception as e:
            print(f"{tag}   FAILED for {name}, skipping to next target: {e}", flush=True)
            continue


if __name__ == '__main__':
    main()
