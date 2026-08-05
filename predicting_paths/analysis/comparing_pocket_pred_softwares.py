import os
import sys
import csv
import glob
import zipfile
import pickle
import math
import statistics
import itertools
import numpy as np
import mdtraj as md
import matplotlib.pyplot as plt
import seaborn as sns
import glob

BASE_DIR= "/ptmp/adlouet/camb/sequence_to_binding_paths/comparing_binding_pocket_pred_softwares/"

# ── shared blue-themed publication style, same module every other figure in
# the paper imports from -- pull it in directly (rather than keeping a local
# hand-copied palette that can silently drift) so this figure is guaranteed
# to match ─────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'paper_final_figures_code'))
import plot_style as ps
TOPOLOGY_FILE     = os.path.join(BASE_DIR, 'template_AB.pdb')
TRAJECTORY_FILE   = os.path.join(BASE_DIR, 'traj_all-skip-0-noW_AB.xtc')
HANDOFF_PKL       = os.path.join(BASE_DIR, 'ab_handoff.pkl')
AB42_PATH_TXT     = os.path.join(BASE_DIR, 'my_output', 'ab42_1.txt')
MEAN_DISTANCE_PKL = os.path.join(BASE_DIR, 'mean_ca_distance_matrix.pkl')
SCRATCH_DIR       = os.path.join(BASE_DIR, '_p2rank_scratch')

NUM_TOP_RESIDUES = 12      # for tools that only give a per-residue score, how many top residues to consider
DISTANCE_CUTOFF  = 6.0     # Angstroms - residues closer than this get grouped into the same pocket
POOL_SIZE        = 20000   # how many trajectory frames to subsample down to for the ensemble calculations

# ── colors pulled straight from plot_style.py, so this figure matches every
# other figure in ../paper_final_figures_code exactly ──────────────────────
PRIMARY       = ps.PRIMARY         # marginal score bars
SECONDARY     = ps.SECONDARY       # coherence score bars
TERTIARY      = ps.TERTIARY        # shuttling score bars
INK_PRIMARY   = ps.INK_PRIMARY     # titles
INK_SECONDARY = ps.INK_SECONDARY   # tick/axis labels
INK_MUTED     = ps.INK_MUTED       # footnotes
BASELINE_GRAY = ps.BASELINE_GRAY   # surviving spines
SURFACE       = ps.SURFACE         # bar edges / figure background

three_to_one = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
}

# all 5 frames must stay ACTIVE here - the whole point of this version is to
# evaluate every possible combination of them afterwards, so we need all 5
# frames' pockets collected up front, not a pre-trimmed subset
frame_info = [
    ('frame_1', 'frame_1_origidx_0'),
    ('frame_2', 'frame_2_origidx_58968'),
    ('frame_3', 'frame_3_origidx_224328'),
    ('frame_4', 'frame_4_origidx_36852'),
    ('frame_5', 'frame_5_origidx_62592'),
]


# ── pick 5 structurally diverse frames from the trajectory ─────────────────
already_have_frames = all(
    len(glob.glob(os.path.join(BASE_DIR, 'ScanNet', frame_folder, '*.pdb'))) > 0
    for frame_folder, _ in frame_info
)

if not already_have_frames:
    topology = md.load_topology(TOPOLOGY_FILE)
    ca_atom_indices = topology.select('name CA')

    with md.formats.XTCTrajectoryFile(TRAJECTORY_FILE, 'r') as xtc_file:
        total_frames = len(xtc_file)

    stride = max(1, total_frames // POOL_SIZE)
    pool_traj = md.load(TRAJECTORY_FILE, top=TOPOLOGY_FILE, atom_indices=ca_atom_indices, stride=stride)

    n_ca = pool_traj.n_atoms
    residue_pairs = np.array([[i, j] for i in range(n_ca) for j in range(i + 1, n_ca)])
    distance_descriptors = md.compute_distances(pool_traj, residue_pairs)

    selected_pool_indices = [0]
    min_dist_to_selected = np.sqrt(np.sum((distance_descriptors - distance_descriptors[0]) ** 2, axis=1))
    while len(selected_pool_indices) < 5:
        next_index = int(np.argmax(min_dist_to_selected))
        selected_pool_indices.append(next_index)
        dist_to_new = np.sqrt(np.sum((distance_descriptors - distance_descriptors[next_index]) ** 2, axis=1))
        min_dist_to_selected = np.minimum(min_dist_to_selected, dist_to_new)

    selected_original_frame_indices = [pool_index * stride for pool_index in selected_pool_indices]
    print('STEP 1: selected trajectory frames:', selected_original_frame_indices)

    for rank, orig_index in enumerate(selected_original_frame_indices, start=1):
        full_frame = md.load_frame(TRAJECTORY_FILE, index=orig_index, top=TOPOLOGY_FILE)
        full_frame.save_pdb(os.path.join(BASE_DIR, 'ScanNet', f'frame_{rank}', f'frame_{rank}_origidx_{orig_index}.pdb'))


# ── mean pairwise C-alpha distance matrix (for the shuttling score) ────────
if not os.path.exists(MEAN_DISTANCE_PKL):
    topology = md.load_topology(TOPOLOGY_FILE)
    ca_atom_indices = topology.select('name CA')
    n_ca = len(ca_atom_indices)

    with md.formats.XTCTrajectoryFile(TRAJECTORY_FILE, 'r') as xtc_file:
        total_frames = len(xtc_file)
    stride = max(1, total_frames // POOL_SIZE)
    pool_traj = md.load(TRAJECTORY_FILE, top=TOPOLOGY_FILE, atom_indices=ca_atom_indices, stride=stride)

    residue_pairs = np.array([[i, j] for i in range(n_ca) for j in range(i + 1, n_ca)])
    distances = md.compute_distances(pool_traj, residue_pairs)
    mean_distance_per_pair = distances.mean(axis=0)

    mean_distance_matrix = np.zeros((n_ca, n_ca))
    for (i, j), mean_dist_nm in zip(residue_pairs, mean_distance_per_pair):
        mean_distance_matrix[i, j] = mean_dist_nm * 10.0   # nm -> Angstrom
        mean_distance_matrix[j, i] = mean_dist_nm * 10.0

    with open(MEAN_DISTANCE_PKL, 'wb') as f:
        pickle.dump(mean_distance_matrix, f)
else:
    with open(MEAN_DISTANCE_PKL, 'rb') as f:
        mean_distance_matrix = pickle.load(f)


# ── extract pockets from each software -- PocketMiner, CryptoSite, and
# ScanNet are not explicit pockets, and therefore require clustering of output
all_pocket_rows = []   # every pocket : [software, structure, pocket_rank, score, n_residues, residues]

for frame_folder, frame_name in frame_info:

    scannet_frame_dir = os.path.join(BASE_DIR, 'ScanNet', frame_folder)
    pdb_path = os.path.join(scannet_frame_dir, frame_name + '.pdb')

    coordinates = {}
    sequence_by_residue = {}
    with open(pdb_path) as f:
        for line in f:
            if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                residue_number = int(line[22:26])
                residue_name = line[17:20].strip()
                coordinates[residue_number] = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
                sequence_by_residue[residue_number] = three_to_one.get(residue_name, 'X')

    #  fpocket
    fpocket_out_dir = os.path.join(scannet_frame_dir, frame_name + '_out')
    info_txt_path = os.path.join(fpocket_out_dir, frame_name + '_info.txt')
    fpocket_scores = {}
    if os.path.exists(info_txt_path):
        current_pocket = None
        with open(info_txt_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith('Pocket') and ':' in line:
                    current_pocket = int(line.split()[1])
                    fpocket_scores[current_pocket] = {}
                elif line.startswith('Score') and current_pocket is not None:
                    fpocket_scores[current_pocket]['score'] = float(line.split(':')[-1].strip())
    if len(fpocket_scores) == 0:
        all_pocket_rows.append(['fpocket', frame_folder, '', '', 0, 'NO POCKETS FOUND'])
    else:
        for pocket_number in sorted(fpocket_scores.keys()):
            residues_in_pocket = set()
            with open(os.path.join(fpocket_out_dir, 'pockets', f'pocket{pocket_number}_atm.pdb')) as f:
                for line in f:
                    if line.startswith('ATOM'):
                        residues_in_pocket.add(int(line[22:26]))
            residue_labels = ','.join(f'{sequence_by_residue[r]}{r}' for r in sorted(residues_in_pocket))
            all_pocket_rows.append(['fpocket', frame_folder, pocket_number, fpocket_scores[pocket_number]['score'],
                                     len(residues_in_pocket), residue_labels])

    #  P2Rank
    zip_path = glob.glob(os.path.join(BASE_DIR, 'P2Rank', frame_folder, 'prankweb-*.zip'))[0]
    unzip_dir = os.path.join(SCRATCH_DIR, frame_folder)
    os.makedirs(unzip_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(unzip_dir)
    predictions_csv_path = glob.glob(os.path.join(unzip_dir, '*_predictions.csv'))[0]
    with open(predictions_csv_path) as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for row in reader:
            residue_ids = row['residue_ids'].strip().split()
            residue_numbers = [int(rid.split('_')[1]) for rid in residue_ids]
            residue_labels = ','.join(f"{sequence_by_residue.get(r, 'X')}{r}" for r in residue_numbers)
            all_pocket_rows.append(['P2Rank', frame_folder, int(row['rank']), float(row['score']),
                                     len(residue_numbers), residue_labels])

    #  PocketMiner, CryptoSite, ScanNet
    score_only_tools = [
        ('PocketMiner', glob.glob(os.path.join(BASE_DIR, 'PocketMiner', frame_folder, 'result_model*.pdb')), 'pdb_bfactor'),
        ('CryptoSite', glob.glob(os.path.join(BASE_DIR, 'cryptosite', frame_folder, 'cryptosite.pol*.pred')), 'pred_table'),
        ('ScanNet', glob.glob(os.path.join(scannet_frame_dir, frame_name + '_single_ScanNet_idp_noMSA', f'predictions_{frame_name}.csv')), 'scannet_csv'),
    ]

    for software_name, matching_files, file_format in score_only_tools:

        if len(matching_files) == 0:
            all_pocket_rows.append([software_name, frame_folder, '', '', 0, 'NO OUTPUT YET'])
            continue

        per_residue_scores = {}
        if file_format == 'pdb_bfactor':
            with open(matching_files[0]) as f:
                for line in f:
                    if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                        per_residue_scores[int(line[22:26])] = float(line[60:66])
        elif file_format == 'pred_table':
            with open(matching_files[0]) as f:
                next(f)   # skip header
                for line in f:
                    columns = line.split()
                    per_residue_scores[int(columns[2])] = float(columns[-1])
        elif file_format == 'scannet_csv':
            with open(matching_files[0]) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    per_residue_scores[int(row['Residue Index'])] = float(row['Binding site probability'])

        top_residues = sorted(per_residue_scores.keys(), key=per_residue_scores.get, reverse=True)[:NUM_TOP_RESIDUES]

        # start: every top residue is its own tiny pocket, then repeatedly
        # merge any two pockets that have a residue pair within DISTANCE_CUTOFF,
        # restarting the scan after every merge (list size changed underneath us)
        clusters = [[r] for r in top_residues]
        merged_something = True
        while merged_something:
            merged_something = False
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    close_enough = False
                    for res_i in clusters[i]:
                        for res_j in clusters[j]:
                            x1, y1, z1 = coordinates[res_i]
                            x2, y2, z2 = coordinates[res_j]
                            distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
                            if distance <= DISTANCE_CUTOFF:
                                close_enough = True
                    if close_enough:
                        clusters[i] = clusters[i] + clusters[j]
                        clusters.pop(j)
                        merged_something = True
                        break
                if merged_something:
                    break

        clusters = [c for c in clusters if len(c) >= 2]   # a lone residue isn't a "pocket"
        clusters.sort(key=lambda c: max(per_residue_scores[r] for r in c), reverse=True)

        if len(clusters) == 0:
            all_pocket_rows.append([software_name, frame_folder, '', '', 0, 'NO POCKETS FOUND'])
        for pocket_rank, cluster in enumerate(clusters, start=1):
            max_score = max(per_residue_scores[r] for r in cluster)
            residue_labels = ','.join(f'{sequence_by_residue[r]}{r}' for r in sorted(cluster))
            all_pocket_rows.append([software_name, frame_folder, pocket_rank, round(max_score, 3),
                                     len(cluster), residue_labels])


# ── score every pocket against the real MD handoff matrix ──────────────────
with open(HANDOFF_PKL, 'rb') as f:
    handoff_matrix = pickle.load(f)
n_residues = handoff_matrix.shape[0]

marginal_traffic = {}
for i in range(n_residues):
    residue_number = i + 1
    marginal_traffic[residue_number] = handoff_matrix[i, :].sum() + handoff_matrix[:, i].sum() - handoff_matrix[i, i]
max_marginal = max(marginal_traffic.values())
marginal_score_normalized = {r: v / max_marginal for r, v in marginal_traffic.items()}

max_offdiagonal_handoff = 0.0
for i in range(n_residues):
    for j in range(n_residues):
        if i != j:
            max_offdiagonal_handoff = max(max_offdiagonal_handoff, handoff_matrix[i, j] + handoff_matrix[j, i])

# add my_output's own predicted path (ab42_1.txt), deduplicated to unique residue sets
raw_clusters = []
with open(AB42_PATH_TXT) as f:
    lines = f.readlines()
for line in lines[1:]:
    parts = line.split()
    if len(parts) < 2:
        continue
    cluster_id = int(parts[0])
    raw_clusters.append((16 - cluster_id, parts[1]))   # cluster 1 -> synthetic score 15 (highest)

seen_residue_sets = {}
for synthetic_score, residues_str in raw_clusters:
    residue_set = frozenset(residues_str.split(','))
    if residue_set not in seen_residue_sets or synthetic_score > seen_residue_sets[residue_set]:
        seen_residue_sets[residue_set] = synthetic_score

for residue_set, synthetic_score in seen_residue_sets.items():
    residue_number_to_one_letter = {}
    for three_letter_residue in residue_set:
        residue_number = int(''.join(ch for ch in three_letter_residue if ch.isdigit()))
        residue_number_to_one_letter[residue_number] = three_to_one[three_letter_residue[:3]]

    residue_numbers = sorted(residue_number_to_one_letter.keys())
    residue_labels = ','.join(f'{residue_number_to_one_letter[r]}{r}' for r in residue_numbers)
    # "predicted_path" is not a frame name - my_output's pockets show up in EVERY
    # frame subset below, unchanged, because they were never frame-dependent
    all_pocket_rows.append(['my_output', 'predicted_path', '', synthetic_score, len(residue_numbers), residue_labels])

# top3_pockets_comparison.csv (the single original-structure baseline) is
# deliberately NOT included anymore - only the 5-frame results + my_output

ground_truth_scores = []   # [software, structure, pocket_rank, n_residues, residues, marginal_score, coherence_score]
for software, structure, pocket_rank, native_score, n_residues_str, residues in all_pocket_rows:
    if residues in ('NO POCKETS FOUND', 'NO OUTPUT YET') or str(n_residues_str) in ('0', '1'):
        continue
    residue_numbers = sorted(int(''.join(ch for ch in label if ch.isdigit())) for label in residues.split(','))

    marginal_score = sum(marginal_score_normalized[r] for r in residue_numbers) / len(residue_numbers)

    pair_values = []
    for a in range(len(residue_numbers)):
        for b in range(a + 1, len(residue_numbers)):
            i, j = residue_numbers[a] - 1, residue_numbers[b] - 1
            pair_values.append(handoff_matrix[i, j] + handoff_matrix[j, i])
    coherence_score = (sum(pair_values) / len(pair_values)) / max_offdiagonal_handoff

    ground_truth_scores.append([software, structure, pocket_rank, len(residue_numbers), residues,
                                 round(marginal_score, 4), round(coherence_score, 4)])

with open(os.path.join(BASE_DIR, 'pocket_ground_truth_scores.csv'), 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['software', 'structure', 'pocket_rank', 'n_residues', 'residues', 'marginal_score', 'coherence_score'])
    writer.writerows(ground_truth_scores)


# ── shuttling score - does a pocket bridge residues that are far apart? ────
max_mean_distance = mean_distance_matrix.max()

shuttling_scores = []   # [software, structure, pocket_rank, residues, shuttling_score]
for software, structure, pocket_rank, n_residues, residues, marginal_score, coherence_score in ground_truth_scores:
    residue_numbers = sorted(int(''.join(ch for ch in label if ch.isdigit())) for label in residues.split(','))
    pair_shuttling_scores = []
    for a in range(len(residue_numbers)):
        for b in range(a + 1, len(residue_numbers)):
            i, j = residue_numbers[a] - 1, residue_numbers[b] - 1
            normalized_handoff = (handoff_matrix[i, j] + handoff_matrix[j, i]) / max_offdiagonal_handoff
            normalized_distance = mean_distance_matrix[i, j] / max_mean_distance
            pair_shuttling_scores.append(normalized_handoff * normalized_distance)
    pocket_shuttling_score = sum(pair_shuttling_scores) / len(pair_shuttling_scores)
    shuttling_scores.append([software, structure, pocket_rank, residues, round(pocket_shuttling_score, 4)])

with open(os.path.join(BASE_DIR, 'pocket_shuttling_scores.csv'), 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['software', 'structure', 'pocket_rank', 'residues', 'shuttling_score'])
    writer.writerows(shuttling_scores)


# ── evaluate every combination of the 5 frames, not just their pooled mean ─
# instead of one mean per software (from ALL 5 frames pooled together),
# evaluate EVERY possible combination of the 5 frames - frame_1 alone,
# frame_1+frame_2, frame_2 alone, frame_1+frame_3+frame_5, ... all
# 2^5 - 1 = 31 non-empty combinations - and compute each software's mean
# score WITHIN each combination. The final number we report per software is
# then the mean AND standard deviation of those 31 per-combination means, so
# we can see how much the result actually depends on which frames happened
# to be picked.
#
# my_output's pockets aren't tagged with a frame at all ('predicted_path'),
# so they pass the filter for every single combination unchanged - its
# per-combination mean is IDENTICAL all 31 times, giving it a standard
# deviation of exactly 0 by construction.
frame_folder_names = [frame_folder for frame_folder, frame_name in frame_info]

all_frame_combinations = []
for combination_size in range(1, len(frame_folder_names) + 1):
    for combination in itertools.combinations(frame_folder_names, combination_size):
        all_frame_combinations.append(set(combination))

print(f'Evaluating {len(all_frame_combinations)} combinations of the {len(frame_folder_names)} frames...\n')

marginal_by_software_per_combo, coherence_by_software_per_combo, shuttling_by_software_per_combo = {}, {}, {}

for frame_combination in all_frame_combinations:

    combo_marginal_by_software, combo_coherence_by_software = {}, {}
    for software, structure, pocket_rank, n_residues, residues, marginal_score, coherence_score in ground_truth_scores:
        if n_residues < 3:
            continue
        if structure != 'predicted_path' and structure not in frame_combination:
            continue
        combo_marginal_by_software.setdefault(software, []).append(marginal_score)
        combo_coherence_by_software.setdefault(software, []).append(coherence_score)

    for software in combo_marginal_by_software:
        combo_mean_marginal = sum(combo_marginal_by_software[software]) / len(combo_marginal_by_software[software])
        combo_mean_coherence = sum(combo_coherence_by_software[software]) / len(combo_coherence_by_software[software])
        marginal_by_software_per_combo.setdefault(software, []).append(combo_mean_marginal)
        coherence_by_software_per_combo.setdefault(software, []).append(combo_mean_coherence)

    combo_shuttling_by_software = {}
    for software, structure, pocket_rank, residues, shuttling_score in shuttling_scores:
        if len(residues.split(',')) < 3:
            continue
        if structure != 'predicted_path' and structure not in frame_combination:
            continue
        combo_shuttling_by_software.setdefault(software, []).append(shuttling_score)

    for software in combo_shuttling_by_software:
        combo_mean_shuttling = sum(combo_shuttling_by_software[software]) / len(combo_shuttling_by_software[software])
        shuttling_by_software_per_combo.setdefault(software, []).append(combo_mean_shuttling)

marginal_mean, marginal_std = {}, {}
coherence_mean, coherence_std = {}, {}
shuttling_mean, shuttling_std = {}, {}

all_software_names = sorted(
    marginal_by_software_per_combo.keys(),
    key=lambda sw: -sum(marginal_by_software_per_combo[sw]) / len(marginal_by_software_per_combo[sw]),
)

for software in all_software_names:
    values = marginal_by_software_per_combo[software]
    marginal_mean[software] = sum(values) / len(values)
    marginal_std[software] = statistics.pstdev(values) if len(values) > 1 else 0.0

    values = coherence_by_software_per_combo[software]
    coherence_mean[software] = sum(values) / len(values)
    coherence_std[software] = statistics.pstdev(values) if len(values) > 1 else 0.0

    if software in shuttling_by_software_per_combo:
        values = shuttling_by_software_per_combo[software]
        shuttling_mean[software] = sum(values) / len(values)
        shuttling_std[software] = statistics.pstdev(values) if len(values) > 1 else 0.0
    else:
        shuttling_mean[software] = float('nan')
        shuttling_std[software] = 0.0

print(f"{'software':<15}{'n_combos':<10}{'marginal (mean +/- std)':<28}{'coherence (mean +/- std)':<28}{'shuttling (mean +/- std)':<28}")
for software in all_software_names:
    n_combos = len(marginal_by_software_per_combo[software])
    marginal_text = f'{marginal_mean[software]:.4f} +/- {marginal_std[software]:.4f}'
    coherence_text = f'{coherence_mean[software]:.4f} +/- {coherence_std[software]:.4f}'
    shuttling_text = f'{shuttling_mean[software]:.4f} +/- {shuttling_std[software]:.4f}'
    print(f'{software:<15}{n_combos:<10}{marginal_text:<28}{coherence_text:<28}{shuttling_text:<28}')


# ── plot: mean +/- std (across all 31 frame combinations) per software,
# styled with the exact same publication theme as every other figure in
# ../paper_final_figures_code ───────────────────────────────────────────────
ps.set_publication_theme(font_scale=1.05)

# display name for the x-axis only -- 'my_output' stays the actual key used
# everywhere above (dict lookups, all_software_names, etc.); edit this if the
# method's public name changes again
SOFTWARE_DISPLAY_NAMES = {
    'my_output': 'Our Method',
}

# save everything needed to redraw this figure later on any machine with
# just matplotlib + this one file - no trajectory, no handoff matrix, no
# tool output folders required
plot_data = {
    'all_software_names': all_software_names,
    'marginal_mean': marginal_mean,
    'marginal_std': marginal_std,
    'coherence_mean': coherence_mean,
    'coherence_std': coherence_std,
    'shuttling_mean': shuttling_mean,
    'shuttling_std': shuttling_std,
}
with open(os.path.join(BASE_DIR, 'plot_data.pkl'), 'wb') as f:
    pickle.dump(plot_data, f)
print('Saved plot_data.pkl - everything needed to redraw the figure elsewhere\n')

bar_width = 0.26
x_positions = np.arange(len(all_software_names))

fig_b, ax_b = plt.subplots(figsize=(10, 5.2))
ax_b.set_facecolor(SURFACE)

marginal_values = [marginal_mean[sw] for sw in all_software_names]
marginal_errors = [marginal_std[sw] for sw in all_software_names]
coherence_values = [coherence_mean[sw] for sw in all_software_names]
coherence_errors = [coherence_std[sw] for sw in all_software_names]
shuttling_values = [shuttling_mean[sw] for sw in all_software_names]
shuttling_errors = [shuttling_std[sw] for sw in all_software_names]

error_bar_style = {'ecolor': INK_SECONDARY, 'elinewidth': 1.0, 'capsize': 2.5, 'capthick': 1.0}

bars_marginal = ax_b.bar(x_positions - bar_width, marginal_values, width=bar_width, color=PRIMARY,
                          edgecolor=SURFACE, linewidth=0.6, yerr=marginal_errors, error_kw=error_bar_style,
                          label='Marginal', zorder=3)
bars_coherence = ax_b.bar(x_positions, coherence_values, width=bar_width, color=SECONDARY,
                           edgecolor=SURFACE, linewidth=0.6, yerr=coherence_errors, error_kw=error_bar_style,
                           label='Coherence', zorder=3)
bars_shuttling = ax_b.bar(x_positions + bar_width, shuttling_values, width=bar_width, color=TERTIARY,
                           edgecolor=SURFACE, linewidth=0.6, yerr=shuttling_errors, error_kw=error_bar_style,
                           label='Shuttling', zorder=3)

for bar_group, error_values in ((bars_marginal, marginal_errors), (bars_coherence, coherence_errors), (bars_shuttling, shuttling_errors)):
    for bar, error_value in zip(bar_group, error_values):
        height = bar.get_height()
        ax_b.text(bar.get_x() + bar.get_width() / 2, height + error_value + 0.014, f'{height:.2f}',
                   ha='center', va='bottom', color=INK_SECONDARY, fontsize=10)

ax_b.set_xticks(x_positions)
x_tick_labels = [SOFTWARE_DISPLAY_NAMES.get(sw, sw) for sw in all_software_names]
ax_b.set_xticklabels(x_tick_labels, color=INK_SECONDARY, fontsize=13)
ax_b.set_ylabel('Score', color=INK_SECONDARY, fontsize=13)
ax_b.set_ylim(0, max(v + e for v, e in zip(marginal_values + coherence_values + shuttling_values,
                                            marginal_errors + coherence_errors + shuttling_errors)) * 1.22)
ax_b.tick_params(axis='x', length=0)
ax_b.tick_params(axis='y', colors=INK_SECONDARY, labelsize=13)
sns.despine(fig=fig_b)
ax_b.spines['bottom'].set_color(BASELINE_GRAY)
ax_b.spines['left'].set_color(BASELINE_GRAY)
ax_b.set_title('Ground-truth Scores by Software ',color=INK_PRIMARY, fontsize=15)
ax_b.legend(loc='upper right', frameon=False, fontsize=13, labelcolor=INK_SECONDARY)

fig_b.tight_layout()

out_path_b = os.path.join(BASE_DIR, 'pocket_metrics_comparison_with_std.pdf')
fig_b.savefig(out_path_b, bbox_inches='tight', facecolor=SURFACE)
fig_b.savefig(out_path_b.replace('.pdf', '.png'), dpi=300, bbox_inches='tight', facecolor=SURFACE)
plt.show()
