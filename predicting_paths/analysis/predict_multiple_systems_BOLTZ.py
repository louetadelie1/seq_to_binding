from transformers import AutoModel, AutoTokenizer
import torch
import glob
import pickle
import linecache
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plot_style as ps
from itertools import combinations
import random
from collections import OrderedDict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

K_NN         = 50
BATCH_SIZE   = 4096
TOP_ACTIVES  = 20
EF_FRACTIONS = [0.01, 0.05, 0.10]
N_CHANCE_SHUFFLES = 100   # random re-orderings of the real triplet pools used for the chance baseline
WEIGHTS      = '/ptmp/adlouet/camb/sequence_to_binding_paths/fine_tuning_experiments/exp_5_boltz_distances_inf_pred/weights/best_model_change_ranking.pt'
idps_base    = '//ptmp/adlouet/camb/sequence_to_binding_paths/post_processed_idps'

PROTEINS = ['abeta', 'htt', 'p53_c', 'p53_n', 'tau', 'tdp_43', 'alpha_CGRP', 'PACAP_27','fus','prion']

PROTEIN_NAMES = {
    'abeta':      'Amyloid-β (Aβ)',
    'htt':        'Huntingtin (HTT)',
    'p53_c':      'p53 C-terminal domain',
    'p53_n':      'p53 N-terminal domain',
    'tau':        'Tau (MAPT)',
    'tdp_43':     'TDP-43',
    'alpha_CGRP': 'α-CGRP',
    'PACAP_27':   'PACAP-27',
    'fus':        'FUS',
    'prion':      'Prion protein (PrP)',
}

random.seed(42)   # reproducible ZINC-pair sampling across runs
rng = np.random.default_rng(42)   # reproducible chance-baseline shuffles

tokeniser  = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
esm_model  = AutoModel.from_pretrained("facebook/esm2_t12_35M_UR50D").to(device)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.leaky_relu = torch.nn.LeakyReLU()
        self.dropout    = torch.nn.Dropout(0.5)
        self.layer1     = torch.nn.Linear(480 * 3 + 10, 512)
        self.ln1        = torch.nn.LayerNorm([512])
        self.layer2     = torch.nn.Linear(512, 128)
        self.ln2        = torch.nn.LayerNorm([128])
        self.layer3     = torch.nn.Linear(128, 32)
        self.ln3        = torch.nn.LayerNorm([32])
        self.regression_head = torch.nn.Linear(32, 1)

    def forward(self, x1, x2, x3, pos):
        x = torch.cat([x1, x2, x3, pos], dim=-1)
        x = self.dropout(self.leaky_relu(self.ln1(self.layer1(x))))
        x = self.dropout(self.leaky_relu(self.ln2(self.layer2(x))))
        x = self.dropout(self.leaky_relu(self.ln3(self.layer3(x))))
        return self.regression_head(x)


def clean(input):
    return {tuple(int(i) for i in key): float(value[0]) for key, value in input.items()}


def shuffled_chance_curves(pool, actives, top_actives, x_grid, ef_fractions, n_shuffles, rng):
    N = len(pool)
    actives_set = set(actives)
    is_active = np.fromiter((1.0 if t in actives_set else 0.0 for t in pool), dtype=np.float64, count=N)
    scan_ext = np.concatenate(([0.0], np.arange(1, N + 1) / N))
    ef_fractions_arr = np.asarray(ef_fractions)
    curves = np.empty((n_shuffles, len(x_grid)))
    efs    = np.empty((n_shuffles, len(ef_fractions)))
    for s in range(n_shuffles):
        shuffled  = rng.permutation(is_active)
        find_ext  = np.concatenate(([0.0], np.cumsum(shuffled) / top_actives))
        curves[s] = np.interp(x_grid, scan_ext, find_ext)
        efs[s]    = np.interp(ef_fractions_arr, scan_ext, find_ext) / ef_fractions_arr
    return curves, efs


model = Model().to(device)
model.load_state_dict(torch.load(WEIGHTS, map_location=device))
model.eval()

os.makedirs('figures', exist_ok=True)

X_GRID = np.linspace(0, 1, 200)   # shared x-axis so the two replicate directions can be compared

auc_data = []   # (protein, pred_A, pred_B, base_AB, base_BA) curves interpolated onto X_GRID
ef_data  = []   # (protein, ef_pred_A, ef_pred_B, ef_base_AB, ef_base_BA) arrays over EF_FRACTIONS

for protein in PROTEINS:
    zinc_pkls = sorted(glob.glob(f'{idps_base}/{protein}/*/pickled_files/p_eq_keys.pckl'))
    if len(zinc_pkls) < 2:
        print(f"skipping {protein}: fewer than 2 processed ZINC compounds")
        continue

    ref_dir = sorted(glob.glob(f'{idps_base}/{protein}/*'))[0]
    fasta_path = f'{ref_dir}/sequence.fasta'
    boltz_dist_path = f'{ref_dir}/ca_dist_matrix_boltz.pkl'
    if not os.path.exists(fasta_path):
        print(f"skipping {protein}: no sequence.fasta yet")
        continue
    if not os.path.exists(boltz_dist_path):
        print(f"skipping {protein}: no ca_dist_matrix_boltz.pkl yet "
              f"(run generate_boltz_distances.py first)")
        continue

    seq = linecache.getline(fasta_path, 2).strip()
    L = len(seq)

    # ── ESM embeddings ────────────────────────────────────────────────────
    inputs = tokeniser(seq, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        embeddings = esm_model(**inputs).last_hidden_state.squeeze(0).detach().cpu()

    # ── distance matrix (Boltz diffusion-sample mean, nm) + kNN candidate
    # triplets ────────────────────────────────────────────────────────────
    _loaded = pickle.load(open(boltz_dist_path, 'rb'))
    ca_dist_matrix = _loaded['matrix'] if isinstance(_loaded, dict) and 'matrix' in _loaded else _loaded
    n_res = ca_dist_matrix.shape[0]
    if n_res != L:
        print(f"skipping {protein}: ca_dist_matrix_boltz shape {ca_dist_matrix.shape} != seq length {L}")
        continue

    candidate_triplets = set()
    for i in range(n_res):
        neighbors = np.argsort(ca_dist_matrix[i])[1:K_NN + 1]
        for j, l in combinations(neighbors, 2):
            candidate_triplets.add(tuple(sorted((i, int(j), int(l)))))
    candidate_triplets = list(candidate_triplets)

    all_x1, all_x2, all_x3, all_pos, valid_triplets = [], [], [], [], []
    for trio in candidate_triplets:
        ri, rj, rk = trio
        if any(r >= L for r in (ri, rj, rk)):
            continue
        dist_ij, dist_jk, dist_ik = (float(ca_dist_matrix[ri, rj]), float(ca_dist_matrix[rj, rk]), float(ca_dist_matrix[ri, rk]))
        min_dist, max_dist = min(dist_ij, dist_jk, dist_ik), max(dist_ij, dist_jk, dist_ik)
        mean_dist   = (dist_ij + dist_jk + dist_ik) / 3
        compactness = min_dist / max_dist if max_dist > 0 else 0.0
        all_x1.append(embeddings[ri + 1])
        all_x2.append(embeddings[rj + 1])
        all_x3.append(embeddings[rk + 1])
        all_pos.append(torch.tensor([ri/L, rj/L, rk/L, dist_ij, dist_jk, dist_ik,
                                      min_dist, max_dist, mean_dist, compactness], dtype=torch.float32))
        valid_triplets.append(trio)

    # ── batched inference ──────────────────────────────────────────────────
    preds = []
    with torch.no_grad():
        for start in range(0, len(all_x1), BATCH_SIZE):
            x1_b  = torch.stack(all_x1[start:start + BATCH_SIZE]).to(device)
            x2_b  = torch.stack(all_x2[start:start + BATCH_SIZE]).to(device)
            x3_b  = torch.stack(all_x3[start:start + BATCH_SIZE]).to(device)
            pos_b = torch.stack(all_pos[start:start + BATCH_SIZE]).to(device)
            preds.extend(model(x1_b, x2_b, x3_b, pos_b).squeeze(-1).cpu().tolist())

    ranked_triplets = sorted(zip(valid_triplets, preds), key=lambda t: t[1], reverse=True)
    pred_triplets, _ = zip(*ranked_triplets)
    pred_dict = dict(zip(valid_triplets, preds))
    N_pred = len(preds)

    # ── pick 2 ZINC compounds, load their experimental p_eq rankings ────────
    path_A, path_B = random.sample(zinc_pkls, 2)
    zinc_A = os.path.basename(os.path.dirname(os.path.dirname(path_A)))
    zinc_B = os.path.basename(os.path.dirname(os.path.dirname(path_B)))

    dict_A = OrderedDict(sorted(clean(pickle.load(open(path_A, 'rb'))).items(), key=lambda item: item[1], reverse=True))
    triplets_A, _ = map(list, zip(*dict_A.items()))
    dict_B = OrderedDict(sorted(clean(pickle.load(open(path_B, 'rb'))).items(), key=lambda item: item[1], reverse=True))
    triplets_B, _ = map(list, zip(*dict_B.items()))

    actives_A = triplets_A[:TOP_ACTIVES]
    actives_B = triplets_B[:TOP_ACTIVES]

    # ── enrichment curves: predicted->A, predicted->B, A->B, B->A ───────────
    hits = 0; frac_scan_A, frac_find_A = [0.0], [0.0]
    for idx, triplet in enumerate(pred_triplets):
        if triplet in actives_A: hits += 1
        frac_scan_A.append((idx + 1) / N_pred); frac_find_A.append(hits / TOP_ACTIVES)

    hits = 0; frac_scan_B, frac_find_B = [0.0], [0.0]
    for idx, triplet in enumerate(pred_triplets):
        if triplet in actives_B: hits += 1
        frac_scan_B.append((idx + 1) / N_pred); frac_find_B.append(hits / TOP_ACTIVES)

    hits = 0; frac_scan_A_B, frac_find_A_B = [0.0], [0.0]
    for idx, triplet in enumerate(triplets_A):
        if triplet in actives_B: hits += 1
        frac_scan_A_B.append((idx + 1) / len(triplets_A)); frac_find_A_B.append(hits / TOP_ACTIVES)

    hits = 0; frac_scan_B_A, frac_find_B_A = [0.0], [0.0]
    for idx, triplet in enumerate(triplets_B):
        if triplet in actives_A: hits += 1
        frac_scan_B_A.append((idx + 1) / len(triplets_B)); frac_find_B_A.append(hits / TOP_ACTIVES)

    # keep both replicate directions (not just their average) so each
    # per-protein panel can show a mean +/- std band from the two directions
    pred_A_curve  = np.interp(X_GRID, frac_scan_A, frac_find_A)
    pred_B_curve  = np.interp(X_GRID, frac_scan_B, frac_find_B)
    base_AB_curve = np.interp(X_GRID, frac_scan_A_B, frac_find_A_B)
    base_BA_curve = np.interp(X_GRID, frac_scan_B_A, frac_find_B_A)

    ef_pred_A = np.array([np.interp(f, frac_scan_A, frac_find_A) / f for f in EF_FRACTIONS])
    ef_pred_B = np.array([np.interp(f, frac_scan_B, frac_find_B) / f for f in EF_FRACTIONS])
    ef_base_AB = np.array([np.interp(f, frac_scan_A_B, frac_find_A_B) / f for f in EF_FRACTIONS])
    ef_base_BA = np.array([np.interp(f, frac_scan_B_A, frac_find_B_A) / f for f in EF_FRACTIONS])

    # ── chance baseline: same triplet pools, random order instead of predicted /
    # p_eq-ranked order, repeated N_CHANCE_SHUFFLES times and pooled across all
    # four (pool, actives) pairs used above ───────────────────────────────────
    chance_curves, chance_efs = [], []
    for pool, actives in [(pred_triplets, actives_A), (pred_triplets, actives_B),
                           (triplets_A, actives_B), (triplets_B, actives_A)]:
        c, e = shuffled_chance_curves(pool, actives, TOP_ACTIVES, X_GRID, EF_FRACTIONS,
                                       N_CHANCE_SHUFFLES, rng)
        chance_curves.append(c)
        chance_efs.append(e)
    chance_curves = np.concatenate(chance_curves, axis=0)
    chance_efs    = np.concatenate(chance_efs, axis=0)
    chance_curve_mean, chance_curve_std = chance_curves.mean(0), chance_curves.std(0)
    chance_ef_mean, chance_ef_std       = chance_efs.mean(0), chance_efs.std(0)

    auc_data.append((protein, pred_A_curve, pred_B_curve, base_AB_curve, base_BA_curve,
                      chance_curve_mean, chance_curve_std))
    ef_data.append((protein, ef_pred_A, ef_pred_B, ef_base_AB, ef_base_BA,
                     chance_ef_mean, chance_ef_std))

    print(f"{protein}: {n_res} residues, {len(valid_triplets):,} triplets, "
          f"vs {zinc_A} & {zinc_B} -- done")


# ══════════════════════════ 2 summary figures, one panel per protein ══════════════════════════
ps.set_publication_theme(font_scale=1.3)

# green/violet pair (predicted vs. experiment<->experiment reference) + neutral
# for chance. Uses PAIRED_LIGHT/PAIRED_DARK rather than PRIMARY/SECONDARY so the
# two lines stay on the dedicated "exactly two series" pair -- see plot_style.py.
# Their mean+/-std bands are drawn in BAND_COLOR (light hydro blue), not in
# these two hues, so every uncertainty band in the paper reads the same way
# regardless of which brand hue the line above it uses.
PRED, EXP, MUTED = ps.PAIRED_DARK, ps.PAIRED_LIGHT, ps.MUTED
N_SYS = len(auc_data)
ncols = 5
nrows = int(np.ceil(N_SYS / ncols))

# PLOT 1 -- AUC: one panel per protein. Each panel shows both curves as a mean +/-1 std
# band over their two directions (n=2): experiment<->experiment is A->B / B->A, and
# predicted->experiment is predicted->actives_A / predicted->actives_B (i.e. the same
# ranked list scored against each of the two experimental active sets).
fig1, axes1 = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.6 * nrows),
                            sharex=True, sharey=True, squeeze=False)
axes1 = axes1.flatten()
for i, (ax, (protein, pred_A_curve, pred_B_curve, base_AB_curve, base_BA_curve,
             chance_curve_mean, chance_curve_std)) in enumerate(zip(axes1, auc_data)):
    base_stack = np.stack([base_AB_curve, base_BA_curve])
    base_mean, base_std = base_stack.mean(0), base_stack.std(0)
    pred_stack = np.stack([pred_A_curve, pred_B_curve])
    pred_mean, pred_std = pred_stack.mean(0), pred_stack.std(0)

    ax.fill_between(X_GRID, np.clip(chance_curve_mean - chance_curve_std, 0, 1),
                     np.clip(chance_curve_mean + chance_curve_std, 0, 1),
                     color=MUTED, alpha=0.35, linewidth=0, zorder=1)
    ax.plot(X_GRID, chance_curve_mean, color=MUTED, lw=1.2, ls='--', zorder=1, label='Chance (shuffled triplets)')
    ax.fill_between(X_GRID, np.clip(base_mean - base_std, 0, 1), np.clip(base_mean + base_std, 0, 1),
                     color=ps.BAND_COLOR, alpha=0.35, linewidth=0, zorder=2)
    ax.fill_between(X_GRID, np.clip(pred_mean - pred_std, 0, 1), np.clip(pred_mean + pred_std, 0, 1),
                     color=ps.BAND_COLOR, alpha=0.35, linewidth=0, zorder=2)
    ax.plot(X_GRID, base_mean, color=EXP, lw=2.0, zorder=3, label='Experiment ↔ experiment')
    ax.plot(X_GRID, pred_mean, color=PRED, lw=2.0, zorder=4, label='Predicted → experiment')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
    ax.tick_params(labelsize=13)
    ax.set_title(PROTEIN_NAMES.get(protein, protein), fontsize=15)
for ax in axes1[N_SYS:]:
    ax.axis('off')
# one shared x/y label for the whole grid instead of repeating per panel
# (fig.supxlabel/supylabel need matplotlib>=3.4; this mpl is 3.1, so fig.text)
fig1.text(0.5, 0.02, 'Fraction of ranked list screened', ha='center', va='center', fontsize=18)
fig1.text(0.01, 0.5, f'Fraction of top-{TOP_ACTIVES} actives recovered', ha='center', va='center',
          rotation='vertical', fontsize=18)
handles, labels = axes1[0].get_legend_handles_labels()
fig1.legend(handles, labels, loc='upper center', ncol=4, frameon=False, fontsize=16, bbox_to_anchor=(0.5, 1.04))
sns.despine(fig=fig1)
fig1.tight_layout(rect=[0.035, 0.05, 1, 0.90])
fig1.subplots_adjust(wspace=0.12, hspace=0.35)
fig1.savefig('figures/summary_auc.png', dpi=300, bbox_inches='tight')

# PLOT 2 -- EF: one panel per protein. Each panel is a grouped bar chart over
# EF_FRACTIONS, with error bars from each side's two directions (n=2): predicted
# ->actives_A / predicted->actives_B for the predicted bars, A->B / B->A for the
# experiment<->experiment bars.
fig2, axes2 = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.4 * nrows),
                            sharex=True, squeeze=False)
axes2 = axes2.flatten()
x_pos = np.arange(len(EF_FRACTIONS)); w = 0.27
for i, (ax, (protein, ef_pred_A, ef_pred_B, ef_base_AB, ef_base_BA,
             chance_ef_mean, chance_ef_std)) in enumerate(zip(axes2, ef_data)):
    ef_pred_mean = np.stack([ef_pred_A, ef_pred_B]).mean(0)
    ef_pred_std  = np.stack([ef_pred_A, ef_pred_B]).std(0)
    ef_base_mean = np.stack([ef_base_AB, ef_base_BA]).mean(0)
    ef_base_std  = np.stack([ef_base_AB, ef_base_BA]).std(0)

    ax.bar(x_pos - w, ef_pred_mean, width=w, color=PRED, edgecolor='white', linewidth=0.6,
           yerr=ef_pred_std, error_kw={'ecolor': MUTED, 'elinewidth': 1.1, 'capthick': 1.1},
           capsize=3, label='Predicted → experiment')
    ax.bar(x_pos, ef_base_mean, width=w, color=EXP, edgecolor='white', linewidth=0.6,
           yerr=ef_base_std, error_kw={'ecolor': MUTED, 'elinewidth': 1.1, 'capthick': 1.1},
           capsize=3, label='Experiment ↔ experiment')
    # chance/random moved to the rightmost slot in each triplet -- it's the
    # reference floor, not a third result on equal footing with the other two
    ax.bar(x_pos + w, chance_ef_mean, width=w, color=MUTED, edgecolor='white', linewidth=0.6,
           yerr=chance_ef_std, error_kw={'ecolor': MUTED, 'elinewidth': 1.1, 'capthick': 1.1},
           capsize=3, label='Chance (shuffled triplets)')
    ax.set_xticks(x_pos); ax.set_xticklabels([f'{f*100:.0f}%' for f in EF_FRACTIONS])
    ax.tick_params(labelsize=13)
    ax.set_title(PROTEIN_NAMES.get(protein, protein), fontsize=15)
for ax in axes2[N_SYS:]:
    ax.axis('off')
# one shared x/y label for the whole grid instead of repeating per panel
fig2.text(0.5, 0.02, 'Top fraction of ranked list screened', ha='center', va='center', fontsize=18)
fig2.text(0.01, 0.5, 'Enrichment factor', ha='center', va='center', rotation='vertical', fontsize=18)
handles, labels = axes2[0].get_legend_handles_labels()
fig2.legend(handles, labels, loc='upper center', ncol=3, frameon=False, fontsize=16, bbox_to_anchor=(0.5, 1.04))
sns.despine(fig=fig2)
fig2.tight_layout(rect=[0.035, 0.05, 1, 0.90])
fig2.subplots_adjust(wspace=0.3, hspace=0.35)
fig2.savefig('figures/summary_ef.png', dpi=300, bbox_inches='tight')

plt.show()
