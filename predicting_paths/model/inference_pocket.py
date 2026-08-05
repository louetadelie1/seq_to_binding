"""
Predict on all kNN-filtered residue triplets for each protein.
Plots binding profiles: one predicted profile (red) + one actual profile per ZINC.
"""

from transformers import AutoModel, AutoTokenizer
import torch
import glob
import pickle
import linecache
import numpy as np
from collections import defaultdict
# from scipy.stats import spearmanr
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import plot_style as ps
from itertools import combinations
import re
from itertools import combinations
from scipy.stats import spearmanr
import random
from collections import OrderedDict
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS, TSNE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# TOP_ACTIVES = 10
K_NN       = 40 #50
BATCH_SIZE = 4096
# N_SHUFFLES = 100
WEIGHTS      = '/ptmp/adlouet/camb/sequence_to_binding_paths/fine_tuning_experiments/exp_5_boltz_distances_inf_pred/weights/best_model_change_ranking.pt'
# BASE       = '//ptmp/adlouet/camb/sequence_to_binding_paths/post_processed_idps'
tokeniser  = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
esm_model  = AutoModel.from_pretrained("facebook/esm2_t12_35M_UR50D").to(device)


# base = '//ptmp/adlouet/camb/sequence_to_binding_paths/post_processed_probes'
base='//ptmp/adlouet/camb/sequence_to_binding_paths/post_processed_idps/abeta/ZINC000000030986'

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


def make_binding_profile(triplets, scores, n_res):
    """Aggregate triplet scores to a per-residue profile normalised to [0, 1].

    Each triplet contributes its normalised score to the three residues it spans.
    """
    profile = np.zeros(n_res)
    s = np.array(scores, dtype=float)
    s = (s - s.min()) / (s.max() - s.min() + 1e-12)
    for (ri, rj, rk), w in zip(triplets, s):
        profile[ri] += w
        profile[rj] += w
        profile[rk] += w
    mn, mx = profile.min(), profile.max()
    if mx > mn:
        profile = (profile - mn) / (mx - mn)
    return profile


model = Model().to(device)
model.load_state_dict(torch.load(WEIGHTS, map_location=device))
model.eval()
print("Loaded weights from exp_18")

os.makedirs('figures', exist_ok=True)


cmap = plt.cm.tab10


# ── ESM embeddings (once per protein) ────────────────────────────────────
seq = 'DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA'
ca_dist_file = f'{base}/ca_dist_matrix_boltz.pkl'
prot_name = 'ab'

L   = len(seq)
inputs = tokeniser(seq, return_tensors="pt", truncation=True, max_length=1024)
inputs = {k: v.to(device) for k, v in inputs.items()}
with torch.no_grad():
    embeddings = esm_model(**inputs).last_hidden_state.squeeze(0).detach().cpu()
n_emb = embeddings.shape[0]

# ── distance matrix ───────────────────────────────────────────────────────
# ca_dist_file = ref_protein.removesuffix('pickled_files/p_eq_keys.pckl') + 'ca_dist_matrix.pkl'


_loaded = pickle.load(open(ca_dist_file, 'rb'))
ca_dist_matrix = _loaded['matrix'] if isinstance(_loaded, dict) and 'matrix' in _loaded else _loaded
_labels = _loaded.get('labels', None) if isinstance(_loaded, dict) else None

# ── kNN candidate triplets ────────────────────────────────────────────────
n_res = ca_dist_matrix.shape[0]
candidate_triplets = set()
for i in range(n_res):
    neighbors = np.argsort(ca_dist_matrix[i])[1:K_NN + 1]
    for j, l in combinations(neighbors, 2):
        candidate_triplets.add(tuple(sorted((i, int(j), int(l)))))
candidate_triplets = list(candidate_triplets)
print(f"{prot_name}: {n_res} residues, {len(candidate_triplets):,} kNN candidates (k={K_NN})")

# ── build input tensors ───────────────────────────────────────────────────
all_x1, all_x2, all_x3, all_pos, valid_triplets = [], [], [], [], []
for trio in candidate_triplets:
    ri, rj, rk = trio
    if any(r >= L for r in (ri, rj, rk)):
        continue
    dist_ij     = float(ca_dist_matrix[ri, rj])
    dist_jk     = float(ca_dist_matrix[rj, rk])
    dist_ik     = float(ca_dist_matrix[ri, rk])
    min_dist    = min(dist_ij, dist_jk, dist_ik)
    max_dist    = max(dist_ij, dist_jk, dist_ik)
    mean_dist   = (dist_ij + dist_jk + dist_ik) / 3
    compactness = min_dist / max_dist if max_dist > 0 else 0.0
    # embeddings[0] is the <cls> token, so residue i's embedding is at index i+1
    all_x1.append(embeddings[ri + 1])
    all_x2.append(embeddings[rj + 1])
    all_x3.append(embeddings[rk + 1])
    all_pos.append(torch.tensor([
        ri/L, rj/L, rk/L,
        dist_ij, dist_jk, dist_ik,
        min_dist, max_dist, mean_dist, compactness
    ], dtype=torch.float32))
    valid_triplets.append(trio)

# ── batched inference ─────────────────────────────────────────────────────
preds = []
with torch.no_grad():
    for start in range(0, len(all_x1), BATCH_SIZE):
        x1_b  = torch.stack(all_x1[start:start + BATCH_SIZE]).to(device)
        x2_b  = torch.stack(all_x2[start:start + BATCH_SIZE]).to(device)
        x3_b  = torch.stack(all_x3[start:start + BATCH_SIZE]).to(device)
        pos_b = torch.stack(all_pos[start:start + BATCH_SIZE]).to(device)
        preds.extend(model(x1_b, x2_b, x3_b, pos_b).squeeze(-1).cpu().tolist())

ranked_triplets = sorted(zip(valid_triplets, preds), key=lambda t: t[1], reverse=True)

pred_profile = make_binding_profile(valid_triplets, np.array(preds), n_res)
print(f"\n{prot_name}: top predicted peq triplets (residue indices, predicted p_eq):")
for rank, (trio, score) in enumerate(ranked_triplets[:50], start=1):
    print(f"{rank:>4}  {trio}  {score:.4f}")

sorted_scores = np.array([s for _, s in ranked_triplets])
ranks = np.arange(len(sorted_scores))

x_norm = ranks / (len(ranks) - 1)
y_norm = (sorted_scores - sorted_scores.min()) / (np.ptp(sorted_scores) + 1e-12)
x0, y0, x1, y1 = x_norm[0], y_norm[0], x_norm[-1], y_norm[-1]
dx, dy = x1 - x0, y1 - y0
chord_len = np.hypot(dx, dy)
dist_to_chord = np.abs(dy * (x_norm - x0) - dx * (y_norm - y0)) / chord_len
knee_idx = int(np.argmax(dist_to_chord))
print(f"\nKnee (max chord distance) at rank {knee_idx} / {len(sorted_scores):,} "
      f"(top {knee_idx / len(sorted_scores):.2%}), score={sorted_scores[knee_idx]:.4f}")

zero_cross = int(np.searchsorted(-sorted_scores, 0))  # sorted_scores is descending

ps.set_publication_theme(font_scale=1.3)   # applied here (not just before the later plots) so THIS figure, the first one saved, also gets the publication styling instead of raw matplotlib defaults

ZOOM_N = 1000
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, n in zip(axes, [len(sorted_scores), ZOOM_N]):
    ax.plot(ranks[:n], sorted_scores[:n], color=ps.PAIRED_LIGHT, lw=1.6)
    if knee_idx < n:
        ax.axvline(knee_idx, color=ps.PAIRED_DARK, ls='--', lw=1.2, label=f'knee (rank {knee_idx})')
    ax.tick_params(axis='both', labelsize=13)
    ax.legend(frameon=False, fontsize=14)
axes[0].set_title(f'Full curve (n={len(sorted_scores):,})', fontsize=15)
axes[1].set_title(f'Zoomed to top {ZOOM_N:,} ranks', fontsize=15)
# one shared x/y label for both panels instead of repeating the same text on each
fig.text(0.5, 0.02, 'Rank (descending predicted score)', ha='center', va='center', fontsize=17)
fig.text(0.01, 0.5, 'Predicted p_eq', ha='center', va='center', rotation='vertical', fontsize=17)
sns.despine(fig=fig)
fig.tight_layout(rect=[0.03, 0.05, 1, 1])
fig.savefig('figures/rank_score_curve.png', dpi=200)
plt.show()


# resname lookup built straight from `seq` (1-indexed residue numbering) --
# no need to re-parse the PDB via MDAnalysis just to get 3-letter codes.
ONE_TO_THREE = {
    'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
    'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
    'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
    'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL',
}
resid_map = {i: (i + 1, ONE_TO_THREE[aa]) for i, aa in enumerate(seq)}

def fmt_triplet(t):
    return tuple(f"{resid_map[i][1]}{resid_map[i][0]}" for i in t)


# ── Cluster the top-scoring triplets by hydrophobicity, H-bonding, and spatial proximity ──
#
# Two triplets are made "close" in the combined distance either because they share
# residues / sit near each other in space (spatial term, from ca_dist_matrix) or
# because their three residues have similar side-chain chemistry (chemical term).
# Overlapping triplets like (1,2,3) and (2,3,4) fall out of this automatically: their
# spatial term is ~0 since two of their three residues coincide.


KYTE_DOOLITTLE = {
    'ILE': 4.5, 'VAL': 4.2, 'LEU': 3.8, 'PHE': 2.8, 'CYS': 2.5, 'MET': 1.9,
    'ALA': 1.8, 'GLY': -0.4, 'THR': -0.7, 'SER': -0.8, 'TRP': -0.9,
    'TYR': -1.3, 'PRO': -1.6, 'HIS': -3.2, 'GLU': -3.5, 'GLN': -3.5,
    'ASP': -3.5, 'ASN': -3.5, 'LYS': -3.9, 'ARG': -4.5,
}
HBOND_DONORS    = {'ARG', 'LYS', 'TRP', 'ASN', 'GLN', 'HIS', 'SER', 'THR', 'TYR'}
HBOND_ACCEPTORS = {'ASP', 'GLU', 'ASN', 'GLN', 'HIS', 'SER', 'THR', 'TYR'}
CHARGE          = {'ARG': 1.0, 'LYS': 1.0, 'HIS': 0.5, 'ASP': -1.0, 'GLU': -1.0}

# keep everything the model predicts as "more likely to bind than not"
# (the zero-crossing of the rank-ordered score curve, see diagnostic above)
# rather than an arbitrary top-K/top-frac cutoff
n_top = zero_cross
top_triplets = [t for t, _ in ranked_triplets[:n_top]]
top_scores   = np.array([s for _, s in ranked_triplets[:n_top]])
print(f"\nClustering top {n_top} triplets (score > 0, {n_top / len(ranked_triplets):.2%} of {len(ranked_triplets):,} scored)")

def triplet_chem_features(t):
    names = [resid_map[i][1] for i in t]
    hydrophobicity = np.mean([KYTE_DOOLITTLE.get(n, 0.0) for n in names])
    n_donors       = sum(n in HBOND_DONORS for n in names)
    n_acceptors    = sum(n in HBOND_ACCEPTORS for n in names)
    net_charge     = sum(CHARGE.get(n, 0.0) for n in names)
    return np.array([hydrophobicity, n_donors, n_acceptors, net_charge])

chem_raw  = np.stack([triplet_chem_features(t) for t in top_triplets])
chem_z    = (chem_raw - chem_raw.mean(0)) / (chem_raw.std(0) + 1e-12)
chem_dist = squareform(pdist(chem_z, metric='euclidean')) # two triplets are close if they have similar physcial features

# spatial distance between two triplets = mean C-alpha distance across all 9 cross-pairs
# (residues shared between the two triplets contribute 0, pulling overlapping/adjacent
# triplets together automatically)

spatial_dist = np.zeros((n_top, n_top))
for a in range(n_top):
    for b in range(a + 1, n_top):
        d = np.mean([ca_dist_matrix[i, j] for i in top_triplets[a] for j in top_triplets[b]])
        spatial_dist[a, b] = spatial_dist[b, a] = d

def normalise(mat):
    mx = mat.max()
    return mat / mx if mx > 0 else mat

# score similarity term -- "cluster the top p_eq together": two triplets with
# very different predicted scores are pulled apart even if they sit right next
# to each other, so a strong local hotspot doesn't get diluted by weaker
# neighbours merging into the same cluster.

score_z    = (top_scores - top_scores.mean()) / (top_scores.std() + 1e-12)
score_dist = squareform(pdist(score_z.reshape(-1, 1), metric='euclidean'))

# spatial dominates (keeps clusters physically compact); chemistry and score
# similarity shape which nearby triplets actually merge

W_SCORE = 0.15 #score distance 
W_SPATIAL = 0.60 #spatial distance 
W_CHEM= 0.25 #chemical distance 

combined_dist = (W_SPATIAL * normalise(spatial_dist)
                  + W_CHEM * normalise(chem_dist)
                  + W_SCORE * normalise(score_dist))

np.fill_diagonal(combined_dist, 0.0)


# this is where things get confusing 

Z = linkage(squareform(combined_dist, checks=False), method='average') # just provdes the triplet clustering

MAX_SPAN = 6  # residues

# Locally-constrained agglomerative merge: walk the linkage tree bottom-up

n_top_ = len(top_triplets)
dsu_parent = list(range(n_top_))

def find(x):
    while dsu_parent[x] != x:
        dsu_parent[x] = dsu_parent[dsu_parent[x]]
        x = dsu_parent[x]
    return x

leaves_of    = {i: {i} for i in range(n_top_)}
residues_of  = {i: set(top_triplets[i]) for i in range(n_top_)}
capped       = set()

for step, (a, b, _dist, _cnt) in enumerate(Z):
    a, b = int(a), int(b)
    new_id = n_top_ + step
    if a in capped or b in capped:
        capped.add(new_id)
        leaves_of[new_id]   = leaves_of[a] | leaves_of[b]
        residues_of[new_id] = residues_of[a] | residues_of[b]
        continue
    merged_residues     = residues_of[a] | residues_of[b]
    leaves_of[new_id]   = leaves_of[a] | leaves_of[b]
    residues_of[new_id] = merged_residues
    if len(merged_residues) > MAX_SPAN:
        capped.add(new_id)
        continue
    ra, rb = find(next(iter(leaves_of[a]))), find(next(iter(leaves_of[b])))
    if ra != rb:
        dsu_parent[ra] = rb

cluster_ids_raw = np.array([find(i) for i in range(n_top_)])

def cluster_span(labels, cid):
    residues = set()
    for i in np.where(labels == cid)[0]:
        residues.update(top_triplets[i])
    return len(residues)

n_clusters_found = len(set(cluster_ids_raw))
spans_found = [cluster_span(cluster_ids_raw, c) for c in set(cluster_ids_raw)]

raw_sum = pd.Series(top_scores).groupby(cluster_ids_raw).sum().sort_values(ascending=False)
relabel = {raw_id: rank for rank, raw_id in enumerate(raw_sum.index, start=1)}
cluster_ids = np.array([relabel[c] for c in cluster_ids_raw])

# res_lo    = [min(resid_map[i][0] for i in t) for t in top_triplets]
# res_hi    = [max(resid_map[i][0] for i in t) for t in top_triplets]
res_tuple = [tuple(resid_map[i][0] for i in t) for t in top_triplets]
res_fmt   = [fmt_triplet(t) for t in top_triplets]   # e.g. ('VAL18', 'LEU17', 'PHE19')

cluster_df = pd.DataFrame({
    'triplet':            [' - '.join(fmt_triplet(t)) for t in top_triplets],
    'cluster':            cluster_ids,
    'pred_score':         top_scores,
    'hydrophobicity':     chem_raw[:, 0],
    'n_hbond_donors':     chem_raw[:, 1].astype(int),
    'n_hbond_acceptors':  chem_raw[:, 2].astype(int),
    'net_charge':         chem_raw[:, 3],
    # 'res_lo':             res_lo,
    # 'res_hi':             res_hi,
    'res_tuple':          res_tuple,
    'res_fmt':            res_fmt,
}).sort_values(['cluster', 'pred_score'], ascending=[True, False])

cluster_df.drop(columns=['res_tuple', 'res_fmt']).to_csv('figures/top_triplet_clusters.csv', index=False)

def n_distinct_residues(group):
    residues = set()
    for r in group['res_tuple']:
        residues.update(r)
    return len(residues)

def residues_formatted(group):
    # union of every residue (as 'VAL18' style) touched by any triplet in
    # this cluster, sorted by residue number rather than alphabetically
    residues = set()
    for r in group['res_fmt']:
        residues.update(r)
    return ','.join(sorted(residues, key=lambda s: int(''.join(ch for ch in s if ch.isdigit()))))

cluster_summary = cluster_df.groupby('cluster').agg(
    n_triplets=('triplet', 'count'),
    sum_score=('pred_score', 'sum'),
    mean_score=('pred_score', 'mean'),
    # res_lo=('res_lo', 'min'),
    # res_hi=('res_hi', 'max'),
    mean_hydrophobicity=('hydrophobicity', 'mean'),
    mean_hbond_donors=('n_hbond_donors', 'mean'),
    mean_hbond_acceptors=('n_hbond_acceptors', 'mean'),
).round(3)

cluster_summary['n_residues']  = cluster_df.groupby('cluster').apply(n_distinct_residues)
cluster_summary['residues']    = cluster_df.groupby('cluster').apply(residues_formatted)
print(cluster_summary[['n_triplets', 'sum_score', 'n_residues', 'residues']].head(15))
# cluster_summary['seq_range']   = cluster_summary['res_hi'] - cluster_summary['res_lo'] + 1
# cluster_summary = cluster_summary.sort_values('sum_score', ascending=False)
# cluster_summary.to_csv('figures/top_triplet_cluster_summary.csv')

# plotting

OTHER_COLOR    = ps.BASELINE_GRAY

ps.set_publication_theme(font_scale=1.3)


def find_knee(values_desc, window=50):
    values_desc = values_desc[:window]
    x = np.arange(len(values_desc)) / (len(values_desc) - 1)
    y = (values_desc - values_desc.min()) / (np.ptp(values_desc) + 1e-12)
    x0, y0, x1, y1 = x[0], y[0], x[-1], y[-1]
    dx, dy = x1 - x0, y1 - y0
    dist = np.abs(dy * (x - x0) - dx * (y - y0)) / np.hypot(dx, dy)
    return int(np.argmax(dist))


sum_sorted  = np.sort(cluster_summary['sum_score'].values)[::-1]
mean_sorted = np.sort(cluster_summary['mean_score'].values)[::-1]

sum_knee  = find_knee(sum_sorted)
mean_knee = find_knee(mean_sorted)


sum_norm  = (sum_sorted  - sum_sorted.min())  / (np.ptp(sum_sorted)  + 1e-12)
mean_norm = (mean_sorted - mean_sorted.min()) / (np.ptp(mean_sorted) + 1e-12)

cluster_ranks = np.arange(1, len(sum_sorted) + 1)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(cluster_ranks, sum_norm,  color=ps.PAIRED_LIGHT, lw=2, label='Σ score (normalised)')
ax.plot(cluster_ranks, mean_norm, color=ps.PAIRED_DARK, lw=2, label='mean score (normalised)')
ax.axvline(sum_knee + 1,  color=ps.PAIRED_LIGHT, ls='--', lw=1.2, alpha=0.8)
        #    label=f'Σ score knee (cluster rank {sum_knee + 1})')
ax.axvline(mean_knee + 1, color=ps.PAIRED_DARK, ls='--', lw=1.2, alpha=0.8)
        #    label=f'mean score knee (cluster rank {mean_knee + 1})')
ax.set_xlabel('Cluster rank', fontsize=18)
ax.set_ylabel('Normalised score', fontsize=18)
ax.tick_params(axis='both', labelsize=13)
ax.legend(frameon=False, fontsize=16)
sns.despine(fig=fig)
fig.tight_layout()
fig.savefig('figures/cluster_score_knee.png', dpi=300, bbox_inches='tight')
plt.show()

# this plot sht eoutput of the clusters in a dimernion to show distance betwee who is clsie to wh

print(f"\nCluster Σscore knee at rank {sum_knee + 1} / {len(sum_sorted)} "
      f"(value={sum_sorted[sum_knee]:.3f})")
print(f"Cluster mean_score knee at rank {mean_knee + 1} / {len(mean_sorted)} "
      f"(value={mean_sorted[mean_knee]:.3f})")

# drop singleton clusters entirely -- a lone triplet isn't a "cluster" and
# including it in the MDS fit only pulls/distorts the layout for the
# clusters that actually matter
cluster_sizes = pd.Series(cluster_ids).value_counts()
keep_mask = np.array([cluster_sizes[c] > 1 for c in cluster_ids])
print(f"MDS map: keeping {keep_mask.sum()} / {len(cluster_ids)} triplets "
      f"in {cluster_sizes[cluster_sizes > 1].shape[0]} non-singleton clusters "
      f"(dropped {(~keep_mask).sum()} singletons)")

cluster_ids_mds  = cluster_ids[keep_mask]
top_scores_mds   = top_scores[keep_mask]
combined_dist_mds = combined_dist[np.ix_(keep_mask, keep_mask)]

N_HIGHLIGHT    = sum_knee + 1
# brand-rotation categorical colors, sized to however many clusters are above
# the knee cutoff -- ps.categorical_blues() covers the AZURE/GREEN/VIOLET/
# AMBER validated 4-color rotation, extends with PINK/CHARTREUSE for 5-6,
# then falls back to an OKLCH sweep across that same hue range beyond that
# (see plot_style.py; this replaces a local hand-rolled duplicate of the
# same logic that used to live here, anchored to the old blue-only PRIMARY)
CLUSTER_COLORS = ps.categorical_blues(N_HIGHLIGHT)


tsne_perplexity = min(30, (keep_mask.sum() - 1) // 3)
tsne_coords = TSNE(n_components=2, metric='precomputed', init='random', random_state=0,
                    perplexity=tsne_perplexity).fit_transform(combined_dist_mds)

# score -> marker size, shared by every scatter point below: same min/range
# across both plots, so size is comparable rather than rescaled per-plot
score_floor = top_scores_mds.min()
score_span = np.ptp(top_scores_mds) + 1e-12

# two views of the same t-SNE embedding: the full one (every cluster above
# the Sigma-score knee gets its own color) and a top-5-only one where just
# the 5 highest-Sigma-score clusters stay highlighted -- everything else
# (including clusters 6..N_HIGHLIGHT, which WERE colored in the full view)
# drops into the grey "other" bucket, drawn smaller and further behind,
# while the 5 highlighted clusters are drawn larger so they read as the
# main story
embedding_plots = [
    # n_highlight, out_path,                                    other_size_base, other_size_range, main_size_base, main_size_range
    (N_HIGHLIGHT, 'figures/top_triplet_clusters_tsne.png',       30, 180, 90, 260),
    (5,           'figures/top_triplet_clusters_tsne_top5.png',  15, 90, 150, 420),
]

for n_highlight, out_path, other_size_base, other_size_range, main_size_base, main_size_range in embedding_plots:
    fig, ax = plt.subplots(figsize=(10, 10))

    below_cutoff_mask = cluster_ids_mds > n_highlight
    if below_cutoff_mask.any():
        other_sizes = other_size_base + other_size_range * (top_scores_mds[below_cutoff_mask] - score_floor) / score_span
        ax.scatter(tsne_coords[below_cutoff_mask, 0], tsne_coords[below_cutoff_mask, 1], s=other_sizes, color=OTHER_COLOR,
                   edgecolor='none', alpha=0.25, zorder=1,
                   label='Other clusters (below cutoff Σscore)')

    for cid in range(1, n_highlight + 1):
        cluster_mask = cluster_ids_mds == cid
        if not cluster_mask.any():
            continue
        cluster_sizes = main_size_base + main_size_range * (top_scores_mds[cluster_mask] - score_floor) / score_span
        cluster_score_sum = top_scores_mds[cluster_mask].sum()
        ax.scatter(tsne_coords[cluster_mask, 0], tsne_coords[cluster_mask, 1], s=cluster_sizes, color=CLUSTER_COLORS[cid - 1],
                   edgecolor='white', linewidth=0.6, alpha=0.9, zorder=2,
                   label=f'Cluster {cid} (n={cluster_mask.sum()}, Σscore={cluster_score_sum:.2f})')

    ax.set_xlabel('t-SNE Dim 1', fontsize=24)
    ax.set_ylabel('t-SNE Dim 2', fontsize=24)
    ax.tick_params(axis='both', labelsize=22)
    # legend = ax.legend(frameon=False, fontsize=22, loc='upper left', bbox_to_anchor=(1.02, 1.0), scatterpoints=1)
    # for handle in getattr(legend, 'legend_handles', None) or legend.legendHandles:
    #     handle.set_sizes([100])
    sns.despine(fig=fig)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()

cluster_summary['sum_score_percent'] = cluster_summary['sum_score'] / cluster_summary['sum_score'].sum() * 100

# same cutoff used to highlight clusters in the MDS plot -- only clusters
# above the Sigma-score knee are shown here
top_cluster_summary = cluster_summary.loc[cluster_summary.index <= N_HIGHLIGHT]

fig, ax = plt.subplots(figsize=(8, 5))
bar_colors = [CLUSTER_COLORS[cid - 1] for cid in top_cluster_summary.index]
ax.bar(top_cluster_summary.index.astype(str), top_cluster_summary['sum_score_percent'],
       color=bar_colors, edgecolor='white', linewidth=0.6)
for x, pct in enumerate(top_cluster_summary['sum_score_percent']):
    ax.text(x, pct, f'{pct:.1f}%', ha='center', va='bottom', fontsize=13)
ax.set_xlabel('Cluster', fontsize=18)
ax.set_ylabel('Share of total Σ score (%)', fontsize=18)
ax.tick_params(axis='both', labelsize=13)
sns.despine(fig=fig)
fig.tight_layout()
fig.savefig('figures/top_cluster_score_percent.png', dpi=300, bbox_inches='tight')
plt.show()

