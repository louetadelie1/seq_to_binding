from transformers import AutoModel, AutoTokenizer
import torch
import glob
import pickle
import linecache
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import seaborn as sns
import plot_style as ps
from itertools import combinations
import random
from collections import OrderedDict, Counter
import ast
import re
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

K_NN         = 50
BATCH_SIZE   = 4096
TOP_ACTIVES  = 20
EF_FRACTIONS = [0.01, 0.05, 0.10]
WEIGHTS      = '/ptmp/adlouet/camb/sequence_to_binding_paths/fine_tuning_experiments/exp_5_boltz_distances_inf_pred/weights/best_model_change_ranking.pt'
ordered_protein_base    = "//ptmp/adlouet/camb/sequence_to_binding_paths/post_processed_ordered_proteins"

# using the weights from bolt trained CA distances
numbering_offset = {
    "IL-2": 0.0,
    "TEM-1_beta-lactamase": 25.0,
    "DC-SIGN": 248.0,
    "SH3_domain": 84.0,
    "Acetylcholinesterase": 1.0,
    "BRD4_bromodomain_BD1": 41.0,
    "Trypsin": 0.0,          # blank in the source list -- no confirmed offset yet
    "KRAS_G12C": 0.0,
    "HIV1_protease": 0.0,
    "Influenza_neuraminidase": None,  # offset 82 only validated for the 150-loop residues (118/151/152);
                                       # the other 5 residues in that pocket use a different (N2) numbering
                                       # convention and are not reliable -- excluded until resolved
    "CYP3A4": 21.0,
}

# display names for figures -- keys are the raw post_processed_ordered_proteins
# folder names (also numbering_offset's keys), values are what actually gets
# drawn on axes/labels. Never rename the dict keys above -- only this mapping
# changes what the reader sees.
PROTEIN_NAMES = {
    "IL-2":                     "IL-2",
    "TEM-1_beta-lactamase":     "TEM-1 β-lactamase",
    "DC-SIGN":                  "DC-SIGN",
    "SH3_domain":               "SH3 domain",
    "Acetylcholinesterase":     "Acetylcholinesterase",
    "BRD4_bromodomain_BD1":     "BRD4 bromodomain (BD1)",
    "Trypsin":                  "Trypsin",
    "KRAS_G12C":                "KRAS G12C",
    "HIV1_protease":            "HIV-1 protease",
    "Influenza_neuraminidase":  "Influenza neuraminidase",
    "CYP3A4":                   "CYP3A4",
}

# display names for pockets -- keys are (protein, state) as they appear in
# pockets_with_per_row_sources.csv, values are the tidied label. Falls back
# to a plain underscore->space swap (_tidy_state) for any (protein, state)
# pair not listed here, so a new CSV row never crashes the figure, just
# renders untidied until it's added below.
POCKET_NAMES = {
    ("IL-2", "apo_closed_hotspot"):                                   "Apo closed hotspot",
    ("IL-2", "apo_transient_open_hotspot"):                           "Apo transient-open hotspot",
    ("IL-2", "ligand_bound_open_groove"):                             "Ligand-bound open groove",
    ("TEM-1_beta-lactamase", "catalytic_site"):                       "Catalytic site",
    ("TEM-1_beta-lactamase", "enhanced_sampling_variant_site"):       "Enhanced-sampling variant site",
    ("DC-SIGN", "gatekeeper_closed_mimic"):                           "Gatekeeper (closed mimic)",
    ("DC-SIGN", "gatekeeper_open_residue"):                           "Gatekeeper (open)",
    ("SH3_domain", "orientation_determining_salt_bridge"):            "Orientation-determining salt bridge",
    ("Acetylcholinesterase", "CAS_catalytic_anionic_site"):           "CAS (catalytic anionic site)",
    ("Acetylcholinesterase", "PAS_peripheral_anionic_site"):          "PAS (peripheral anionic site)",
    ("Acetylcholinesterase", "acyl_pocket_inward_state"):             "Acyl pocket (inward state)",
    ("Acetylcholinesterase", "outward_PAS_dominant_state"):           "PAS-dominant (outward state)",
    ("Acetylcholinesterase", "inward_CAS_state"):                     "CAS-dominant (inward state)",
    ("BRD4_bromodomain_BD1", "KAc_core_recognition"):                 "KAc core recognition",
    ("BRD4_bromodomain_BD1", "WPF_shelf"):                            "WPF shelf",
    ("BRD4_bromodomain_BD1", "ZA_BC_loop_variable_contacts"):         "ZA/BC loop (variable contacts)",
    ("Trypsin", "catalytic_triad"):                                   "Catalytic triad",
    ("Trypsin", "S3_secondary_electronegative_site"):                 "S3 secondary site",
    ("Trypsin", "S4_final_bound_S1_pocket"):                          "S4 / S1 pocket",
    ("KRAS_G12C", "switch_II_pocket_open_cryptic_His95_rotated"):     "Switch-II pocket (His95-rotated)",
    ("KRAS_G12C", "switch_II_pocket_extended_contacts"):              "Switch-II pocket (extended contacts)",
    ("HIV1_protease", "flap_tip_residues"):                           "Flap-tip residues",
    ("HIV1_protease", "catalytic_dyad"):                              "Catalytic dyad",
    ("Influenza_neuraminidase", "closed_150loop_catalytic_site"):     "150-loop catalytic site (closed)",
    ("CYP3A4", "two_ligand_binding_contacts"):                        "Two-ligand binding contacts",
}


random.seed(42)   # reproducible ZINC-pair sampling across runs

tokeniser  = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
esm_model  = AutoModel.from_pretrained("facebook/esm2_t12_35M_UR50D").to(device)

# Just as an internal TEST KEEP AWAY FROM WONDERING EYES
def make_binding_profile(triplets, scores, n_res):
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


def get_binding_regions(path):
    with open(path) as f:
        text = f.read()
    match = re.search(r'^binding_regions\t(.*)$', text, re.MULTILINE)
    return match.group(1) if match else None


model = Model().to(device)
model.load_state_dict(torch.load(WEIGHTS, map_location=device))
model.eval()

os.makedirs('figures', exist_ok=True)

X_GRID = np.linspace(0, 1, 200)   

proteins=sorted(glob.glob(f'/{ordered_protein_base}/*'))

hit_records = []   # (protein_name, best_rank_hit_or_None) -- feeds the H coefficient summary at the end
hit_records_knee = []

RECALL_THRESHOLD = 0.75   # a pocket counts as "recovered" once >=75% of its residues are in the accepted clusters
pocket_hit_records      = []   # (protein_name, pocket_idx, n_pocket_residues, recovery_rank_or_None, precision_at_recovery_or_None, best_recall_reached)
pocket_hit_records_knee = []   # same, but the walk is restricted to clusters at/above the knee cutoff

pocket_recovery_table = []   # one row per literature pocket: true vs predicted residues, accuracy rank, knee cutoff

GROUND_TRUTH_COLOR = ps.PAIRED_DARK   # violet wash, distinct from the green predicted-profile line via hue as well as alpha/fill vs. line
POCKET_CMAP        = plt.cm.Reds   # darker = higher-ranked (rank 1 = highest predicted score)

# ── one shared grid, filled in live as each protein is processed ───────────
ps.set_publication_theme(font_scale=1.3)
ALL_NCOLS = min(3, max(len(proteins), 1))
ALL_NROWS = int(np.ceil(len(proteins) / ALL_NCOLS))
fig_all, axes_all = plt.subplots(ALL_NROWS, ALL_NCOLS, figsize=(5.5 * ALL_NCOLS, 3 * ALL_NROWS), squeeze=False)
plot_idx = 0
plot_data=[]

protein_data = {}
for protein in proteins:
    try:
        protein_name = protein.split('/')[-1]
        binding_regions_raw = get_binding_regions(f'{protein}/binding_profile.txt')
    except:
        continue

    if not binding_regions_raw:
        print(f"skipping {protein_name}: binding_profile.txt has no binding_regions")
        continue

    offset = numbering_offset.get(protein_name, 0.0)
    if offset is None:
        print(f"skipping {protein_name}: numbering offset not yet confirmed against UniProt")
        continue

    # binding_profile.txt residue numbers use whatever convention the source literature used
    # (UniProt canonical, Ambler numbering, mature-chain numbering, full-length numbering
    # against a truncated domain construct, ...) -- convert to local 1-indexed positions
    # matching sequence.fasta / ca_dist_matrix_boltz.pkl.
    binding_regions_truth = [np.array(r) - int(offset) for r in ast.literal_eval(binding_regions_raw)]

    fasta_path = f'{protein}/sequence.fasta'
    boltz_dist_path = f'{protein}/ca_dist_matrix_boltz.pkl'

    if not os.path.exists(fasta_path):
        print(f"skipping {protein}: no sequence.fasta yet")
        continue
    if not os.path.exists(boltz_dist_path):
        print(f"skipping {protein}: no ca_dist_matrix_boltz.pkl yet "
              f"(run generate_boltz_distances.py first)")
        continue

    seq = linecache.getline(fasta_path, 2).strip()
    L = len(seq)

    flat = list(set(int(r) for region in binding_regions_truth for r in region))

    if len(flat) >= len(seq):
        continue

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


    ranked_triplets = sorted(zip(valid_triplets, preds), key=lambda t: t[1], reverse=True)
    pred_profile = make_binding_profile(valid_triplets, np.array(preds), n_res)
    # print(f"\n{protein_name}: top predicted peq triplets (residue indices, predicted p_eq):")
    # for rank, (trio, score) in enumerate(ranked_triplets[:50], start=1):
    #     print(f"{rank:>4}  {trio}  {score:.4f}")

    residue_number = np.arange(1, n_res + 1)

    protein_data[protein_name] = {
        'ranked_triplets':       ranked_triplets,
        'seq':                   seq,
        'ca_dist_matrix':        ca_dist_matrix,
        'binding_regions_truth': binding_regions_truth,
        'residue_number':        residue_number,
        'pred_profile':          pred_profile,
    }

    ps.set_publication_theme(font_scale=1.05)
    fig, ax = plt.subplots(figsize=(9, 3))

    for i, region in enumerate(binding_regions_truth):
        runs = np.split(region, np.where(np.diff(region) != 1)[0] + 1)
        for j, run in enumerate(runs):
            ax.axvspan(run.min() - 0.5, run.max() + 0.5, color=GROUND_TRUTH_COLOR, alpha=0.3, zorder=1,
                       label='Ground truth binding region' if (i == 0 and j == 0) else None)

    ax.plot(residue_number, pred_profile, color=ps.PAIRED_LIGHT, lw=1.8, zorder=3, label='Predicted')
    ax.fill_between(residue_number, pred_profile, color=ps.PAIRED_LIGHT, alpha=0.15, zorder=2)

    ax.set_xlim(residue_number[0], residue_number[-1])
    ax.set_ylim(0, 1)
    ax.set_xlabel('Residue number', fontsize=9)
    ax.set_ylabel('Predicted binding score', fontsize=9)
    ax.set_title(protein_name, fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc='upper right')
    sns.despine(fig=fig)
    fig.tight_layout()
    plt.close()




############################################ Clustering binding profiles and seeing if they match ############################################
for protein_name, pdata in protein_data.items():
    ranked_triplets       = pdata['ranked_triplets']
    seq                   = pdata['seq']
    ca_dist_matrix        = pdata['ca_dist_matrix']
    binding_regions_truth = pdata['binding_regions_truth']
    residue_number        = pdata['residue_number']
    pred_profile          = pdata['pred_profile']

    sorted_scores = np.array([s for _, s in ranked_triplets])
    ranks = np.arange(len(sorted_scores))

    x_norm = ranks / (len(ranks) - 1)
    y_norm = (sorted_scores - sorted_scores.min()) / (np.ptp(sorted_scores) + 1e-12)
    x0, y0, x1, y1 = x_norm[0], y_norm[0], x_norm[-1], y_norm[-1]
    dx, dy = x1 - x0, y1 - y0
    chord_len = np.hypot(dx, dy)
    dist_to_chord = np.abs(dy * (x_norm - x0) - dx * (y_norm - y0)) / chord_len
    knee_idx = int(np.argmax(dist_to_chord))
    # print(f"\nKnee (max chord distance) at rank {knee_idx} / {len(sorted_scores):,} "
    #       f"(top {knee_idx / len(sorted_scores):.2%}), score={sorted_scores[knee_idx]:.4f}")

    zero_cross = int(np.searchsorted(-sorted_scores, 0))  # sorted_scores is descending

    ZOOM_N = 1000
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, n in zip(axes, [len(sorted_scores), ZOOM_N]):
        ax.plot(ranks[:n], sorted_scores[:n], color=ps.PAIRED_LIGHT, lw=1.6)
        if knee_idx < n:
            ax.axvline(knee_idx, color=ps.PAIRED_DARK, ls='--', lw=1.2, label=f'knee (rank {knee_idx})')
        ax.set_xlabel('Rank (descending predicted score)')
        ax.set_ylabel('Predicted p_eq')
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_title(f'Full curve (n={len(sorted_scores):,})')
    axes[1].set_title(f'Zoomed to top {ZOOM_N:,} ranks')
    sns.despine(fig=fig)
    fig.tight_layout()
    plt.close()


    ONE_TO_THREE = {
        'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
        'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
        'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
        'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL',
    }

    resid_map = {i: (i + 1, ONE_TO_THREE[aa]) for i, aa in enumerate(seq)}

    def fmt_triplet(t):
        return tuple(f"{resid_map[i][1]}{resid_map[i][0]}" for i in t)

    W_SPATIAL    = 0.5     
    W_CHEM       = 0.5

    KYTE_DOOLITTLE = {
        'ILE': 4.5, 'VAL': 4.2, 'LEU': 3.8, 'PHE': 2.8, 'CYS': 2.5, 'MET': 1.9,
        'ALA': 1.8, 'GLY': -0.4, 'THR': -0.7, 'SER': -0.8, 'TRP': -0.9,
        'TYR': -1.3, 'PRO': -1.6, 'HIS': -3.2, 'GLU': -3.5, 'GLN': -3.5,
        'ASP': -3.5, 'ASN': -3.5, 'LYS': -3.9, 'ARG': -4.5,
    }
    HBOND_DONORS    = {'ARG', 'LYS', 'TRP', 'ASN', 'GLN', 'HIS', 'SER', 'THR', 'TYR'}
    HBOND_ACCEPTORS = {'ASP', 'GLU', 'ASN', 'GLN', 'HIS', 'SER', 'THR', 'TYR'}
    CHARGE          = {'ARG': 1.0, 'LYS': 1.0, 'HIS': 0.5, 'ASP': -1.0, 'GLU': -1.0}


    #  problematic down below . solve asap
    n_top = 3000  #zero_cross <- this would be noce to have but this is how the matrix explodes. either reduce knn or set this as a hard stop
    top_triplets = [t for t, _ in ranked_triplets[:n_top]]
    top_scores   = np.array([s for _, s in ranked_triplets[:n_top]])
    # print(f"\nClustering top {n_top} triplets (score > 0, {n_top / len(ranked_triplets):.2%} of {len(ranked_triplets):,} scored)")

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

    T = np.array(top_triplets)   # (n_top, 3) residue indices
    spatial_dist = np.empty((n_top, n_top))
    SPATIAL_BLOCK = 500   # bounds peak memory to ~ SPATIAL_BLOCK * n_top * 9 floats per block
    for start in range(0, n_top, SPATIAL_BLOCK):
        end = min(start + SPATIAL_BLOCK, n_top)
        # ca_dist_matrix[T[start:end]][:, None, :, None] pairs each of the 3
        # residues in a block triplet against each of the 3 residues in every
        # other triplet -> (block, n_top, 3, 3), then average over the 3x3 pairs.
        block = ca_dist_matrix[T[start:end, None, :, None], T[None, :, None, :]]
        spatial_dist[start:end] = block.mean(axis=(2, 3))

    def normalise(mat):
        mx = mat.max()
        return mat / mx if mx > 0 else mat

    score_z    = (top_scores - top_scores.mean()) / (top_scores.std() + 1e-12)
    score_dist = squareform(pdist(score_z.reshape(-1, 1), metric='euclidean'))

    W_SCORE = 0.25 #score distance 
    W_SPATIAL = 0.6 #spatial distance 
    W_CHEM= 0.15 #chemical distance 

    combined_dist = (W_SPATIAL * normalise(spatial_dist)
                    + W_CHEM * normalise(chem_dist)
                    + W_SCORE * normalise(score_dist))

    np.fill_diagonal(combined_dist, 0.0)

    Z = linkage(squareform(combined_dist, checks=False), method='average') # just provdes the triplet clustering

    MAX_SPAN = 15  # residues
    MERGE_PERCENTILE = 90 # percentile to cutoff at t=something valude

    def cluster_span(labels, cid): # this goes line by line in Z collapsing reisdues in each clister
        residues = set()
        for i in np.where(labels == cid)[0]:
            residues.update(top_triplets[i])
        return len(residues)

    merge_dists = np.unique(Z[:, 2])
    MAX_MERGE = np.percentile(merge_dists, MERGE_PERCENTILE)

    cluster_ids_raw, best_t = fcluster(Z, t=merge_dists[0], criterion='distance'), merge_dists[0]

    for t in merge_dists:
        labels = fcluster(Z, t=t, criterion='distance')
        spans = [cluster_span(labels, cid) for cid in set(labels)]
        if max(spans) <= MAX_SPAN and t <= MAX_MERGE:
            cluster_ids_raw, best_t = labels, t   # keep growing -- take the largest valid t

    n_clusters_found = len(set(cluster_ids_raw))
    spans_found = [cluster_span(cluster_ids_raw, c) for c in set(cluster_ids_raw)]

    raw_sum = pd.Series(top_scores).groupby(cluster_ids_raw).sum().sort_values(ascending=False)
    relabel = {raw_id: rank for rank, raw_id in enumerate(raw_sum.index, start=1)}
    cluster_ids = np.array([relabel[c] for c in cluster_ids_raw])

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
        'res_tuple':          res_tuple,
        'res_fmt':            res_fmt,
    }).sort_values(['cluster', 'pred_score'], ascending=[True, False])

    def n_distinct_residues(group):
        residues = set()
        for r in group['res_tuple']:
            residues.update(r)
        return len(residues)

    def residues_formatted(group):
        residues = set()
        for r in group['res_fmt']:
            residues.update(r)
        return ','.join(sorted(residues, key=lambda s: int(''.join(ch for ch in s if ch.isdigit()))))

    cluster_summary = cluster_df.groupby('cluster').agg(
        n_triplets=('triplet', 'count'),
        sum_score=('pred_score', 'sum'),
        mean_score=('pred_score', 'mean'),
        mean_hydrophobicity=('hydrophobicity', 'mean'),
        mean_hbond_donors=('n_hbond_donors', 'mean'),
        mean_hbond_acceptors=('n_hbond_acceptors', 'mean'),
    ).round(3)

    cluster_summary['n_residues']  = cluster_df.groupby('cluster').apply(n_distinct_residues)
    cluster_summary['residues']    = cluster_df.groupby('cluster').apply(residues_formatted)

    OTHER_COLOR    = ps.BASELINE_GRAY

    ps.set_publication_theme(font_scale=1.15)

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

    # this decides where we can to cut out the clusters 
    cluster_ranks = np.arange(1, len(sum_sorted) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(cluster_ranks, sum_norm,  color=ps.PAIRED_LIGHT, lw=2, label='Σ score (normalised)')
    ax.plot(cluster_ranks, mean_norm, color=ps.PAIRED_DARK, lw=2, label='mean score (normalised)')
    ax.axvline(sum_knee + 1,  color=ps.PAIRED_LIGHT, ls='--', lw=1.2, alpha=0.8,
            label=f'Σ score knee (cluster rank {sum_knee + 1})')
    ax.axvline(mean_knee + 1, color=ps.PAIRED_DARK, ls='--', lw=1.2, alpha=0.8,
            label=f'mean score knee (cluster rank {mean_knee + 1})')
    ax.set_xlabel('Cluster rank (sorted independently by each metric)', fontsize=15)
    ax.set_ylabel('Normalised score', fontsize=15)
    ax.tick_params(axis='both', labelsize=15)
    ax.legend(frameon=False, fontsize=10)
    sns.despine(fig=fig)
    fig.tight_layout()
    plt.close()

    # print(f"\nCluster Σscore knee at rank {sum_knee + 1} / {len(sum_sorted)} "
    #       f"(value={sum_sorted[sum_knee]:.3f})")
    # print(f"Cluster mean_score knee at rank {mean_knee + 1} / {len(mean_sorted)} "
    #       f"(value={mean_sorted[mean_knee]:.3f})")

    cluster_summary['sum_score_percent'] = cluster_summary['sum_score'] / cluster_summary['sum_score'].sum() * 100

    top_cluster_above_knee = cluster_summary.index[cluster_summary.index <= sum_knee + 1]
    # print(f"\nTop clusters above the knee (cluster IDs): {list(top_cluster_above_knee)}")



###### Now we compare to experimental ground truth data

    cluster_residues = {
        cid: {r for t in cluster_df.loc[cluster_df['cluster'] == cid, 'res_tuple'] for r in t}
        for cid in cluster_summary.index
    }
    pocket_residue_sets = [set(int(r) for r in region) for region in binding_regions_truth]

    recovery_rank={}
    for pocket_idx, truth in enumerate(pocket_residue_sets):
        matched_rank, matched_prediction, matched_overlap = None, None, None
        for keys,prediction in cluster_residues.items():
            intersection = truth & prediction
            union= truth | prediction
            #recall= len(intersection) / len(truth)
            overlap_coefficient = len(intersection) / min(len(truth), len(prediction))
            if overlap_coefficient >= 0.5:
                recovery_rank[frozenset(truth)]=[keys]
                matched_rank, matched_prediction, matched_overlap = keys, prediction, overlap_coefficient
                break

        pocket_recovery_table.append({
            'protein':             protein_name,
            'pocket_idx':          pocket_idx,
            'true_pocket':         sorted(truth),
            'predicted_pocket':    sorted(matched_prediction) if matched_prediction is not None else None,
            'accuracy_rank':       matched_rank,
            'overlap_coefficient': round(matched_overlap, 3) if matched_overlap is not None else None,
            'knee':                sum_knee,
        })

    print(f'{protein_name} has {recovery_rank},and the knee is {sum_knee}')

########### Summary dataframe: true pockets vs predicted pockets vs accuracy rank, across all proteins ###########
pocket_recovery_df = pd.DataFrame(pocket_recovery_table)
print("\n" + "=" * 80)
print("Pocket recovery summary (true vs. predicted, per literature pocket)")
print("=" * 80)
print(pocket_recovery_df.to_string(index=False))


########### Cross-reference against per-pocket literature sources ###########
SOURCES_CSV = "//ptmp/adlouet/camb/sequence_to_binding_paths/post_processed_ordered_proteins/pockets_with_per_row_sources.csv"
sources_df = pd.read_csv(SOURCES_CSV)
sources_df['pocket_idx'] = sources_df.groupby('protein_system').cumcount()
sources_df = sources_df.rename(columns={'protein_system': 'protein'})

pocket_recovery_df = pocket_recovery_df.merge(
    sources_df[['protein', 'pocket_idx', 'state', 'source_article', 'source_pdb']],
    on=['protein', 'pocket_idx'], how='left'
)

# numbered citation list, one entry per unique (protein, source_article) pair
print("\n" + "=" * 80)
print("Literature sources per protein (numbered -- referenced in the figure caption)")
print("=" * 80)
citation_number = {}
ref_id = 1
for pname in pocket_recovery_df['protein'].unique():
    print(f"\n{PROTEIN_NAMES.get(pname, pname)}:")
    seen = pocket_recovery_df.loc[pocket_recovery_df['protein'] == pname, ['source_article', 'source_pdb']].drop_duplicates()
    for _, r in seen.iterrows():
        citation_number[(pname, r['source_article'])] = ref_id
        pdb_note = f" [PDB {r['source_pdb']}]" if pd.notna(r['source_pdb']) and r['source_pdb'] else ""
        print(f"  [{ref_id}] {r['source_article']}{pdb_note}")
        ref_id += 1

pocket_recovery_df['citation_ref'] = pocket_recovery_df.apply(
    lambda r: citation_number.get((r['protein'], r['source_article'])), axis=1
)

pocket_recovery_df.to_csv('pocket_recovery_summary.csv', index=False)


########### Illustration: recovery rank per pocket, colored by within-knee vs beyond-knee vs never recovered ###########
INK_PRIMARY   = ps.INK_PRIMARY
STATUS_GOOD   = ps.PAIRED_DARK    # recovered within knee cutoff -- violet, the more emphatic of the pair
SERIES_BLUE   = ps.PAIRED_LIGHT   # recovered beyond knee cutoff -- green, the lighter of the pair
CRITICAL      = ps.CRITICAL     # reserved accent: never recovered (the one true alert state)
BASELINE_GRAY = ps.BASELINE_GRAY
SURFACE       = ps.SURFACE
BAND_COLOR    = ps.BAND_COLOR

def _tidy_state(protein, s):
    return POCKET_NAMES.get((protein, s), str(s).replace('_', ' '))

# one section-header row per protein, followed by its pocket rows -- avoids repeating the
# protein name on every row and keeps the y-axis legible (forest-plot style grouping)
grouped = pocket_recovery_df.sort_values(['protein', 'pocket_idx']).groupby('protein', sort=False)
combined_rows = []
for pname, g in grouped:
    combined_rows.append({'type': 'header', 'protein': pname})
    for _, r in g.iterrows():
        combined_rows.append({'type': 'data', **r.to_dict()})

n_rows = len(combined_rows)
y_positions = np.arange(n_rows)[::-1]
for row, y in zip(combined_rows, y_positions):
    row['y'] = y

fig = plt.figure(figsize=(11.5, 0.4 * n_rows + 1.5))
fig.patch.set_facecolor(SURFACE)
gs = fig.add_gridspec(1, 2, width_ratios=[6, 1], wspace=0.035)
ax  = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharey=ax)
for a in (ax, ax2):
    a.set_facecolor(SURFACE)

# alternating band per protein group, spanning header + its data rows
band_on = False
for pname, _ in grouped:
    group_ys = [r['y'] for r in combined_rows if r.get('protein') == pname]
    if band_on:
        for a in (ax, ax2):
            a.axhspan(min(group_ys) - 0.5, max(group_ys) + 0.5, color=BAND_COLOR, zorder=0, lw=0)
    band_on = not band_on

legend_handles = {}
tick_positions, tick_labels_list, tick_colors, tick_weights, tick_sizes = [], [], [], [], []
for row in combined_rows:
    y = row['y']
    if row['type'] == 'header':
        tick_positions.append(y)
        tick_labels_list.append(PROTEIN_NAMES.get(row['protein'], row['protein']))
        tick_colors.append(INK_PRIMARY)
        tick_weights.append('semibold')
        tick_sizes.append(16)
        continue

    rank, knee = row['accuracy_rank'], row['knee']
    ref = row.get('citation_ref')
    ref_tag = f"  [{int(ref)}]" if pd.notna(ref) else ""
    tick_positions.append(y)
    tick_labels_list.append(f"    {_tidy_state(row['protein'], row['state'])}{ref_tag}")
    tick_colors.append(INK_PRIMARY)
    tick_weights.append('normal')
    tick_sizes.append(14)

    if pd.isna(rank):
        h = ax2.scatter(0, y, s=100, color=CRITICAL, marker='x', zorder=3, linewidth=2.2)
        legend_handles['never'] = h
    elif rank <= knee:
        h = ax.scatter(rank, y, s=105, color=STATUS_GOOD, marker='o', zorder=3,
                        edgecolor=SURFACE, linewidth=1.3)
        legend_handles['within'] = h
    else:
        h = ax.scatter(rank, y, s=105, color=SERIES_BLUE, marker='o', zorder=3,
                        edgecolor=SURFACE, linewidth=1.3)
        legend_handles['beyond'] = h

ax.set_yticks(tick_positions)
ax.set_yticklabels(tick_labels_list)
for tick, color, weight, size in zip(ax.get_yticklabels(), tick_colors, tick_weights, tick_sizes):
    tick.set_color(color)
    tick.set_fontweight(weight)
    tick.set_fontsize(size)
ax.tick_params(axis='y', length=0, pad=8)

ax.set_xscale('log')
ax.set_xlim(0.7, max(1000, pocket_recovery_df['accuracy_rank'].max(skipna=True) * 1.5 if pocket_recovery_df['accuracy_rank'].notna().any() else 1000))
ax.set_ylim(-0.5, n_rows - 0.5)
ax.set_xlabel('Recovery rank (log scale)', fontsize=18, color=INK_PRIMARY)
ax.tick_params(axis='x', labelsize=14, colors=INK_PRIMARY)
for spine in ('top', 'right', 'left'):
    ax.spines[spine].set_visible(False)
ax.spines['bottom'].set_color(BASELINE_GRAY)

ax2.set_xlim(-1, 1)
ax2.set_xticks([0])
ax2.set_xticklabels(['not\nrecovered'], fontsize=13, color=INK_PRIMARY)
ax2.tick_params(axis='x', length=0)
ax2.tick_params(axis='y', length=0, labelleft=False)
for spine in ('top', 'right', 'left', 'bottom'):
    ax2.spines[spine].set_visible(False)
ax2.axvline(-1, color=BASELINE_GRAY, lw=0.8, ymin=0.02, ymax=0.98)   # subtle divider marking the axis break

order = ['within', 'beyond', 'never']
label_text = {'within': 'Recovered within knee cutoff', 'beyond': 'Recovered, below knee cutoff', 'never': 'Never recovered'}
handles = [legend_handles[k] for k in order if k in legend_handles]
labs = [label_text[k] for k in order if k in legend_handles]

fig.legend(handles, labs, loc='upper center', ncol=3, frameon=False, fontsize=16, bbox_to_anchor=(0.57, 0.995))

fig.subplots_adjust(left=0.32, right=0.98, top=0.97, bottom=0.06)
fig.savefig('figures/pocket_recovery_vs_literature.pdf', dpi=300, facecolor=SURFACE, bbox_inches='tight')
plt.show()
