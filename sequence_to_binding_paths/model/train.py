"""
EXPERIMENT: Distances + Geometric Statistics
Adds pairwise distances + geometric statistics:
- d(i,j), d(j,k), d(i,k)
- min, max, mean distance
- compactness = min/max
Total input: 480*3 + 10
"""

"""
ESM2_t12_35M frozen. Labels are log(p_eq) normalised to [0,1] per protein.
Loss is MSELoss — replaces MarginRankingLoss so gradient magnitude reflects
actual p_eq differences rather than treating all pairs equally.
"""

from transformers import AutoModel, AutoTokenizer
import wandb
import torch
import glob
import pickle
import linecache
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from collections import OrderedDict
import random
from scipy.stats import spearmanr
import os
import random
from itertools import combinations, islice

print(torch.cuda.is_available())
print(torch.__version__)

os.environ['OPENBLAS_NUM_THREADS'] = '1'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

os.makedirs('figures', exist_ok=True)
os.makedirs('weights', exist_ok=True)
weights="/ptmp/adlouet/camb/sequence_to_binding_paths/fine_tuning_experiments/exp_5_boltz_distances_inf_pred/weights"
tokeniser = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
esm_model = AutoModel.from_pretrained("facebook/esm2_t12_35M_UR50D").to(device)

TOP_ACTIVES = 10


class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.leaky_relu = torch.nn.LeakyReLU()
        self.dropout = torch.nn.Dropout(0.5)
        self.layer1 = torch.nn.Linear(480 * 3 + 10, 512)  # +3 dist +4 stats
        self.ln1 = torch.nn.LayerNorm([512])
        self.layer2 = torch.nn.Linear(512, 128)
        self.ln2 = torch.nn.LayerNorm([128])
        self.layer3 = torch.nn.Linear(128, 32)
        self.ln3 = torch.nn.LayerNorm([32])
        self.regression_head = torch.nn.Linear(32, 1)

    def forward(self, x1, x2, x3, pos):
        # Noise augmentation (from exp7): add Gaussian noise during training
        if self.training:
            x1 = x1 + torch.randn_like(x1) * 0.01
            x2 = x2 + torch.randn_like(x2) * 0.01
            x3 = x3 + torch.randn_like(x3) * 0.01
        
        x = torch.cat([x1, x2, x3, pos], dim=-1)
        x = self.dropout(self.leaky_relu(self.ln1(self.layer1(x))))
        x = self.dropout(self.leaky_relu(self.ln2(self.layer2(x))))
        x = self.dropout(self.leaky_relu(self.ln3(self.layer3(x))))
        return self.regression_head(x)


model = Model().to(device)
learning_rate = 1e-4
loss_fn   = torch.nn.MSELoss()
optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=5)


def clean(input):
    return {tuple(int(i) for i in key): float(value[0]) for key, value in input.items()}


class TripletDataset(Dataset):

    def __init__(self, protein_paths, fasta_path, tokeniser, esm_model):
        self.samples = []

        for protein, fasta in zip(protein_paths, fasta_path):
            try:
                p_eq_dict = clean(pickle.load(open(protein, 'rb')))
                p_eq_dict = OrderedDict(sorted(p_eq_dict.items(), key=lambda item: item[1], reverse=True))

                triplets, P_eq = map(list, zip(*p_eq_dict.items()))

                seq = linecache.getline(fasta, 2).strip()
                print(protein)

                inputs = tokeniser(seq, return_tensors="pt", truncation=True, max_length=1024)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    output = esm_model(**inputs)
                embeddings = output.last_hidden_state.squeeze(0).detach().cpu()

                # Load C-alpha distance matrix
                ca_dist_file = protein.replace('p_eq_keys.pkl', 'ca_dist_matrix_boltz.pkl')
                if not os.path.exists(ca_dist_file):
                    print(f"  WARNING: No distance matrix for {protein}")
                    ca_dist_matrix = None
                else:
                    ca_dist_dict = pickle.load(open(ca_dist_file, 'rb'))
                    ca_dist_matrix = ca_dist_dict['matrix']

                n = len(triplets)
                n_emb = embeddings.shape[0]

                top_indices  = list(range(0, min(50, n)))
                mid_indices  = list(range(50, min(200, n), 5))
                tail_indices = list(range(200, n, max(1, n // 50))) if n > 200 else []
                indices      = sorted(set(top_indices + mid_indices + tail_indices))
                print(f"numbers for {protein}", len(top_indices), len(mid_indices), len(tail_indices))

                # log(p_eq) normalised to [0,1] within each protein's sampled set
                sampled_peq = np.array([P_eq[i] for i in indices], dtype=np.float64)
                log_peq     = np.log(sampled_peq + 1e-12)
                lo, hi      = log_peq.min(), log_peq.max()
                norm_labels = (log_peq - lo) / (hi - lo) if hi > lo else np.ones(len(indices))

                L = len(seq)
                for k, idx in enumerate(indices):
                    triplet= triplets[idx]
                    triplet_sorted = tuple(sorted(triplet))
                    if any(i >= L for i in triplet_sorted):
                        continue
                    label = float(norm_labels[k])
                    # embeddings[0] is the <cls> token, so residue i's embedding is at index i+1
                    x1  = embeddings[triplet_sorted[0] + 1].clone()
                    x2  = embeddings[triplet_sorted[1] + 1].clone()
                    x3  = embeddings[triplet_sorted[2] + 1].clone()
                    i, j, k = triplet_sorted

                    # Pairwise distances + geometric statistics
                    if ca_dist_matrix is not None:
                        dist_ij = float(ca_dist_matrix[i, j])
                        dist_jk = float(ca_dist_matrix[j, k])
                        dist_ik = float(ca_dist_matrix[i, k])

                        # Geometric statistics
                        min_dist = min(dist_ij, dist_jk, dist_ik)
                        max_dist = max(dist_ij, dist_jk, dist_ik)
                        mean_dist = (dist_ij + dist_jk + dist_ik) / 3
                        compactness = min_dist / max_dist if max_dist > 0 else 0.0
                    else:
                        dist_ij = dist_jk = dist_ik = 0.0
                        min_dist = max_dist = mean_dist = compactness = 0.0

                    pos = torch.tensor([
                        i/L, j/L, k/L,                       # Positional (3)
                        dist_ij, dist_jk, dist_ik,           # Pairwise (3)
                        min_dist, max_dist, mean_dist, compactness  # Stats (4)
                    ], dtype=torch.float32)
                    self.samples.append((x1, x2, x3, label, triplet_sorted, protein, pos))

                # Add 50 non-existent triplets with label -0.5 (below the [0,1] real range)
                existing_set = set(tuple(sorted(t)) for t in triplets)
                rng      = np.random.default_rng(seed=abs(hash(protein)) % (2**32))
                sampled  = 0
                attempts = 0
                while sampled < 50 and attempts < 2500:
                    t = tuple(sorted(rng.choice(L, size=3, replace=False).tolist()))
                    attempts += 1
                    if t in existing_set:
                        continue
                    i, j, k = t
                    x1 = embeddings[i + 1].clone()
                    x2 = embeddings[j + 1].clone()
                    x3 = embeddings[k + 1].clone()
                    if ca_dist_matrix is not None:
                        dist_ij     = float(ca_dist_matrix[i, j])
                        dist_jk     = float(ca_dist_matrix[j, k])
                        dist_ik     = float(ca_dist_matrix[i, k])
                        min_dist    = min(dist_ij, dist_jk, dist_ik)
                        max_dist    = max(dist_ij, dist_jk, dist_ik)
                        mean_dist   = (dist_ij + dist_jk + dist_ik) / 3
                        compactness = min_dist / max_dist if max_dist > 0 else 0.0
                    else:
                        dist_ij = dist_jk = dist_ik = 0.0
                        min_dist = max_dist = mean_dist = compactness = 0.0
                    pos = torch.tensor([
                        i/L, j/L, k/L,
                        dist_ij, dist_jk, dist_ik,
                        min_dist, max_dist, mean_dist, compactness
                    ], dtype=torch.float32)
                    self.samples.append((x1, x2, x3, -0.5, t, protein, pos))
                    sampled += 1

            except Exception as e:
                print(f"Skipping {protein}: {e}")
                continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x1, x2, x3, label, triplet, protein_id, pos = self.samples[idx]
        return (x1, x2, x3, torch.tensor(label, dtype=torch.float32), triplet, protein_id, pos)


base = '//ptmp/adlouet/camb/sequence_to_binding_paths/post_processed_probes'

all_proteins    = sorted(glob.glob(f'{base}/*/p_eq_keys.pkl'))
fasta_files_all = [p.replace('p_eq_keys.pkl', 'sequence.fasta') for p in all_proteins]
pairs           = [(p, f) for p, f in zip(all_proteins, fasta_files_all) if os.path.exists(f)]
protein_files, fasta_files = zip(*pairs)

random.seed(42)
indices = list(range(len(protein_files)))
random.shuffle(indices)
split     = int(0.8 * len(indices))
train_idx = indices[:split]
val_idx   = indices[split:]

train_proteins = [protein_files[i] for i in train_idx]
train_fastas   = [fasta_files[i]   for i in train_idx]
val_proteins   = [protein_files[i] for i in val_idx]
val_fastas     = [fasta_files[i]   for i in val_idx]

print(f"Train proteins: {len(train_proteins)}, Val proteins: {len(val_proteins)}")

train_dataset = TripletDataset(train_proteins, train_fastas, tokeniser, esm_model)
val_dataset   = TripletDataset(val_proteins,   val_fastas,   tokeniser, esm_model)
print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")


def collate_fn(batch):
    x1_batch         = torch.stack([item[0] for item in batch])
    x2_batch         = torch.stack([item[1] for item in batch])
    x3_batch         = torch.stack([item[2] for item in batch])
    labels_batch     = torch.stack([item[3] for item in batch])
    triplets_batch   = [item[4] for item in batch]
    protein_ids_batch= [item[5] for item in batch]
    pos_batch        = torch.stack([item[6] for item in batch])
    return x1_batch, x2_batch, x3_batch, labels_batch, triplets_batch, protein_ids_batch, pos_batch


class ProteinBatchSampler:
    def __init__(self, dataset, shuffle=True):
        protein_to_indices = {}
        for i, sample in enumerate(dataset.samples):
            pid = sample[5]
            protein_to_indices.setdefault(pid, []).append(i)
        self.batches = [idxs for idxs in protein_to_indices.values() if len(idxs) >= 2]
        self.shuffle = shuffle

    def __iter__(self):
        order = list(range(len(self.batches)))
        if self.shuffle:
            random.shuffle(order)
        for i in order:
            yield self.batches[i]

    def __len__(self):
        return len(self.batches)


train_dataloader = DataLoader(train_dataset, batch_sampler=ProteinBatchSampler(train_dataset, shuffle=True),  collate_fn=collate_fn)
val_dataloader   = DataLoader(val_dataset,   batch_sampler=ProteinBatchSampler(val_dataset,   shuffle=False), collate_fn=collate_fn)

num_epochs = 100

wandb.init(
    project="seq_to_paths",
    name="exp18_holy_trifecta_cls_offset_fix",
    config={
        "learning_rate":  learning_rate,
        "epochs":         num_epochs,
        "batch_size":     "per_protein",
        "n_train_proteins": len(train_proteins),
        "n_val_proteins":   len(val_proteins),
        "loss":           "MSE on log(p_eq) normalised per protein",
        "dropout":        0.5,
        "noise_augmentation": True,
        "sampling":       "hybrid (50 top + 50 mid + tail)",
        "weight_decay":   1e-2,
        "split":          "protein-level 80/20",
        "scheduler":      "ReduceLROnPlateau(factor=0.5, patience=5)",
    }
)

progress_bar      = tqdm(range(num_epochs * len(train_dataloader)))
train_losses      = []
val_spearmans     = []
best_val_spearman = -float('inf')
patience          = 10
epochs_no_improve = 0

for epoch in range(num_epochs):
    # ------------------------------------------------------------------ TRAIN
    model.train()
    total_train_loss = 0

    per_protein_train_preds  = {}
    per_protein_train_labels = {}

    for batch in train_dataloader:
        x1, x2, x3, labels, triplets, protein_ids, pos = batch
        x1, x2, x3, labels, pos = x1.to(device), x2.to(device), x3.to(device), labels.to(device), pos.to(device)

        regression_pred = model(x1, x2, x3, pos).squeeze()
        loss = loss_fn(regression_pred, labels)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        total_train_loss += loss.item()
        progress_bar.update(1)

        preds_np  = regression_pred.detach().cpu().numpy()
        labels_np = labels.cpu().numpy()
        for i, pid in enumerate(protein_ids):
            per_protein_train_preds.setdefault(pid, []).append(float(preds_np[i]))
            per_protein_train_labels.setdefault(pid, []).append(float(labels_np[i]))

    avg_train_loss = total_train_loss / len(train_dataloader)
    train_losses.append(avg_train_loss)

    per_protein_train_spearmans = []
    for pid in per_protein_train_preds:
        if len(per_protein_train_preds[pid]) >= 2:
            r, _ = spearmanr(per_protein_train_preds[pid], per_protein_train_labels[pid])
            if not np.isnan(r):
                per_protein_train_spearmans.append(r)
    train_spearman_r = float(np.mean(per_protein_train_spearmans)) if per_protein_train_spearmans else float('nan')

    # -------------------------------------------------------------------- VAL
    model.eval()
    per_protein_preds    = {}
    per_protein_labels   = {}
    per_protein_triplets = {}
    total_val_loss = 0

    with torch.no_grad():
        for batch in val_dataloader:
            x1, x2, x3, labels, triplets, protein_ids, pos = batch
            x1, x2, x3, labels, pos = x1.to(device), x2.to(device), x3.to(device), labels.to(device), pos.to(device)

            regression_pred  = model(x1, x2, x3, pos).squeeze()
            total_val_loss  += loss_fn(regression_pred, labels).item()

            preds_np  = regression_pred.cpu().numpy()
            labels_np = labels.cpu().numpy()
            for i, pid in enumerate(protein_ids):
                per_protein_preds.setdefault(pid, []).append(float(preds_np[i]))
                per_protein_labels.setdefault(pid, []).append(float(labels_np[i]))
                per_protein_triplets.setdefault(pid, []).append(triplets[i])

    avg_val_loss = total_val_loss / len(val_dataloader)
    scheduler.step(avg_val_loss)

    per_protein_spearmans = []
    per_protein_results   = {}
    for pid in per_protein_preds:
        if len(per_protein_preds[pid]) >= 2:
            r, _ = spearmanr(per_protein_preds[pid], per_protein_labels[pid])
            if not np.isnan(r):
                per_protein_spearmans.append(r)
                per_protein_results[pid] = {
                    'spearman': r,
                    'preds':    per_protein_preds[pid],
                    'labels':   per_protein_labels[pid],
                    'triplets': per_protein_triplets[pid],
                }
    spearman_r = float(np.mean(per_protein_spearmans)) if per_protein_spearmans else float('nan')
    val_spearmans.append(spearman_r)

    if spearman_r > best_val_spearman:
        best_val_spearman = spearman_r
        epochs_no_improve = 0
        torch.save(model.state_dict(), f'{weights}/best_model_change_ranking.pt')
    else:
        epochs_no_improve += 1

    auc_AB_total = []

    for pid, result in per_protein_results.items():
        pred_order = sorted(range(len(result['preds'])), key=lambda i: result['preds'][i], reverse=True)
        list_pred  = [result['triplets'][i] for i in pred_order]

        p_eq_dict_real = OrderedDict(sorted(clean(pickle.load(open(pid, 'rb'))).items(), key=lambda item: item[1], reverse=True))

        # Drop the injected negative/decoy triplets (label -0.5, not real p_eq
        # entries) - they aren't part of the real candidate pool being scanned,
        # and including them let frac_scan exceed 1 and inflated AUC past 1.0.
        list_pred = [t for t in list_pred if t in p_eq_dict_real]

        # This is the auc on teh given 150 samples
        n = len(p_eq_dict_real)

        top_indices  = list(range(0, min(50, n)))
        mid_indices  = list(range(50, min(200, n), 5))
        tail_indices = list(range(200, n, max(1, n // 50))) if n > 200 else []
        indices      = sorted(set(top_indices + mid_indices + tail_indices))

        all_triplets = list(p_eq_dict_real.keys())
        all_peqs     = list(p_eq_dict_real.values())
        sampled_triplets_real = [all_triplets[i] for i in indices]
        sampled_peq= np.array([all_peqs[i] for i in indices], dtype=np.float64)

        actives_real = set(all_triplets[:TOP_ACTIVES])
        N_A=len(indices)

        hits = 0
        frac_scan, frac_find = [0.0], [0.0]
        for n, triplet in enumerate(list_pred):
            if triplet in actives_real:
                hits += 1
            frac_scan.append((n + 1) / N_A)
            frac_find.append(hits / TOP_ACTIVES)
        auc_pred_to_true = np.trapezoid(frac_find, frac_scan)
        auc_AB_total.append((auc_pred_to_true))

    mean_true_to_pred = float(np.mean(auc_AB_total)) if auc_AB_total else float('nan')

    print(f"\nEpoch {epoch+1}/{num_epochs}")
    print(f"  Train Loss:      {avg_train_loss:.4f}")
    print(f"  Val Loss:        {avg_val_loss:.4f}")
    print(f"  Train Spearman:  {train_spearman_r:.4f}  (n={len(per_protein_train_spearmans)} proteins)")
    print(f"  Val Spearman:    {spearman_r:.4f}  (n={len(per_protein_spearmans)} proteins)")
    print(f"  AUC true→pred:  {mean_true_to_pred:.4f}  (scan true, find predicted top {TOP_ACTIVES})")

    wandb.log({
        "train_loss":       avg_train_loss,
        "val_loss":         avg_val_loss,
        "train_spearman_r": train_spearman_r,
        "val_spearman_r":   spearman_r,
        "best_val_spearman": best_val_spearman,
        "epoch":            epoch + 1,
        "auc_true_to_pred": mean_true_to_pred,
        "lr":               optimiser.param_groups[0]['lr'],
    })

    if epochs_no_improve >= patience:
        print(f"\nEarly stopping: no improvement for {patience} epochs. Best val Spearman: {best_val_spearman:.4f}")
        break

torch.save(model.state_dict(), f'{weights}/model_change_ranking.pt')
wandb.finish()
