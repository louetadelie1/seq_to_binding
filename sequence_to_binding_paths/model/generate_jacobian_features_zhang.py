"""
Exact port of the categorical-Jacobian recipe from:
  https://github.com/zzhangzzhang/pLMs-interpretability/tree/main/jac
  (02_get_jac_batch.py's get_categorical_jacobian, plus utils.py's
  get_contacts/do_apc), ported from Meta's fair-esm API to the HuggingFace
  `transformers` ESM-2 checkpoint already used throughout this codebase.

This is deliberately NOT the same recipe as generate_jacobian_features.py.
Three real differences from that from-first-principles version:
  1. Raw logits are differenced directly -- no log_softmax.
  2. Centering is global (sequentially over all 4 tensor axes, i.e. across
     the whole protein at once), not local to each (i,j) 20x20 block.
  3. Symmetrization pairs (position, amino-acid) axes together on the full
     4D tensor -- (i,a,j,b) averaged with (j,b,i,a) -- before collapsing to
     a scalar, rather than transposing the already-collapsed 2D matrix.

The vocabulary index range 4:24 for the 20 canonical amino acids is
identical between fair-esm's alphabet and HuggingFace's ESM-2 tokenizer
(same vocab across all esm2_t*_UR50D checkpoint sizes, confirmed directly),
so this port reproduces the reference computation exactly, not
approximately.

MODEL_NAME below picks which frozen ESM-2 checkpoint the Jacobian is
computed from -- larger checkpoints give a markedly stronger/less noisy
coevolutionary coupling signal (see Zhang et al.), at a roughly
parameter-count-scaled compute cost per protein.
"""

from transformers import AutoTokenizer, EsmForMaskedLM
import torch
import sys
import glob
import os
import pickle
import linecache
import numpy as np
from tqdm.auto import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
MAX_LENGTH = 1024

base = '//ptmp/adlouet/camb/sequence_to_binding_paths/post_processed_probes'

tokeniser = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = EsmForMaskedLM.from_pretrained(MODEL_NAME).to(device).eval()

# HuggingFace's ESM-2 tokenizer places the 20 canonical amino acids at ids
# 4-23, in the same fixed order as fair-esm's alphabet -- so torch.arange(4,
# 24) below is exactly the reference's [4:24] slice/mutation range, not a
# re-derivation.


def get_categorical_jacobian(seq: str) -> np.ndarray:
    """Direct port of the reference get_categorical_jacobian(seq)."""
    inputs = tokeniser(seq, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    x    = inputs["input_ids"].to(device)
    attn = inputs["attention_mask"].to(device)
    ln = x.shape[1] - 2   # actual (possibly truncated) length, not len(seq)

    with torch.no_grad():
        def f(x_, attn_):
            return model(input_ids=x_, attention_mask=attn_).logits[..., 1:(ln + 1), 4:24].cpu().numpy()

        fx = f(x, attn)[0]                                     # (ln, 20)

        x_tiled    = torch.tile(x, [20, 1]).to(device)          # (20, ln+2)
        attn_tiled = torch.tile(attn, [20, 1]).to(device)

        fx_h = np.zeros((ln, 20, ln, 20), dtype=np.float32)
        for n in range(ln):                                     # for each position
            x_h = torch.clone(x_tiled)
            x_h[:, n + 1] = torch.arange(4, 24, device=device)    # mutate to all 20 aa
            fx_h[n] = f(x_h, attn_tiled)

    return fx_h - fx


def _do_apc(x, rm=1):
    """Direct port of the reference do_apc(x, rm=1) (the rm==1 / standard-APC branch)."""
    x = np.copy(x)
    a1 = x.sum(0, keepdims=True)
    a2 = x.sum(1, keepdims=True)
    y = x - (a1 * a2) / x.sum()
    np.fill_diagonal(y, 0)
    return y


def get_contacts(x, symm=True, center=True, rm=1):
    """Direct port of the reference get_contacts(x, symm=True, center=True, rm=1)."""
    j = x.copy()
    if center:
        for i in range(4):
            j -= j.mean(i, keepdims=True)
    j_fn = np.sqrt(np.square(j).sum((1, 3)))
    np.fill_diagonal(j_fn, 0)
    j_fn_corrected = _do_apc(j_fn, rm=rm)
    if symm:
        j_fn_corrected = (j_fn_corrected + j_fn_corrected.T) / 2
    return j_fn_corrected


def categorical_jacobian_zhang(seq: str) -> np.ndarray:
    """Full reference pipeline: jac -> center -> paired-symmetrize -> get_contacts."""
    jac = get_categorical_jacobian(seq)
    for i in range(4):
        jac -= jac.mean(i, keepdims=True)
    jac = (jac + jac.transpose(2, 3, 0, 1)) / 2
    return get_contacts(jac)


def process_one(fasta: str) -> None:
    out_path = fasta.replace('sequence.fasta', 'jacobian_matrix_zhang.pkl')
    if os.path.exists(out_path):
        try:
            existing = pickle.load(open(out_path, 'rb'))
            existing_model = existing.get('model') if isinstance(existing, dict) else None
        except Exception:
            existing_model = None
        if existing_model == MODEL_NAME:
            print(f"skipping {fasta}: {out_path} already computed with {MODEL_NAME}")
            return
        # else: stale (missing model tag, or computed with a smaller/different
        # checkpoint) -- fall through and regenerate with MODEL_NAME.
    if not os.path.exists(fasta):
        print(f"skipping {fasta}: no sequence.fasta")
        return

    seq = linecache.getline(fasta, 2).strip()
    try:
        C = categorical_jacobian_zhang(seq)
    except Exception as e:
        print(f"skipping {fasta}: {e}")
        return

    with open(out_path, 'wb') as f:
        pickle.dump({'matrix': C, 'model': MODEL_NAME}, f)
    print(f"wrote {out_path}  (L={len(seq)}, model={MODEL_NAME})")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # array-job mode: process exactly the sequence.fasta paths given on
        # the command line (see submit_jacobian_array_zhang.sh, which passes
        # one path per SLURM array task drawn from jacobian_manifest.txt)
        for fasta in sys.argv[1:]:
            process_one(fasta)
    else:
        # standalone sweep mode: every probe under post_processed_probes
        fasta_files_all = sorted(glob.glob(f'{base}/*/sequence.fasta'))
        for fasta in tqdm(fasta_files_all):
            process_one(fasta)
