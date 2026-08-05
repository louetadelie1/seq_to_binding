"""
Overlay the mean enrichment curve + enrichment factor from EVERY ablation
condition trained so far on one pair of publication figures. Superset of
plot_combined_conditions.py (which only covers the original 4 train/inference
-distance conditions) -- this one also pulls in exp_5's and exp_6's
permutation-invariant variants and all of exp_7's Boltz+Jacobian combinations
(full, lean mean_c-only, and permutation-invariant).

  exp_2_negative_data/figures/summary_data.pkl                                    -- MD C-alpha distances
  exp_3_no_dist/figures/summary_data.pkl                                          -- No distances (sequence only)
  exp_5_boltz_distances_inf_pred/figures/summary_data.pkl                         -- Boltz C-alpha distances
  exp_5_boltz_distances_inf_pred/permutations/figures/summary_data_permutations.pkl -- Boltz distances, permutation-invariant encoder
  exp_6_esm_jacobian_contacts_no_c_alpha/figures/summary_data.pkl                 -- ESM-2 categorical-Jacobian coupling only
  exp_6_esm_jacobian_contacts_no_c_alpha/permuations/figures/summary_data_permutations.pkl -- Jacobian coupling, permutation-invariant encoder
  exp_7_botz_jacobian_combined/figures/summary_data.pkl                          -- Boltz distances + Jacobian coupling, combined
  exp_7_botz_jacobian_combined/figures_lean/summary_data_lean_coupling.pkl       -- Boltz distances + lean Jacobian coupling (mean_c only)
  exp_7_botz_jacobian_combined/permuations/figures/summary_data_permutations.pkl -- Boltz + Jacobian coupling, permutation-invariant encoder

Each summary_data*.pkl is written by that condition's own
predict_multiple_systems_average*.py script (see the 'cache raw per-system
arrays' block at the end of each). Any condition whose predict script hasn't
been (re)run yet is skipped with a printed warning rather than crashing --
rerun that condition's predict script first if you want it included.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plot_style as ps

BASE = '/ptmp/adlouet/camb/sequence_to_binding_paths/fine_tuning_experiments'

# (path to summary_data*.pkl relative to BASE, display label)
SOURCES = [
    ('exp_2_negative_data/figures/summary_data.pkl',
     'MD C-alpha distances'),
    ('exp_3_no_dist/figures/summary_data.pkl',
     'No distances (sequence only)'),
    ('exp_5_boltz_distances_inf_pred/figures/summary_data.pkl',
     'Boltz C-alpha distances'),
    ('exp_6_esm_jacobian_contacts_no_c_alpha/figures/summary_data.pkl',
     'Jacobian coupling only'),
    ('exp_7_botz_jacobian_combined/figures/summary_data.pkl',
     'Boltz + Jacobian coupling (combined)'),
    # ('exp_7_botz_jacobian_combined/figures_lean/summary_data_lean_coupling.pkl',
    #  'Boltz + Jacobian coupling (lean, mean_c only)'),
]

results = []
for rel_path, label in SOURCES:
    path = f'{BASE}/{rel_path}'
    if not os.path.exists(path):
        print(f"skipping {label!r}: {path} not found yet "
              f"(run that condition's predict_multiple_systems_average*.py first)")
        continue
    with open(path, 'rb') as f:
        d = pickle.load(f)
    d['label'] = label  # use our label regardless of what was cached
    results.append(d)
    print(f"{label:48s} n={len(d['systems_used']):2d} systems: {d['systems_used']}")

if not results:
    raise SystemExit("No summary_data*.pkl files found for any condition -- "
                      "run at least one predict_multiple_systems_average*.py first.")

ps.set_publication_theme(font_scale=1.3)
MUTED = ps.MUTED
# dynamic brand-rotation palette, sized to however many conditions were actually
# found. categorical_blues (not blues): these color DISTINCT conditions, not a
# magnitude -- blues() is the hydro-blue sequential ramp for that job; this is
# the validated AZURE/GREEN/VIOLET/AMBER rotation instead (see plot_style.py).
COLORS = ps.categorical_blues(len(results))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2))

# PANEL 1 -- mean enrichment curve +/- 1 std across systems, one line per condition
ax1.plot([0, 1], [0, 1], color=MUTED, lw=1.0, ls='--', zorder=1, label='Random')
for d, color in zip(results, COLORS):
    X_GRID = d['X_GRID']
    pred_curves = np.stack(d['pred_curves'])
    pred_mean, pred_std = pred_curves.mean(0), pred_curves.std(0)
    auc_per_system = np.trapezoid(pred_curves, X_GRID, axis=1)
    auc_mean, auc_std = auc_per_system.mean(), auc_per_system.std()

    ax1.fill_between(X_GRID, np.clip(pred_mean - pred_std, 0, 1), np.clip(pred_mean + pred_std, 0, 1),
                      color=ps.BAND_COLOR, alpha=0.2, linewidth=0, zorder=2)
    ax1.plot(X_GRID, pred_mean, color=color, lw=2.0, zorder=3,
              label=f"{d['label']} (AUC={auc_mean:.2f}±{auc_std:.2f})")

ax1.set_xlim(0, 1); ax1.set_ylim(0, 1); ax1.set_aspect('equal')
ax1.set_xlabel('Fraction of ranked list screened', fontsize=15)
ax1.set_ylabel(f"Fraction of top-{results[0]['top_actives']} actives recovered", fontsize=15)
ax1.set_title('AUC Curve across systems', fontsize=15)
ax1.tick_params(axis='both', labelsize=13)
# ax1.legend(loc='lower right', frameon=False, fontsize=11)

# PANEL 2 -- mean enrichment factor +/- 1 std across systems, grouped bars per condition
ef_fractions = results[0]['ef_fractions']
n_conditions = len(results)
w = 0.8 / n_conditions
x_pos = np.arange(len(ef_fractions))
ax2.axhline(1, color=MUTED, ls='--', lw=1.0, zorder=1)
for i, (d, color) in enumerate(zip(results, COLORS)):
    ef_pred_rows = np.stack(d['ef_pred_rows'])
    ef_mean, ef_std = ef_pred_rows.mean(0), ef_pred_rows.std(0)
    offset = (i - (n_conditions - 1) / 2) * w
    ax2.bar(x_pos + offset, ef_mean, width=w, color=color, edgecolor='white', linewidth=0.6,
            yerr=ef_std, error_kw={'ecolor': 'black', 'elinewidth': 1.0, 'capthick': 1.0},
            capsize=2.5, label=d['label'], zorder=2)

ax2.set_xticks(x_pos); ax2.set_xticklabels([f'{f*100:.0f}%' for f in ef_fractions])
ax2.set_xlabel('Top fraction of ranked list screened', fontsize=15)
ax2.set_ylabel('Enrichment factor', fontsize=15)
ax2.set_title('Enrichment Factors across systems', fontsize=15)
ax2.tick_params(axis='both', labelsize=13)
ax2.legend(frameon=False, fontsize=11, ncol=1)

sns.despine(fig=fig)
fig.tight_layout()

out_dir = f'{BASE}/combined_ablation_figures'
os.makedirs(out_dir, exist_ok=True)
out_path = f'{out_dir}/ef_auc_all_ablations_combined.pdf'
fig.savefig(out_path, bbox_inches='tight')
fig.savefig(out_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
print(f"\nSaved combined figure ({len(results)} conditions) to {out_path}")

plt.show()
