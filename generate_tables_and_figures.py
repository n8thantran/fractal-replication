"""
Generate all tables and figures for the paper replication.
Tables A.5-A.8: Real-world ARI scores per dataset
Tables A.1-A.4: Synthetic ARI scores (averaged)
Tables 2-5: Aggregate statistics (% wins, avg win/loss %)
Table A.9: Wilcoxon signed-rank test
Figures 2-5: Boxplots for synthetic data
"""

import json
import os
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Load results
with open('results/real_world_results.json') as f:
    rw_results = json.load(f)

with open('results/synthetic_results.json') as f:
    syn_results = json.load(f)

with open('results/synthetic_raw_results.json') as f:
    syn_raw = json.load(f)

os.makedirs('results/tables', exist_ok=True)
os.makedirs('results/figures', exist_ok=True)

# Dataset order from paper
DATASETS = [
    'Breast tissue', 'Breast Wisconsin', 'Ecoli', 'Glass', 'Haberman',
    'Ionosphere', 'Iris', 'Movement libras', 'Musk', 'Parkinsons',
    'Segmentation', 'Sonar all', 'Spectf', 'Transfusion', 'Vehicle',
    'Vertebral column', 'Vowel context', 'Wine', 'Wine quality red', 'Yeast'
]

DR_METHODS = ['PCA', 'Kernel PCA', 'VAE', 'Isomap', 'MDS']
DR_LEVELS = ['k-1', '25%', '50%']
CLUSTERING = ['k-means', 'AHC', 'GMM', 'OPTICS']
SYNTHETIC_TYPES = ['Circles', 'Moons', 'RSG', 'Repliclust']

def get_dr_key(method, level):
    return f"{method}_{level}"

# ============================================================
# Tables A.5-A.8: Real-world ARI scores
# ============================================================
def generate_real_world_table(clust_method, table_label):
    """Generate CSV table for a clustering method on real-world data."""
    data = rw_results[clust_method]
    
    header = ['Dataset', 'No Reduction']
    for dr in DR_METHODS:
        for level in DR_LEVELS:
            header.append(f"{dr}_{level}")
    
    rows = []
    for ds in DATASETS:
        if ds not in data:
            continue
        row = [ds]
        nr = data[ds].get('No Reduction', 0)
        row.append(f"{nr:.2f}")
        for dr in DR_METHODS:
            for level in DR_LEVELS:
                key = get_dr_key(dr, level)
                val = data[ds].get(key, 0)
                row.append(f"{val:.2f}")
        rows.append(row)
    
    # Write CSV
    fname = f"results/tables/table_{table_label}_real_{clust_method.replace(' ', '_').replace('-', '')}.csv"
    with open(fname, 'w') as f:
        f.write(','.join(header) + '\n')
        for row in rows:
            f.write(','.join(row) + '\n')
    
    return fname

# Generate Tables A.5-A.8
table_labels = {'k-means': 'A5', 'AHC': 'A6', 'GMM': 'A7', 'OPTICS': 'A8'}
for clust in CLUSTERING:
    fname = generate_real_world_table(clust, table_labels[clust])
    print(f"Generated {fname}")

# ============================================================
# Tables A.1-A.4: Synthetic ARI scores (averaged)
# ============================================================
def generate_synthetic_table(syn_type, table_label):
    """Generate CSV table for a synthetic dataset type."""
    data = syn_results[syn_type]
    
    header = ['Algorithm', 'No Reduction']
    for dr in DR_METHODS:
        for level in DR_LEVELS:
            header.append(f"{dr}_{level}")
    
    rows = []
    for clust in CLUSTERING:
        if clust not in data:
            continue
        row = [clust]
        nr = data[clust].get('No Reduction', 0)
        row.append(f"{nr:.3f}")
        for dr in DR_METHODS:
            for level in DR_LEVELS:
                key = get_dr_key(dr, level)
                val = data[clust].get(key, 0)
                row.append(f"{val:.3f}")
        rows.append(row)
    
    fname = f"results/tables/table_{table_label}_synthetic_{syn_type}.csv"
    with open(fname, 'w') as f:
        f.write(','.join(header) + '\n')
        for row in rows:
            f.write(','.join(row) + '\n')
    
    return fname

syn_labels = {'Circles': 'A1', 'Moons': 'A2', 'RSG': 'A3', 'Repliclust': 'A4'}
for syn_type in SYNTHETIC_TYPES:
    if syn_type in syn_results:
        fname = generate_synthetic_table(syn_type, syn_labels[syn_type])
        print(f"Generated {fname}")

# ============================================================
# Tables 2-5: Aggregate statistics
# ============================================================
def compute_aggregate_stats_real(clust_method):
    """Compute % wins and avg win/loss % for real-world data."""
    data = rw_results[clust_method]
    
    results = {}
    for dr in DR_METHODS:
        for level in DR_LEVELS:
            key = get_dr_key(dr, level)
            wins = 0
            total = 0
            pct_changes = []
            
            for ds in DATASETS:
                if ds not in data:
                    continue
                nr = data[ds].get('No Reduction', 0)
                val = data[ds].get(key, None)
                if val is None:
                    continue
                
                total += 1
                if val > nr + 0.005:  # Small tolerance for "win"
                    wins += 1
                
                # Compute percentage change
                if abs(nr) > 0.001:
                    pct_change = ((val - nr) / abs(nr)) * 100
                else:
                    pct_change = (val - nr) * 100
                pct_changes.append(pct_change)
            
            win_pct = (wins / total * 100) if total > 0 else 0
            avg_pct = np.mean(pct_changes) if pct_changes else 0
            
            results[(dr, level)] = {
                'win_pct': win_pct,
                'avg_pct': avg_pct,
                'n_datasets': total
            }
    
    return results

def compute_aggregate_stats_synthetic(clust_method):
    """Compute % wins and avg win/loss % for synthetic data.
    
    syn_raw structure: {syn_type: {clust_method: {dr_key: [values per repeat]}}}
    """
    results = {}
    for dr in DR_METHODS:
        for level in DR_LEVELS:
            key = get_dr_key(dr, level)
            wins = 0
            total = 0
            pct_changes = []
            
            for syn_type in SYNTHETIC_TYPES:
                if syn_type not in syn_raw:
                    continue
                if clust_method not in syn_raw[syn_type]:
                    continue
                
                clust_data = syn_raw[syn_type][clust_method]
                nr_vals = clust_data.get('No Reduction', [])
                dr_vals = clust_data.get(key, [])
                
                if not nr_vals or not dr_vals:
                    continue
                
                # Compare per repeat
                for i in range(min(len(nr_vals), len(dr_vals))):
                    run_nr = nr_vals[i]
                    run_val = dr_vals[i]
                    total += 1
                    if run_val > run_nr + 0.005:
                        wins += 1
                    if abs(run_nr) > 0.001:
                        pct_change = ((run_val - run_nr) / abs(run_nr)) * 100
                    else:
                        pct_change = (run_val - run_nr) * 100
                    pct_changes.append(pct_change)
            
            win_pct = (wins / total * 100) if total > 0 else 0
            avg_pct = np.mean(pct_changes) if pct_changes else 0
            
            results[(dr, level)] = {
                'win_pct': win_pct,
                'avg_pct': avg_pct,
                'n_datasets': total
            }
    
    return results

def generate_aggregate_table(clust_method, table_num):
    """Generate aggregate table for a clustering method."""
    real_stats = compute_aggregate_stats_real(clust_method)
    syn_stats = compute_aggregate_stats_synthetic(clust_method)
    
    fname = f"results/tables/table_{table_num}_aggregate_{clust_method.replace(' ', '_').replace('-', '')}.csv"
    
    with open(fname, 'w') as f:
        f.write("Method,Reduction,Win% Synthetic,Win% Real,Avg% Synthetic,Avg% Real\n")
        for dr in DR_METHODS:
            for level in DR_LEVELS:
                s = syn_stats.get((dr, level), {'win_pct': 0, 'avg_pct': 0})
                r = real_stats.get((dr, level), {'win_pct': 0, 'avg_pct': 0})
                f.write(f"{dr},{level},{s['win_pct']:.2f},{r['win_pct']:.2f},{s['avg_pct']:.2f},{r['avg_pct']:.2f}\n")
    
    # Also write formatted text
    fname_txt = fname.replace('.csv', '.txt')
    with open(fname_txt, 'w') as f:
        f.write(f"Aggregate statistics for {clust_method}\n")
        f.write(f"{'Method':<15} {'Reduction':<10} {'Win% Syn':>10} {'Win% Real':>10} {'Avg% Syn':>10} {'Avg% Real':>10}\n")
        f.write("-" * 70 + "\n")
        for dr in DR_METHODS:
            for level in DR_LEVELS:
                s = syn_stats.get((dr, level), {'win_pct': 0, 'avg_pct': 0})
                r = real_stats.get((dr, level), {'win_pct': 0, 'avg_pct': 0})
                f.write(f"{dr:<15} {level:<10} {s['win_pct']:>10.2f} {r['win_pct']:>10.2f} {s['avg_pct']:>10.2f} {r['avg_pct']:>10.2f}\n")
    
    return fname

agg_labels = {'k-means': '2', 'AHC': '3', 'GMM': '4', 'OPTICS': '5'}
for clust in CLUSTERING:
    fname = generate_aggregate_table(clust, agg_labels[clust])
    print(f"Generated {fname}")

# ============================================================
# Table A.9: Wilcoxon signed-rank test
# ============================================================
def compute_wilcoxon_tests():
    """Compute Wilcoxon signed-rank test for real-world data."""
    results = {}
    
    for clust in CLUSTERING:
        data = rw_results[clust]
        results[clust] = {}
        
        for dr in DR_METHODS:
            for level in DR_LEVELS:
                key = get_dr_key(dr, level)
                baseline_scores = []
                dr_scores = []
                
                for ds in DATASETS:
                    if ds not in data:
                        continue
                    nr = data[ds].get('No Reduction', 0)
                    val = data[ds].get(key, None)
                    if val is not None:
                        baseline_scores.append(nr)
                        dr_scores.append(val)
                
                baseline_scores = np.array(baseline_scores)
                dr_scores = np.array(dr_scores)
                diffs = dr_scores - baseline_scores
                
                # Remove zeros (ties)
                nonzero = np.abs(diffs) > 1e-10
                if np.sum(nonzero) < 2:
                    p_value = 1.0
                else:
                    try:
                        stat, p_value = stats.wilcoxon(diffs[nonzero], alternative='greater')
                    except:
                        p_value = 1.0
                
                results[clust][(dr, level)] = p_value
    
    return results

wilcoxon_results = compute_wilcoxon_tests()

fname = "results/tables/table_A9_wilcoxon.csv"
with open(fname, 'w') as f:
    header = ['Algorithm']
    for dr in DR_METHODS:
        for level in DR_LEVELS:
            header.append(f"{dr}_{level}")
    f.write(','.join(header) + '\n')
    
    for clust in CLUSTERING:
        row = [clust]
        for dr in DR_METHODS:
            for level in DR_LEVELS:
                p = wilcoxon_results[clust].get((dr, level), 1.0)
                row.append(f"{p:.3f}")
        f.write(','.join(row) + '\n')

# Also write formatted text
fname_txt = fname.replace('.csv', '.txt')
with open(fname_txt, 'w') as f:
    f.write("Wilcoxon signed-rank test results on 20 real-world benchmarks\n")
    f.write("One-sided test: H1: ARI_method > ARI_baseline, alpha=0.05, n=20\n")
    f.write("Significant p-values (< 0.05) marked with *\n\n")
    
    f.write(f"{'Algorithm':<12}")
    for dr in DR_METHODS:
        for level in DR_LEVELS:
            col = f"{dr}_{level}"
            f.write(f" {col:>14}")
    f.write("\n")
    f.write("-" * 230 + "\n")
    
    for clust in CLUSTERING:
        f.write(f"{clust:<12}")
        for dr in DR_METHODS:
            for level in DR_LEVELS:
                p = wilcoxon_results[clust].get((dr, level), 1.0)
                marker = "*" if p < 0.05 else " "
                col = f"{dr}_{level}"
                f.write(f" {p:>13.3f}{marker}")
        f.write("\n")

print(f"Generated {fname}")

# ============================================================
# Figures 2-5: Boxplots for synthetic data
# ============================================================
def generate_boxplot(clust_method, fig_num):
    """Generate boxplot for a clustering method on synthetic data."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
    fig.suptitle(f'ARI scores for {clust_method} on synthetic datasets', fontsize=14, fontweight='bold')
    
    colors = {
        'No Reduction': '#808080',
        'PCA': '#1f77b4',
        'Kernel PCA': '#ff7f0e',
        'VAE': '#2ca02c',
        'Isomap': '#d62728',
        'MDS': '#9467bd'
    }
    
    for idx, syn_type in enumerate(SYNTHETIC_TYPES):
        ax = axes[idx]
        
        if syn_type not in syn_raw or clust_method not in syn_raw[syn_type]:
            ax.set_title(syn_type)
            continue
        
        clust_data = syn_raw[syn_type][clust_method]
        
        # Collect data for boxplot
        box_data = []
        labels = []
        box_colors = []
        
        # No Reduction
        nr_vals = clust_data.get('No Reduction', [])
        if nr_vals:
            box_data.append(nr_vals)
            labels.append('NR')
            box_colors.append(colors['No Reduction'])
        
        # DR methods at each level
        for dr in DR_METHODS:
            for level in DR_LEVELS:
                key = get_dr_key(dr, level)
                vals = clust_data.get(key, [])
                if vals:
                    box_data.append(vals)
                    short_dr = dr[:3] if dr != 'Kernel PCA' else 'KPC'
                    short_level = level.replace('%', '').replace('k-1', 'k1')
                    labels.append(f"{short_dr}_{short_level}")
                    box_colors.append(colors[dr])
        
        if box_data:
            bp = ax.boxplot(box_data, patch_artist=True, widths=0.6,
                          medianprops=dict(color='black', linewidth=1.5))
            for patch, color in zip(bp['boxes'], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_xticklabels(labels, rotation=90, fontsize=6)
        
        ax.set_title(syn_type, fontsize=12)
        if idx == 0:
            ax.set_ylabel('ARI', fontsize=12)
        ax.grid(True, alpha=0.3)
    
    # Legend
    legend_patches = [mpatches.Patch(color=colors[m], label=m, alpha=0.7) 
                     for m in ['No Reduction'] + DR_METHODS]
    fig.legend(handles=legend_patches, loc='lower center', ncol=6, fontsize=9,
              bbox_to_anchor=(0.5, -0.05))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    fname = f"results/figures/figure_{fig_num}_boxplot_{clust_method.replace(' ', '_').replace('-', '')}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    return fname

fig_labels = {'k-means': '2', 'AHC': '3', 'GMM': '4', 'OPTICS': '5'}
for clust in CLUSTERING:
    fname = generate_boxplot(clust, fig_labels[clust])
    print(f"Generated {fname}")

# ============================================================
# Generate heatmap of real-world results
# ============================================================
def generate_heatmap(clust_method, fig_suffix):
    """Generate heatmap of ARI scores for real-world data."""
    data = rw_results[clust_method]
    
    # Build matrix
    all_keys = ['No Reduction']
    for dr in DR_METHODS:
        for level in DR_LEVELS:
            all_keys.append(get_dr_key(dr, level))
    
    matrix = []
    ds_labels = []
    for ds in DATASETS:
        if ds not in data:
            continue
        row = []
        for key in all_keys:
            row.append(data[ds].get(key, 0))
        matrix.append(row)
        ds_labels.append(ds)
    
    matrix = np.array(matrix)
    
    fig, ax = plt.subplots(figsize=(18, 10))
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=-0.2, vmax=1.0)
    
    ax.set_xticks(range(len(all_keys)))
    ax.set_xticklabels(all_keys, rotation=90, fontsize=7)
    ax.set_yticks(range(len(ds_labels)))
    ax.set_yticklabels(ds_labels, fontsize=8)
    
    plt.colorbar(im, ax=ax, label='ARI')
    ax.set_title(f'ARI Heatmap: {clust_method} on Real-World Datasets', fontsize=14)
    
    plt.tight_layout()
    fname = f"results/figures/heatmap_{fig_suffix}_{clust_method.replace(' ', '_').replace('-', '')}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    return fname

for clust in CLUSTERING:
    fname = generate_heatmap(clust, 'real')
    print(f"Generated {fname}")

# ============================================================
# Print comparison with paper values for real-world k-means
# ============================================================
print("\n" + "=" * 80)
print("COMPARISON: Our k-means results vs Paper Table A.5 (selected datasets)")
print("=" * 80)

paper_kmeans = {
    'Iris': {'No Reduction': 0.62, 'PCA_k-1': 0.62, 'PCA_25%': 0.80, 'Kernel PCA_k-1': 0.60},
    'Wine': {'No Reduction': 0.90, 'PCA_k-1': 0.90, 'Kernel PCA_k-1': 0.77},
    'Breast Wisconsin': {'No Reduction': 0.67, 'PCA_k-1': 0.66, 'Isomap_k-1': 0.72},
    'Ecoli': {'No Reduction': 0.51, 'PCA_k-1': 0.51, 'Isomap_50%': 0.71},
}

for ds in paper_kmeans:
    print(f"\n{ds}:")
    our_data = rw_results['k-means'].get(ds, {})
    for key, paper_val in paper_kmeans[ds].items():
        our_val = our_data.get(key, 'N/A')
        if isinstance(our_val, (int, float)):
            match = "✓" if abs(our_val - paper_val) < 0.05 else "~" if abs(our_val - paper_val) < 0.15 else "✗"
            print(f"  {key}: Paper={paper_val:.2f}, Ours={our_val:.2f}, {match}")
        else:
            print(f"  {key}: Paper={paper_val:.2f}, Ours={our_val}")

# ============================================================
# Print summary of Wilcoxon test
# ============================================================
print("\n" + "=" * 80)
print("WILCOXON TEST RESULTS (p-values, * = significant at 0.05)")
print("=" * 80)
for clust in CLUSTERING:
    print(f"\n{clust}:")
    for dr in DR_METHODS:
        vals = []
        for level in DR_LEVELS:
            p = wilcoxon_results[clust].get((dr, level), 1.0)
            marker = "*" if p < 0.05 else ""
            vals.append(f"{p:.3f}{marker}")
        print(f"  {dr:<15}: k-1={vals[0]}, 25%={vals[1]}, 50%={vals[2]}")

print("\nPaper's significant results: OPTICS + Kernel PCA at k-1 (0.047) and 25% (0.046)")

print("\n\nAll tables and figures generated successfully!")
print(f"Tables: {sorted(os.listdir('results/tables'))}")
print(f"Figures: {sorted(os.listdir('results/figures'))}")
