"""
Generate all tables and figures for the paper replication.
Tables A.1-A.4: Synthetic data results
Tables A.5-A.8: Real-world data results  
Tables 2-5: Aggregate statistics
Table A.9: Wilcoxon signed-rank test
Figures: Boxplots for synthetic and real data
"""
import numpy as np
import json
import os
import csv
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = '/workspace/results'
TABLES_DIR = os.path.join(RESULTS_DIR, 'tables')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

DR_METHODS = ['PCA', 'Kernel PCA', 'VAE', 'Isomap', 'MDS']
LEVELS = ['k-1', '25%', '50%']
CLUSTERING = ['k-means', 'AHC', 'GMM', 'OPTICS']
SYNTH_TYPES = ['Circles', 'Moons', 'RSG', 'Repliclust']

DR_COLS = []
for dr in DR_METHODS:
    for lv in LEVELS:
        DR_COLS.append(f'{dr}_{lv}')

ALL_COLS = ['No Reduction'] + DR_COLS


def load_data():
    """Load real and synthetic results."""
    with open(os.path.join(RESULTS_DIR, 'real_world_results.json')) as f:
        raw_real = json.load(f)
    
    # Restructure: {dataset: {clustering: {col: val}}}
    real = {}
    for cn in raw_real:
        for ds in raw_real[cn]:
            if ds not in real:
                real[ds] = {}
            real[ds][cn] = raw_real[cn][ds]
    
    with open(os.path.join(RESULTS_DIR, 'synthetic_results_v3.json')) as f:
        synth = json.load(f)
    
    # Load raw synthetic for per-repeat stats
    synth_raw = None
    raw_path = os.path.join(RESULTS_DIR, 'synthetic_raw_v3.json')
    if os.path.exists(raw_path):
        with open(raw_path) as f:
            synth_raw = json.load(f)
    
    return real, synth, synth_raw


def fmt(v, decimals=2):
    if v is None:
        return '-'
    return f'{v:.{decimals}f}'


def generate_synthetic_tables(synth):
    """Tables A.1-A.4: One table per synthetic type."""
    table_map = {
        'Circles': 'A1', 'Moons': 'A2', 'RSG': 'A3', 'Repliclust': 'A4'
    }
    
    for dtype in SYNTH_TYPES:
        tnum = table_map[dtype]
        fname = f'table_{tnum}_synthetic_{dtype}'
        
        rows = []
        for cn in CLUSTERING:
            row = [cn]
            if cn in synth.get(dtype, {}):
                for col in ALL_COLS:
                    v = synth[dtype][cn].get(col)
                    row.append(fmt(v, 3))
            else:
                row.extend(['-'] * len(ALL_COLS))
            rows.append(row)
        
        # CSV
        with open(os.path.join(TABLES_DIR, fname + '.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['Algorithm'] + ALL_COLS)
            for row in rows:
                w.writerow(row)
        
        # TXT
        with open(os.path.join(TABLES_DIR, fname + '.txt'), 'w') as f:
            f.write(f'Table {tnum}: Average ARI for {dtype} synthetic data\n')
            f.write('=' * 200 + '\n')
            header = f'{"Algorithm":<12}' + ''.join(f'{c:>14}' for c in ALL_COLS)
            f.write(header + '\n')
            f.write('-' * 200 + '\n')
            for row in rows:
                line = f'{row[0]:<12}' + ''.join(f'{v:>14}' for v in row[1:])
                f.write(line + '\n')
        
        print(f'  Generated {fname}')


def generate_real_tables(real):
    """Tables A.5-A.8: One table per clustering algorithm."""
    table_map = {
        'k-means': ('A5', 'table_A5_real_kmeans'),
        'AHC': ('A6', 'table_A6_real_AHC'),
        'GMM': ('A7', 'table_A7_real_GMM'),
        'OPTICS': ('A8', 'table_A8_real_OPTICS'),
    }
    
    datasets = sorted(real.keys())
    
    for cn, (tnum, fname) in table_map.items():
        rows = []
        for ds in datasets:
            row = [ds]
            if cn in real[ds]:
                for col in ALL_COLS:
                    v = real[ds][cn].get(col)
                    row.append(fmt(v))
            else:
                row.extend(['-'] * len(ALL_COLS))
            rows.append(row)
        
        # Add mean row
        mean_row = ['Mean']
        for col in ALL_COLS:
            vals = []
            for ds in datasets:
                if cn in real[ds]:
                    v = real[ds][cn].get(col)
                    if v is not None:
                        vals.append(v)
            mean_row.append(fmt(np.mean(vals)) if vals else '-')
        rows.append(mean_row)
        
        # CSV
        with open(os.path.join(TABLES_DIR, fname + '.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['Dataset'] + ALL_COLS)
            for row in rows:
                w.writerow(row)
        
        # TXT
        with open(os.path.join(TABLES_DIR, fname + '.txt'), 'w') as f:
            f.write(f'Table {tnum}: ARI for {cn} on real-world datasets\n')
            f.write('=' * 260 + '\n')
            header = f'{"Dataset":<22}' + ''.join(f'{c:>14}' for c in ALL_COLS)
            f.write(header + '\n')
            f.write('-' * 260 + '\n')
            for row in rows:
                line = f'{row[0]:<22}' + ''.join(f'{v:>14}' for v in row[1:])
                f.write(line + '\n')
        
        print(f'  Generated {fname}')


def compute_aggregate(real, synth, synth_raw, cn):
    """
    Compute aggregate statistics for a clustering algorithm.
    For each DR method + level:
    - Win% Synthetic: % of synthetic repeats where DR improves over No Reduction
    - Win% Real: % of real datasets where DR improves over No Reduction
    - Avg% Synthetic: average % change in ARI vs No Reduction (synthetic)
    - Avg% Real: average % change in ARI vs No Reduction (real)
    """
    results = []
    datasets = sorted(real.keys())
    
    for dr in DR_METHODS:
        for lv in LEVELS:
            key = f'{dr}_{lv}'
            
            # Synthetic: use raw per-repeat data if available
            syn_wins = 0
            syn_total = 0
            syn_changes = []
            
            if synth_raw:
                for dtype in SYNTH_TYPES:
                    if dtype in synth_raw:
                        for repeat_data in synth_raw[dtype]:
                            if cn in repeat_data:
                                nr = repeat_data[cn].get('No Reduction')
                                dr_val = repeat_data[cn].get(key)
                                if nr is not None and dr_val is not None:
                                    syn_total += 1
                                    if dr_val > nr + 0.001:
                                        syn_wins += 1
                                    if abs(nr) > 0.001:
                                        syn_changes.append((dr_val - nr) / abs(nr) * 100)
                                    else:
                                        syn_changes.append((dr_val - nr) * 100)
            else:
                # Fallback: use averaged synthetic data
                for dtype in SYNTH_TYPES:
                    if dtype in synth and cn in synth[dtype]:
                        nr = synth[dtype][cn].get('No Reduction')
                        dr_val = synth[dtype][cn].get(key)
                        if nr is not None and dr_val is not None:
                            syn_total += 1
                            if dr_val > nr + 0.001:
                                syn_wins += 1
                            if abs(nr) > 0.001:
                                syn_changes.append((dr_val - nr) / abs(nr) * 100)
                            else:
                                syn_changes.append((dr_val - nr) * 100)
            
            # Real wins and avg change
            real_wins = 0
            real_total = 0
            real_changes = []
            
            for ds in datasets:
                if cn in real[ds]:
                    nr = real[ds][cn].get('No Reduction')
                    dr_val = real[ds][cn].get(key)
                    if nr is not None and dr_val is not None:
                        real_total += 1
                        if dr_val > nr + 0.001:
                            real_wins += 1
                        if abs(nr) > 0.001:
                            real_changes.append((dr_val - nr) / abs(nr) * 100)
                        else:
                            real_changes.append((dr_val - nr) * 100)
            
            win_syn = (syn_wins / syn_total * 100) if syn_total > 0 else 0
            win_real = (real_wins / real_total * 100) if real_total > 0 else 0
            avg_syn = np.mean(syn_changes) if syn_changes else 0
            avg_real = np.mean(real_changes) if real_changes else 0
            
            results.append({
                'method': dr,
                'level': lv,
                'win_syn': win_syn,
                'win_real': win_real,
                'avg_syn': avg_syn,
                'avg_real': avg_real,
            })
    
    return results


def generate_aggregate_tables(real, synth, synth_raw):
    """Tables 2-5: Aggregate statistics per clustering algorithm."""
    table_map = {
        'k-means': (2, 'table_2_aggregate_kmeans'),
        'AHC': (3, 'table_3_aggregate_AHC'),
        'GMM': (4, 'table_4_aggregate_GMM'),
        'OPTICS': (5, 'table_5_aggregate_OPTICS'),
    }
    
    for cn, (tnum, fname) in table_map.items():
        agg = compute_aggregate(real, synth, synth_raw, cn)
        
        # CSV
        with open(os.path.join(TABLES_DIR, fname + '.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['Method', 'Reduction', 'Win% Syn', 'Win% Real', 'Avg% Syn', 'Avg% Real'])
            for r in agg:
                w.writerow([r['method'], r['level'], 
                           f"{r['win_syn']:.2f}", f"{r['win_real']:.2f}",
                           f"{r['avg_syn']:.2f}", f"{r['avg_real']:.2f}"])
        
        # TXT
        with open(os.path.join(TABLES_DIR, fname + '.txt'), 'w') as f:
            f.write(f'Table {tnum}: Aggregate statistics for {cn}\n')
            f.write('=' * 78 + '\n')
            f.write(f'{"Method":<16}{"Reduction":<12}{"Win% Syn":>10}{"Win% Real":>12}{"Avg% Syn":>14}{"Avg% Real":>14}\n')
            f.write('-' * 78 + '\n')
            for r in agg:
                f.write(f'{r["method"]:<16}{r["level"]:<12}{r["win_syn"]:>10.2f}{r["win_real"]:>12.2f}'
                       f'{r["avg_syn"]:>14.2f}{r["avg_real"]:>14.2f}\n')
        
        print(f'  Generated {fname}')


def generate_wilcoxon(real):
    """Table A.9: Wilcoxon signed-rank test (one-sided).
    Format matches paper: rows = clustering algorithms, 
    columns = DR methods × levels
    H1: ARI_method > ARI_baseline, alpha=0.05, n=20
    """
    datasets = sorted(real.keys())
    
    # Build the table: rows = clustering, cols = DR × levels
    dr_level_cols = []
    for dr in DR_METHODS:
        for lv in LEVELS:
            dr_level_cols.append(f'{dr}_{lv}')
    
    results_matrix = {}
    
    for cn in CLUSTERING:
        results_matrix[cn] = {}
        for col_key in dr_level_cols:
            nr_vals = []
            dr_vals = []
            
            for ds in datasets:
                if cn in real[ds]:
                    nr = real[ds][cn].get('No Reduction')
                    dv = real[ds][cn].get(col_key)
                    if nr is not None and dv is not None:
                        nr_vals.append(nr)
                        dr_vals.append(dv)
            
            if len(nr_vals) >= 5:
                diffs = np.array(dr_vals) - np.array(nr_vals)
                # One-sided Wilcoxon: H1 = DR > baseline
                try:
                    stat, pval = stats.wilcoxon(diffs, alternative='greater')
                except:
                    try:
                        # Fallback for older scipy
                        stat, pval_two = stats.wilcoxon(diffs)
                        # For one-sided, halve the two-sided p-value
                        # But need to check direction
                        if np.sum(diffs > 0) > np.sum(diffs < 0):
                            pval = pval_two / 2
                        else:
                            pval = 1 - pval_two / 2
                    except:
                        pval = 1.0
            else:
                pval = 1.0
            
            results_matrix[cn][col_key] = pval
    
    # CSV
    with open(os.path.join(TABLES_DIR, 'table_A9_wilcoxon.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        # Header with DR/level grouping
        header1 = ['']
        header2 = ['Algorithm']
        for dr in DR_METHODS:
            for lv in LEVELS:
                header1.append(dr)
                header2.append(lv)
        w.writerow(header1)
        w.writerow(header2)
        
        for cn in CLUSTERING:
            row = [cn]
            for col_key in dr_level_cols:
                pval = results_matrix[cn][col_key]
                row.append(f"{pval:.3f}")
            w.writerow(row)
    
    # TXT (matching paper format)
    with open(os.path.join(TABLES_DIR, 'table_A9_wilcoxon.txt'), 'w') as f:
        f.write('Table A.9: Wilcoxon signed-rank test results on 20 real-world benchmarks\n')
        f.write('One-sided test: H1: ARI_method > ARI_baseline, alpha = 0.05, n = 20\n')
        f.write('Significant p-values (< 0.05) are marked with *\n')
        f.write('=' * 120 + '\n')
        
        # Column headers
        line1 = f'{"":>14}'
        for dr in DR_METHODS:
            span = len(LEVELS) * 8
            line1 += f'{dr:^{span}}'
        f.write(line1 + '\n')
        
        line2 = f'{"Algorithm":>14}'
        for dr in DR_METHODS:
            for lv in LEVELS:
                line2 += f'{lv:>8}'
        f.write(line2 + '\n')
        f.write('-' * 120 + '\n')
        
        for cn in CLUSTERING:
            line = f'{cn:>14}'
            for col_key in dr_level_cols:
                pval = results_matrix[cn][col_key]
                marker = '*' if pval < 0.05 else ' '
                line += f'{pval:>7.3f}{marker}'
            f.write(line + '\n')
        
        f.write('-' * 120 + '\n')
    
    print(f'  Generated table_A9_wilcoxon')
    
    # Count significant results
    sig_count = 0
    total = 0
    for cn in results_matrix:
        for col in results_matrix[cn]:
            total += 1
            if results_matrix[cn][col] < 0.05:
                sig_count += 1
    print(f'  {sig_count}/{total} tests significant at p<0.05')


def generate_real_boxplots(real):
    """Figures for real-world data: boxplots per clustering algorithm."""
    datasets = sorted(real.keys())
    
    for cn in CLUSTERING:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
        safe_cn = cn.replace('-', '_')
        fig.suptitle(f'ARI Distribution for {cn} on Real-World Data', fontsize=14, fontweight='bold')
        
        for li, lv in enumerate(LEVELS):
            ax = axes[li]
            data_to_plot = []
            tick_labels = ['No DR'] + DR_METHODS
            
            # No Reduction
            nr_vals = []
            for ds in datasets:
                if cn in real[ds]:
                    v = real[ds][cn].get('No Reduction')
                    if v is not None:
                        nr_vals.append(v)
            data_to_plot.append(nr_vals)
            
            # Each DR method
            for dr in DR_METHODS:
                key = f'{dr}_{lv}'
                vals = []
                for ds in datasets:
                    if cn in real[ds]:
                        v = real[ds][cn].get(key)
                        if v is not None:
                            vals.append(v)
                data_to_plot.append(vals)
            
            bp = ax.boxplot(data_to_plot, labels=tick_labels, patch_artist=True)
            colors = ['#CCCCCC', '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_title(f'Reduction: {lv}', fontsize=12)
            ax.set_ylabel('ARI' if li == 0 else '')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim(-0.2, 1.05)
        
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f'boxplot_{safe_cn}_real.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  Generated boxplot_{safe_cn}_real.png')


def generate_synthetic_boxplots(synth_raw):
    """Figures for synthetic data: boxplots per clustering algorithm.
    Each boxplot shows ARI values across all synthetic datasets/repeats."""
    if not synth_raw:
        print('  No raw synthetic data available for boxplots')
        return
    
    for cn in CLUSTERING:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
        safe_cn = cn.replace('-', '_')
        fig.suptitle(f'ARI Distribution for {cn} on Synthetic Data', fontsize=14, fontweight='bold')
        
        for li, lv in enumerate(LEVELS):
            ax = axes[li]
            data_to_plot = []
            tick_labels = ['No DR'] + DR_METHODS
            
            # Collect values across all synthetic types and repeats
            # No Reduction
            nr_vals = []
            for dtype in SYNTH_TYPES:
                if dtype in synth_raw:
                    for repeat_data in synth_raw[dtype]:
                        if cn in repeat_data:
                            v = repeat_data[cn].get('No Reduction')
                            if v is not None:
                                nr_vals.append(v)
            data_to_plot.append(nr_vals)
            
            # Each DR method
            for dr in DR_METHODS:
                key = f'{dr}_{lv}'
                vals = []
                for dtype in SYNTH_TYPES:
                    if dtype in synth_raw:
                        for repeat_data in synth_raw[dtype]:
                            if cn in repeat_data:
                                v = repeat_data[cn].get(key)
                                if v is not None:
                                    vals.append(v)
                data_to_plot.append(vals)
            
            bp = ax.boxplot(data_to_plot, labels=tick_labels, patch_artist=True)
            colors = ['#CCCCCC', '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_title(f'Reduction: {lv}', fontsize=12)
            ax.set_ylabel('ARI' if li == 0 else '')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim(-0.2, 1.05)
        
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f'boxplot_{safe_cn}_synthetic.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  Generated boxplot_{safe_cn}_synthetic.png')


def generate_heatmaps(real):
    """Heatmaps showing ARI per dataset per DR method."""
    datasets = sorted(real.keys())
    
    for cn in CLUSTERING:
        safe_cn = cn.replace('-', '_')
        fig, ax = plt.subplots(figsize=(20, 10))
        
        matrix = []
        for ds in datasets:
            row = []
            for col in ALL_COLS:
                v = real[ds].get(cn, {}).get(col)
                row.append(v if v is not None else 0)
            matrix.append(row)
        
        matrix = np.array(matrix)
        
        im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=-0.1, vmax=1.0)
        ax.set_xticks(range(len(ALL_COLS)))
        ax.set_xticklabels(ALL_COLS, rotation=90, fontsize=7)
        ax.set_yticks(range(len(datasets)))
        ax.set_yticklabels(datasets, fontsize=8)
        ax.set_title(f'ARI Heatmap: {cn}', fontsize=14)
        plt.colorbar(im, ax=ax, label='ARI')
        
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f'heatmap_{safe_cn}_real.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  Generated heatmap_{safe_cn}_real.png')


def generate_summary_table(real, synth):
    """Generate a summary table comparing paper's key findings."""
    datasets = sorted(real.keys())
    
    with open(os.path.join(TABLES_DIR, 'summary.txt'), 'w') as f:
        f.write('SUMMARY OF KEY FINDINGS\n')
        f.write('=' * 80 + '\n\n')
        
        for cn in CLUSTERING:
            f.write(f'\n{cn}:\n')
            f.write('-' * 60 + '\n')
            
            # Real-world: mean ARI for No Reduction vs each DR
            nr_vals = []
            for ds in datasets:
                if cn in real[ds]:
                    v = real[ds][cn].get('No Reduction')
                    if v is not None:
                        nr_vals.append(v)
            
            mean_nr = np.mean(nr_vals) if nr_vals else 0
            f.write(f'  No Reduction (real): mean ARI = {mean_nr:.3f}\n')
            
            for dr in DR_METHODS:
                for lv in LEVELS:
                    key = f'{dr}_{lv}'
                    vals = []
                    for ds in datasets:
                        if cn in real[ds]:
                            v = real[ds][cn].get(key)
                            if v is not None:
                                vals.append(v)
                    mean_v = np.mean(vals) if vals else 0
                    diff = mean_v - mean_nr
                    f.write(f'  {dr} ({lv}): mean ARI = {mean_v:.3f} ({"+" if diff >= 0 else ""}{diff:.3f})\n')
            
            # Synthetic
            f.write(f'\n  Synthetic data:\n')
            for dtype in SYNTH_TYPES:
                if dtype in synth and cn in synth[dtype]:
                    nr = synth[dtype][cn].get('No Reduction', 0)
                    f.write(f'    {dtype}: NR={nr:.3f}')
                    best_key = None
                    best_val = nr
                    for dr in DR_METHODS:
                        for lv in LEVELS:
                            key = f'{dr}_{lv}'
                            v = synth[dtype][cn].get(key)
                            if v is not None and v > best_val:
                                best_val = v
                                best_key = key
                    if best_key:
                        f.write(f', best DR: {best_key}={best_val:.3f}')
                    f.write('\n')
        
    print('  Generated summary.txt')


def compare_with_paper(real):
    """Compare our real-world k-means results with selected paper values."""
    # Paper's k-means table (from lines 871-891)
    paper = {
        'Breast tissue': 0.27, 'Breast Wisconsin': 0.67, 'Ecoli': 0.51,
        'Glass': 0.18, 'Haberman': 0.00, 'Ionosphere': 0.17,
        'Iris': 0.62, 'Wine': 0.90, 'Yeast': 0.16,
    }
    
    with open(os.path.join(TABLES_DIR, 'paper_comparison.txt'), 'w') as f:
        f.write('Comparison of our k-means (No Reduction) results with paper values\n')
        f.write('=' * 60 + '\n')
        f.write(f'{"Dataset":<22}{"Paper":>8}{"Ours":>8}{"Diff":>8}\n')
        f.write('-' * 60 + '\n')
        
        for ds_paper, paper_val in paper.items():
            for ds_ours in real:
                if ds_paper.lower().replace(' ', '').replace('_', '') == ds_ours.lower().replace(' ', '').replace('_', ''):
                    our_val = real[ds_ours].get('k-means', {}).get('No Reduction', 0)
                    diff = our_val - paper_val
                    f.write(f'{ds_paper:<22}{paper_val:>8.2f}{our_val:>8.2f}{diff:>+8.2f}\n')
                    break
    
    print('  Generated paper_comparison.txt')


def main():
    print('Loading data...')
    real, synth, synth_raw = load_data()
    
    print(f'\nReal datasets: {len(real)}')
    print(f'Synthetic types: {list(synth.keys())}')
    if synth_raw:
        for dtype in synth_raw:
            print(f'  {dtype}: {len(synth_raw[dtype])} repeats')
    
    print('\n--- Synthetic Tables (A.1-A.4) ---')
    generate_synthetic_tables(synth)
    
    print('\n--- Real-World Tables (A.5-A.8) ---')
    generate_real_tables(real)
    
    print('\n--- Aggregate Tables (2-5) ---')
    generate_aggregate_tables(real, synth, synth_raw)
    
    print('\n--- Wilcoxon Test (A.9) ---')
    generate_wilcoxon(real)
    
    print('\n--- Real-World Boxplots ---')
    generate_real_boxplots(real)
    
    print('\n--- Synthetic Boxplots ---')
    generate_synthetic_boxplots(synth_raw)
    
    print('\n--- Heatmaps ---')
    generate_heatmaps(real)
    
    print('\n--- Summary ---')
    generate_summary_table(real, synth)
    
    print('\n--- Paper Comparison ---')
    compare_with_paper(real)
    
    print('\n\nAll tables and figures generated!')
    print(f'Tables: {TABLES_DIR}/')
    print(f'Figures: {FIGURES_DIR}/')


if __name__ == '__main__':
    main()
