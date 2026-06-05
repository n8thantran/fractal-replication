"""
Run experiments on 20 real-world UCI datasets.
Produces Tables A.5-A.8 (ARI for k-means, AHC, GMM, OPTICS).
"""

import numpy as np
import json
import os
import time
import warnings
warnings.filterwarnings('ignore')

from data_loader import load_all_real_datasets, DATASET_SPECS
from dr_methods import DR_METHODS, get_reduction_levels, apply_dr
from clustering_methods import (
    CLUSTERING_METHODS, PARAM_SEARCH, cluster_and_evaluate,
    find_best_ahc_params, find_best_gmm_params, find_best_optics_params
)

RESULTS_DIR = '/workspace/results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Dataset order from paper
DATASET_ORDER = [
    'Breast tissue', 'Breast Wisconsin', 'Ecoli', 'Glass', 'Haberman',
    'Ionosphere', 'Iris', 'Movement libras', 'Musk', 'Parkinsons',
    'Segmentation', 'Sonar all', 'Spectf', 'Transfusion', 'Vehicle',
    'Vertebral column', 'Vowel context', 'Wine', 'Wine quality red', 'Yeast'
]

DR_METHOD_NAMES = ['PCA', 'Kernel PCA', 'VAE', 'Isomap', 'MDS']
CLUSTERING_NAMES = ['k-means', 'AHC', 'GMM', 'OPTICS']
REDUCTION_LEVELS = ['k-1', '25%', '50%']


def run_real_world_experiments():
    """Run all real-world experiments."""
    print("Loading datasets...")
    datasets = load_all_real_datasets()
    print(f"Loaded {len(datasets)} datasets\n")
    
    # Results structure: {clustering_method: {dataset: {condition: ari}}}
    # condition is 'No Reduction' or 'DR_method_level'
    results = {cm: {} for cm in CLUSTERING_NAMES}
    
    # First, find best params for AHC, GMM, OPTICS on unreduced data
    print("=" * 60)
    print("Phase 1: Finding best parameters for AHC, GMM, OPTICS")
    print("=" * 60)
    
    best_params = {cm: {} for cm in ['AHC', 'GMM', 'OPTICS']}
    
    for ds_name in DATASET_ORDER:
        if ds_name not in datasets:
            print(f"  Skipping {ds_name} (not loaded)")
            continue
        
        ds = datasets[ds_name]
        X, y, k = ds['X'], ds['y'], ds['n_clusters']
        
        print(f"\n  {ds_name} (n={X.shape[0]}, d={X.shape[1]}, k={k}):")
        
        # AHC param search
        t0 = time.time()
        params, ari = find_best_ahc_params(X, y, k)
        best_params['AHC'][ds_name] = params
        print(f"    AHC: {params} -> ARI={ari:.2f} ({time.time()-t0:.1f}s)")
        
        # GMM param search
        t0 = time.time()
        params, ari = find_best_gmm_params(X, y, k)
        best_params['GMM'][ds_name] = params
        print(f"    GMM: {params} -> ARI={ari:.2f} ({time.time()-t0:.1f}s)")
        
        # OPTICS param search
        t0 = time.time()
        params, ari = find_best_optics_params(X, y, k)
        best_params['OPTICS'][ds_name] = params
        print(f"    OPTICS: {params} -> ARI={ari:.2f} ({time.time()-t0:.1f}s)")
    
    # Save best params
    # Convert numpy types to Python types for JSON serialization
    params_serializable = {}
    for cm in best_params:
        params_serializable[cm] = {}
        for ds in best_params[cm]:
            params_serializable[cm][ds] = {}
            for k, v in best_params[cm][ds].items():
                if isinstance(v, (np.integer,)):
                    params_serializable[cm][ds][k] = int(v)
                elif isinstance(v, (np.floating,)):
                    params_serializable[cm][ds][k] = float(v)
                else:
                    params_serializable[cm][ds][k] = v
    
    with open(os.path.join(RESULTS_DIR, 'best_params_real.json'), 'w') as f:
        json.dump(params_serializable, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Phase 2: Running all experiments")
    print("=" * 60)
    
    for ds_name in DATASET_ORDER:
        if ds_name not in datasets:
            continue
        
        ds = datasets[ds_name]
        X, y, k = ds['X'], ds['y'], ds['n_clusters']
        reduction_dims = get_reduction_levels(X.shape[1], k)
        
        print(f"\n  {ds_name} (n={X.shape[0]}, d={X.shape[1]}, k={k})")
        print(f"    Reduction levels: {reduction_dims}")
        
        for cm_name in CLUSTERING_NAMES:
            if ds_name not in results[cm_name]:
                results[cm_name][ds_name] = {}
            
            # Get params for this clustering method
            if cm_name in best_params and ds_name in best_params[cm_name]:
                params = best_params[cm_name][ds_name]
            else:
                params = {}
            
            # No reduction baseline
            t0 = time.time()
            ari_baseline = cluster_and_evaluate(X, y, k, cm_name, params)
            results[cm_name][ds_name]['No Reduction'] = round(ari_baseline, 2)
            print(f"    {cm_name} baseline: ARI={ari_baseline:.2f} ({time.time()-t0:.1f}s)")
            
            # With DR
            for dr_name in DR_METHOD_NAMES:
                for level_name, n_comp in reduction_dims.items():
                    if n_comp >= X.shape[1]:
                        # No actual reduction needed
                        results[cm_name][ds_name][f'{dr_name}_{level_name}'] = round(ari_baseline, 2)
                        continue
                    
                    t0 = time.time()
                    try:
                        X_reduced = apply_dr(dr_name, X, n_comp)
                        ari = cluster_and_evaluate(X_reduced, y, k, cm_name, params)
                    except Exception as e:
                        print(f"      ERROR: {dr_name} {level_name}: {e}")
                        ari = 0.0
                    
                    results[cm_name][ds_name][f'{dr_name}_{level_name}'] = round(ari, 2)
                    elapsed = time.time() - t0
                    if elapsed > 2:
                        print(f"      {dr_name} {level_name}: ARI={ari:.2f} ({elapsed:.1f}s)")
        
        # Save intermediate results
        with open(os.path.join(RESULTS_DIR, 'real_world_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Phase 3: Generating result tables")
    print("=" * 60)
    
    generate_tables(results)
    
    return results


def generate_tables(results):
    """Generate CSV tables matching paper format."""
    for cm_name in CLUSTERING_NAMES:
        rows = []
        header = ['Dataset', 'No Reduction']
        for dr_name in DR_METHOD_NAMES:
            for level in REDUCTION_LEVELS:
                header.append(f'{dr_name}_{level}')
        
        rows.append(header)
        
        for ds_name in DATASET_ORDER:
            if ds_name not in results[cm_name]:
                continue
            row = [ds_name]
            row.append(str(results[cm_name][ds_name].get('No Reduction', '')))
            for dr_name in DR_METHOD_NAMES:
                for level in REDUCTION_LEVELS:
                    key = f'{dr_name}_{level}'
                    val = results[cm_name][ds_name].get(key, '')
                    row.append(str(val))
            rows.append(row)
        
        # Write CSV
        fname = f'table_real_{cm_name.replace("-", "").replace(" ", "_").lower()}.csv'
        with open(os.path.join(RESULTS_DIR, fname), 'w') as f:
            for row in rows:
                f.write(','.join(row) + '\n')
        print(f"  Saved {fname}")


if __name__ == '__main__':
    results = run_real_world_experiments()
    print("\nDone!")
