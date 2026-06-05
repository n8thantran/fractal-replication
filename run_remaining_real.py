"""Resume real-world experiments from where we left off."""
import numpy as np
import json
import os
import time
import warnings
import sys
warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans, AgglomerativeClustering, OPTICS, cluster_optics_xi
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap, MDS
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Import from main script
from run_all_experiments import (
    DR_METHOD_NAMES, CLUSTERING_NAMES, REDUCTION_LEVELS, DATASET_ORDER,
    apply_dr, get_reduction_levels, cluster_and_evaluate,
    find_best_ahc_params, find_best_gmm_params, find_best_optics_params,
    load_real_datasets, VAE
)

RESULTS_DIR = '/workspace/results'

# Override MDS to be faster
import run_all_experiments
_orig_apply_dr = run_all_experiments.apply_dr

def fast_apply_dr(method_name, X, n_components):
    if method_name == 'MDS':
        if n_components >= X.shape[1]:
            return X.copy()
        n_components = max(1, min(n_components, X.shape[1], X.shape[0] - 1))
        try:
            if X.shape[0] > 1500:
                n_init, max_iter = 2, 50
            elif X.shape[0] > 500:
                n_init, max_iter = 4, 100
            else:
                n_init, max_iter = 10, 200
            return MDS(n_components=n_components, random_state=10, n_init=n_init,
                       max_iter=max_iter, normalized_stress='auto').fit_transform(X)
        except:
            return PCA(n_components=n_components).fit_transform(X)
    return _orig_apply_dr(method_name, X, n_components)

# Monkey-patch
run_all_experiments.apply_dr = fast_apply_dr

if __name__ == '__main__':
    # Load existing results
    results_path = os.path.join(RESULTS_DIR, 'real_world_results.json')
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
    else:
        results = {cm: {} for cm in CLUSTERING_NAMES}
    
    # Find which datasets are done
    done = set()
    for cm in CLUSTERING_NAMES:
        if cm in results:
            done = done.union(set(results[cm].keys())) if not done else done.intersection(set(results[cm].keys()))
    
    print(f"Already done: {sorted(done)}")
    remaining = [ds for ds in DATASET_ORDER if ds not in done]
    print(f"Remaining: {remaining}")
    
    datasets = load_real_datasets()
    
    for ds_name in remaining:
        if ds_name not in datasets: continue
        ds = datasets[ds_name]
        X, y, k = ds['X'], ds['y'], ds['n_clusters']
        red_levels = get_reduction_levels(X.shape[1], k)
        
        print(f"\n--- {ds_name} (n={X.shape[0]}, d={X.shape[1]}, k={k}) ---")
        print(f"    Reduction: k-1={red_levels['k-1']}, 25%={red_levels['25%']}, 50%={red_levels['50%']}")
        
        # Find best params
        t0 = time.time()
        ahc_params, _ = find_best_ahc_params(X, y, k)
        gmm_params, _ = find_best_gmm_params(X, y, k)
        optics_params, _ = find_best_optics_params(X, y, k)
        print(f"    Param search: {time.time()-t0:.1f}s")
        
        params_map = {'k-means': {}, 'AHC': ahc_params, 'GMM': gmm_params, 'OPTICS': optics_params}
        
        # Pre-compute DR
        dr_cache = {}
        for dr_name in DR_METHOD_NAMES:
            for level_name, n_comp in red_levels.items():
                if n_comp >= X.shape[1]:
                    dr_cache[(dr_name, level_name)] = X.copy()
                else:
                    t0 = time.time()
                    try:
                        dr_cache[(dr_name, level_name)] = fast_apply_dr(dr_name, X, n_comp)
                    except Exception as e:
                        print(f"    DR ERROR {dr_name} {level_name}: {e}")
                        dr_cache[(dr_name, level_name)] = None
                    elapsed = time.time() - t0
                    if elapsed > 5:
                        print(f"    {dr_name} {level_name} ({n_comp}d): {elapsed:.1f}s")
        
        # Run clustering
        for cm_name in CLUSTERING_NAMES:
            if cm_name not in results:
                results[cm_name] = {}
            results[cm_name][ds_name] = {}
            params = params_map[cm_name]
            
            ari = cluster_and_evaluate(X, y, k, cm_name, params)
            results[cm_name][ds_name]['No Reduction'] = round(ari, 2)
            
            for dr_name in DR_METHOD_NAMES:
                for level_name in REDUCTION_LEVELS:
                    X_r = dr_cache.get((dr_name, level_name))
                    if X_r is None:
                        ari = 0.0
                    else:
                        try:
                            ari = cluster_and_evaluate(X_r, y, k, cm_name, params)
                        except:
                            ari = 0.0
                    results[cm_name][ds_name][f'{dr_name}_{level_name}'] = round(ari, 2)
            
            print(f"    {cm_name}: baseline={results[cm_name][ds_name]['No Reduction']:.2f}")
        
        # Save after each dataset
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    print(f"\nAll done! {len(results[CLUSTERING_NAMES[0]])} datasets completed.")
