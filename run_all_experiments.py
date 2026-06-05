"""
Comprehensive experiment runner for the paper:
"Assessing the impact of dimensionality reduction on clustering performance"

Runs both real-world (20 UCI datasets) and synthetic experiments.
Produces Tables A.1-A.8, Tables 2-5, Table A.9, and boxplot figures.
"""

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
from scipy.stats import wilcoxon
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

RESULTS_DIR = '/workspace/results'
os.makedirs(RESULTS_DIR, exist_ok=True)

DR_METHOD_NAMES = ['PCA', 'Kernel PCA', 'VAE', 'Isomap', 'MDS']
CLUSTERING_NAMES = ['k-means', 'AHC', 'GMM', 'OPTICS']
REDUCTION_LEVELS = ['k-1', '25%', '50%']

# ============================================================
# DR Methods
# ============================================================

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.BatchNorm1d(32), nn.Dropout(0.4),
        )
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(),
            nn.BatchNorm1d(32), nn.Dropout(0.4),
            nn.Linear(32, 64), nn.ReLU(),
            nn.BatchNorm1d(64), nn.Dropout(0.4),
            nn.Linear(64, input_dim), nn.Sigmoid(),
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar


def apply_dr(method_name, X, n_components):
    """Apply dimensionality reduction."""
    if n_components >= X.shape[1]:
        return X.copy()
    n_components = max(1, min(n_components, X.shape[1], X.shape[0] - 1))
    
    if method_name == 'PCA':
        return PCA(n_components=n_components).fit_transform(X)
    
    elif method_name == 'Kernel PCA':
        try:
            result = KernelPCA(n_components=n_components, kernel='rbf').fit_transform(X)
            if result.shape[1] < n_components:
                result = np.hstack([result, np.zeros((result.shape[0], n_components - result.shape[1]))])
            return result
        except:
            return PCA(n_components=n_components).fit_transform(X)
    
    elif method_name == 'VAE':
        # Min-max scale to [0,1]
        X_min, X_max = X.min(0), X.max(0)
        denom = X_max - X_min; denom[denom == 0] = 1
        X_s = (X - X_min) / denom
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        X_t = torch.FloatTensor(X_s).to(device)
        n = X_s.shape[0]; n_train = int(0.7 * n)
        idx = np.random.permutation(n)
        loader = DataLoader(TensorDataset(X_t[idx[:n_train]]), batch_size=64, shuffle=True)
        model = VAE(X.shape[1], n_components).to(device)
        opt = torch.optim.Adam(model.parameters())
        model.train()
        for _ in range(100):
            for (batch,) in loader:
                xr, mu, lv = model(batch)
                loss = nn.functional.mse_loss(xr, batch, reduction='sum') - 0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp())
                opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            mu, _ = model.encode(X_t)
        return mu.cpu().numpy()
    
    elif method_name == 'Isomap':
        try:
            return Isomap(n_components=n_components).fit_transform(X)
        except:
            return PCA(n_components=n_components).fit_transform(X)
    
    elif method_name == 'MDS':
        try:
            # Reduce n_init for large datasets to save time
            n_init = 50 if X.shape[0] < 1000 else 10
            max_iter = 300 if X.shape[0] < 1000 else 150
            return MDS(n_components=n_components, random_state=10, n_init=n_init, 
                       max_iter=max_iter, normalized_stress='auto').fit_transform(X)
        except:
            return PCA(n_components=n_components).fit_transform(X)


def get_reduction_levels(n_features, n_clusters):
    k_minus_1 = max(n_clusters - 1, 2)
    pct_25 = max(int(np.round(0.25 * n_features)), 2)
    pct_50 = max(int(np.round(0.50 * n_features)), 2)
    return {'k-1': k_minus_1, '25%': pct_25, '50%': pct_50}


# ============================================================
# Clustering Methods
# ============================================================

def run_kmeans(X, k, **kw):
    return KMeans(n_clusters=k, init='k-means++', n_init=100, random_state=42).fit_predict(X)

def run_ahc(X, k, affinity='euclidean', linkage='ward', **kw):
    if linkage == 'ward': affinity = 'euclidean'
    try:
        return AgglomerativeClustering(n_clusters=k, metric=affinity if linkage != 'ward' else 'euclidean', linkage=linkage).fit_predict(X)
    except:
        return AgglomerativeClustering(n_clusters=k).fit_predict(X)

def run_gmm(X, k, covariance_type='full', **kw):
    try:
        return GaussianMixture(n_components=k, covariance_type=covariance_type, random_state=42, max_iter=200).fit_predict(X)
    except:
        return GaussianMixture(n_components=k, covariance_type='diag', random_state=42).fit_predict(X)

def run_optics(X, k, min_samples=5, min_cluster_size=0.05, **kw):
    try:
        return OPTICS(min_samples=min_samples, cluster_method='xi', xi=min_cluster_size).fit_predict(X)
    except:
        return -np.ones(X.shape[0], dtype=int)


def find_best_ahc_params(X, y, k):
    best_ari, best_params = -2, {'affinity': 'euclidean', 'linkage': 'ward'}
    for linkage in ['complete', 'average', 'single', 'ward']:
        affs = ['euclidean'] if linkage == 'ward' else ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']
        for aff in affs:
            try:
                labels = run_ahc(X, k, affinity=aff, linkage=linkage)
                ari = adjusted_rand_score(y, labels)
                if ari > best_ari: best_ari, best_params = ari, {'affinity': aff, 'linkage': linkage}
            except: pass
    return best_params, best_ari

def find_best_gmm_params(X, y, k):
    best_ari, best_params = -2, {'covariance_type': 'full'}
    for ct in ['spherical', 'tied', 'diag', 'full']:
        try:
            labels = run_gmm(X, k, covariance_type=ct)
            ari = adjusted_rand_score(y, labels)
            if ari > best_ari: best_ari, best_params = ari, {'covariance_type': ct}
        except: pass
    return best_params, best_ari

def find_best_optics_params(X, y, k):
    best_ari, best_params = -2, {'min_samples': 5, 'min_cluster_size': 0.05}
    xi_values = np.arange(0.01, 1.01, 0.05)
    for ms in range(5, 11):
        try:
            optics = OPTICS(min_samples=ms, cluster_method='xi', xi=0.05)
            optics.fit(X)
            for xi_val in xi_values:
                try:
                    labels, _ = cluster_optics_xi(
                        reachability=optics.reachability_, predecessor=optics.predecessor_,
                        ordering=optics.ordering_, min_samples=ms, xi=xi_val,
                        predecessor_correction=optics.predecessor_correction)
                    ari = adjusted_rand_score(y, labels)
                    if ari > best_ari: best_ari, best_params = ari, {'min_samples': ms, 'min_cluster_size': float(xi_val)}
                except: pass
        except: pass
    return best_params, best_ari


def cluster_and_evaluate(X, y, k, method, params=None):
    if params is None: params = {}
    funcs = {'k-means': run_kmeans, 'AHC': run_ahc, 'GMM': run_gmm, 'OPTICS': run_optics}
    labels = funcs[method](X, k, **params)
    return adjusted_rand_score(y, labels)


# ============================================================
# Real-World Experiments
# ============================================================

DATASET_ORDER = [
    'Breast tissue', 'Breast Wisconsin', 'Ecoli', 'Glass', 'Haberman',
    'Ionosphere', 'Iris', 'Movement libras', 'Musk', 'Parkinsons',
    'Segmentation', 'Sonar all', 'Spectf', 'Transfusion', 'Vehicle',
    'Vertebral column', 'Vowel context', 'Wine', 'Wine quality red', 'Yeast'
]


def load_real_datasets():
    """Load all 20 UCI datasets from cache."""
    from data_loader import DATASET_SPECS
    datasets = {}
    cache_dir = '/workspace/data/uci'
    for name in DATASET_ORDER:
        path = os.path.join(cache_dir, f"{name.replace(' ', '_')}.npz")
        if os.path.exists(path):
            data = np.load(path)
            X = StandardScaler().fit_transform(data['X'])
            datasets[name] = {'X': X, 'y': data['y'], 'n_clusters': DATASET_SPECS[name]['clusters']}
    return datasets


def run_real_experiments():
    """Run all real-world experiments. Returns results dict."""
    print("=" * 70)
    print("REAL-WORLD EXPERIMENTS")
    print("=" * 70)
    
    datasets = load_real_datasets()
    print(f"Loaded {len(datasets)} datasets")
    
    results = {cm: {} for cm in CLUSTERING_NAMES}
    
    for ds_name in DATASET_ORDER:
        if ds_name not in datasets: continue
        ds = datasets[ds_name]
        X, y, k = ds['X'], ds['y'], ds['n_clusters']
        red_levels = get_reduction_levels(X.shape[1], k)
        
        print(f"\n--- {ds_name} (n={X.shape[0]}, d={X.shape[1]}, k={k}) ---")
        print(f"    Reduction: k-1={red_levels['k-1']}, 25%={red_levels['25%']}, 50%={red_levels['50%']}")
        
        # Find best params for AHC, GMM, OPTICS on original data
        t0 = time.time()
        ahc_params, _ = find_best_ahc_params(X, y, k)
        gmm_params, _ = find_best_gmm_params(X, y, k)
        optics_params, _ = find_best_optics_params(X, y, k)
        print(f"    Param search: {time.time()-t0:.1f}s")
        
        params_map = {'k-means': {}, 'AHC': ahc_params, 'GMM': gmm_params, 'OPTICS': optics_params}
        
        # Pre-compute all DR transformations (shared across clustering methods)
        dr_cache = {}
        for dr_name in DR_METHOD_NAMES:
            for level_name, n_comp in red_levels.items():
                if n_comp >= X.shape[1]:
                    dr_cache[(dr_name, level_name)] = X.copy()
                else:
                    t0 = time.time()
                    try:
                        dr_cache[(dr_name, level_name)] = apply_dr(dr_name, X, n_comp)
                    except Exception as e:
                        print(f"    DR ERROR {dr_name} {level_name}: {e}")
                        dr_cache[(dr_name, level_name)] = None
                    elapsed = time.time() - t0
                    if elapsed > 5:
                        print(f"    {dr_name} {level_name} ({n_comp}d): {elapsed:.1f}s")
        
        # Run all clustering
        for cm_name in CLUSTERING_NAMES:
            results[cm_name][ds_name] = {}
            params = params_map[cm_name]
            
            # Baseline
            ari = cluster_and_evaluate(X, y, k, cm_name, params)
            results[cm_name][ds_name]['No Reduction'] = round(ari, 2)
            
            # With DR
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
        
        # Save intermediate
        with open(os.path.join(RESULTS_DIR, 'real_world_results.json'), 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    return results


# ============================================================
# Synthetic Data Generation
# ============================================================

def generate_circles_dataset(n_samples=500, n_noise_dims=20, noise_std=0.05, n_circles=5, random_state=None):
    """Generate concentric circles with noise dimensions."""
    rng = np.random.RandomState(random_state)
    samples_per_circle = n_samples // n_circles
    X_list, y_list = [], []
    for i in range(n_circles):
        r = (i + 1) / n_circles
        theta = rng.uniform(0, 2 * np.pi, samples_per_circle)
        x1 = r * np.cos(theta) + rng.normal(0, noise_std, samples_per_circle)
        x2 = r * np.sin(theta) + rng.normal(0, noise_std, samples_per_circle)
        X_list.append(np.column_stack([x1, x2]))
        y_list.append(np.full(samples_per_circle, i))
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    # Add noise dimensions
    noise = rng.normal(0, 1, (X.shape[0], n_noise_dims))
    X = np.hstack([X, noise])
    return X, y, n_circles


def generate_moons_dataset(n_samples=500, n_noise_dims=20, noise=0.1, random_state=None):
    """Generate two interleaving half circles with noise dimensions."""
    from sklearn.datasets import make_moons
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    rng = np.random.RandomState(random_state)
    noise_dims = rng.normal(0, 1, (X.shape[0], n_noise_dims))
    X = np.hstack([X, noise_dims])
    return X, y, 2


def generate_rsg_dataset(n_samples=500, n_features=20, n_clusters=5, random_state=None):
    """Rodriguez Structured Gaussian - well-separated Gaussian clusters with varying density."""
    rng = np.random.RandomState(random_state)
    samples_per_cluster = n_samples // n_clusters
    X_list, y_list = [], []
    # Generate cluster centers far apart
    centers = rng.uniform(-10, 10, (n_clusters, n_features))
    for i in range(n_clusters):
        # Varying covariance
        scale = rng.uniform(0.3, 1.5)
        X_i = rng.normal(centers[i], scale, (samples_per_cluster, n_features))
        X_list.append(X_i)
        y_list.append(np.full(samples_per_cluster, i))
    return np.vstack(X_list), np.concatenate(y_list), n_clusters


def generate_repliclust_dataset(n_samples=500, n_features=20, n_clusters=5, random_state=None):
    """Repliclust-style dataset - clusters with different shapes and overlaps."""
    rng = np.random.RandomState(random_state)
    samples_per_cluster = n_samples // n_clusters
    X_list, y_list = [], []
    for i in range(n_clusters):
        center = rng.uniform(-5, 5, n_features)
        # Random covariance matrix
        A = rng.randn(n_features, n_features) * 0.5
        cov = A @ A.T + np.eye(n_features) * 0.1
        X_i = rng.multivariate_normal(center, cov, samples_per_cluster)
        X_list.append(X_i)
        y_list.append(np.full(samples_per_cluster, i))
    return np.vstack(X_list), np.concatenate(y_list), n_clusters


SYNTHETIC_GENERATORS = {
    'Circles': generate_circles_dataset,
    'Moons': generate_moons_dataset,
    'RSG': generate_rsg_dataset,
    'Repliclust': generate_repliclust_dataset,
}


def run_synthetic_experiments(n_repeats=10):
    """Run synthetic experiments. Average over n_repeats random seeds."""
    print("\n" + "=" * 70)
    print("SYNTHETIC EXPERIMENTS")
    print("=" * 70)
    
    # Results: {data_type: {clustering: {condition: [ari_per_repeat]}}}
    results = {}
    
    for dtype, gen_func in SYNTHETIC_GENERATORS.items():
        print(f"\n--- {dtype} ---")
        results[dtype] = {cm: {} for cm in CLUSTERING_NAMES}
        
        for repeat in range(n_repeats):
            seed = repeat * 42 + 7
            np.random.seed(seed)
            
            # Generate dataset
            if dtype == 'Circles':
                X, y, k = gen_func(n_samples=500, n_noise_dims=20, random_state=seed)
            elif dtype == 'Moons':
                X, y, k = gen_func(n_samples=500, n_noise_dims=20, random_state=seed)
            elif dtype == 'RSG':
                X, y, k = gen_func(n_samples=500, n_features=20, n_clusters=5, random_state=seed)
            elif dtype == 'Repliclust':
                X, y, k = gen_func(n_samples=500, n_features=20, n_clusters=5, random_state=seed)
            
            # Z-score normalize
            X = StandardScaler().fit_transform(X)
            red_levels = get_reduction_levels(X.shape[1], k)
            
            # Find best params
            ahc_params, _ = find_best_ahc_params(X, y, k)
            gmm_params, _ = find_best_gmm_params(X, y, k)
            optics_params, _ = find_best_optics_params(X, y, k)
            params_map = {'k-means': {}, 'AHC': ahc_params, 'GMM': gmm_params, 'OPTICS': optics_params}
            
            # Pre-compute DR
            dr_cache = {}
            for dr_name in DR_METHOD_NAMES:
                for level_name, n_comp in red_levels.items():
                    if n_comp >= X.shape[1]:
                        dr_cache[(dr_name, level_name)] = X.copy()
                    else:
                        try:
                            dr_cache[(dr_name, level_name)] = apply_dr(dr_name, X, n_comp)
                        except:
                            dr_cache[(dr_name, level_name)] = None
            
            # Run clustering
            for cm_name in CLUSTERING_NAMES:
                params = params_map[cm_name]
                
                # Baseline
                cond = 'No Reduction'
                if cond not in results[dtype][cm_name]:
                    results[dtype][cm_name][cond] = []
                ari = cluster_and_evaluate(X, y, k, cm_name, params)
                results[dtype][cm_name][cond].append(ari)
                
                # With DR
                for dr_name in DR_METHOD_NAMES:
                    for level_name in REDUCTION_LEVELS:
                        cond = f'{dr_name}_{level_name}'
                        if cond not in results[dtype][cm_name]:
                            results[dtype][cm_name][cond] = []
                        X_r = dr_cache.get((dr_name, level_name))
                        if X_r is None:
                            ari = 0.0
                        else:
                            try:
                                ari = cluster_and_evaluate(X_r, y, k, cm_name, params)
                            except:
                                ari = 0.0
                        results[dtype][cm_name][cond].append(ari)
            
            print(f"  Repeat {repeat+1}/{n_repeats} done")
        
        # Save intermediate
        avg_results = {}
        for dt in results:
            avg_results[dt] = {}
            for cm in results[dt]:
                avg_results[dt][cm] = {}
                for cond, vals in results[dt][cm].items():
                    avg_results[dt][cm][cond] = round(np.mean(vals), 3)
        
        with open(os.path.join(RESULTS_DIR, 'synthetic_results.json'), 'w') as f:
            json.dump(avg_results, f, indent=2)
    
    return results


# ============================================================
# Table Generation
# ============================================================

def generate_real_tables(results):
    """Generate CSV tables for real-world results (Tables A.5-A.8)."""
    for cm_name in CLUSTERING_NAMES:
        rows = [['Dataset', 'No Reduction'] + [f'{dr}_{lv}' for dr in DR_METHOD_NAMES for lv in REDUCTION_LEVELS]]
        for ds_name in DATASET_ORDER:
            if ds_name not in results[cm_name]: continue
            row = [ds_name, str(results[cm_name][ds_name].get('No Reduction', ''))]
            for dr in DR_METHOD_NAMES:
                for lv in REDUCTION_LEVELS:
                    row.append(str(results[cm_name][ds_name].get(f'{dr}_{lv}', '')))
            rows.append(row)
        fname = f'table_real_{cm_name.replace("-","").replace(" ","_").lower()}.csv'
        with open(os.path.join(RESULTS_DIR, fname), 'w') as f:
            for row in rows: f.write(','.join(row) + '\n')
        print(f"  Saved {fname}")


def generate_synthetic_tables(results):
    """Generate CSV tables for synthetic results (Tables A.1-A.4)."""
    for dtype in results:
        rows = [['Algorithm', 'No Reduction'] + [f'{dr}_{lv}' for dr in DR_METHOD_NAMES for lv in REDUCTION_LEVELS]]
        for cm_name in CLUSTERING_NAMES:
            row = [cm_name, f"{np.mean(results[dtype][cm_name].get('No Reduction', [0])):.3f}"]
            for dr in DR_METHOD_NAMES:
                for lv in REDUCTION_LEVELS:
                    vals = results[dtype][cm_name].get(f'{dr}_{lv}', [0])
                    row.append(f"{np.mean(vals):.3f}")
            rows.append(row)
        fname = f'table_synthetic_{dtype.lower().replace(" ","_")}.csv'
        with open(os.path.join(RESULTS_DIR, fname), 'w') as f:
            for row in rows: f.write(','.join(row) + '\n')
        print(f"  Saved {fname}")


def generate_aggregate_tables(real_results):
    """Generate aggregate tables (Tables 2-5): for each clustering method,
    count how often each DR method+level beats baseline, and avg improvement."""
    print("\n--- Aggregate Tables (Tables 2-5) ---")
    
    for cm_name in CLUSTERING_NAMES:
        print(f"\n  {cm_name}:")
        header = ['DR Method', 'Level', 'Win Count', 'Loss Count', 'Tie Count', 
                  'Avg ARI Improvement', 'Avg ARI When Better', 'Avg ARI When Worse']
        rows = [header]
        
        for dr in DR_METHOD_NAMES:
            for lv in REDUCTION_LEVELS:
                key = f'{dr}_{lv}'
                wins, losses, ties = 0, 0, 0
                improvements, better_vals, worse_vals = [], [], []
                
                for ds_name in DATASET_ORDER:
                    if ds_name not in real_results[cm_name]: continue
                    baseline = real_results[cm_name][ds_name].get('No Reduction', 0)
                    dr_ari = real_results[cm_name][ds_name].get(key, 0)
                    diff = dr_ari - baseline
                    improvements.append(diff)
                    
                    if diff > 0.005:
                        wins += 1
                        better_vals.append(diff)
                    elif diff < -0.005:
                        losses += 1
                        worse_vals.append(diff)
                    else:
                        ties += 1
                
                avg_imp = np.mean(improvements) if improvements else 0
                avg_better = np.mean(better_vals) if better_vals else 0
                avg_worse = np.mean(worse_vals) if worse_vals else 0
                
                rows.append([dr, lv, str(wins), str(losses), str(ties),
                           f"{avg_imp:.3f}", f"{avg_better:.3f}", f"{avg_worse:.3f}"])
        
        fname = f'table_aggregate_{cm_name.replace("-","").replace(" ","_").lower()}.csv'
        with open(os.path.join(RESULTS_DIR, fname), 'w') as f:
            for row in rows: f.write(','.join(row) + '\n')
        print(f"    Saved {fname}")


def generate_wilcoxon_table(real_results):
    """Generate Wilcoxon signed-rank test table (Table A.9)."""
    print("\n--- Wilcoxon Test (Table A.9) ---")
    
    rows = [['Algorithm'] + [f'{dr}_{lv}' for dr in DR_METHOD_NAMES for lv in REDUCTION_LEVELS]]
    
    for cm_name in CLUSTERING_NAMES:
        row = [cm_name]
        for dr in DR_METHOD_NAMES:
            for lv in REDUCTION_LEVELS:
                key = f'{dr}_{lv}'
                baselines, dr_aris = [], []
                
                for ds_name in DATASET_ORDER:
                    if ds_name not in real_results[cm_name]: continue
                    b = real_results[cm_name][ds_name].get('No Reduction', 0)
                    d = real_results[cm_name][ds_name].get(key, 0)
                    baselines.append(b)
                    dr_aris.append(d)
                
                baselines = np.array(baselines)
                dr_aris = np.array(dr_aris)
                diffs = dr_aris - baselines
                
                # One-sided Wilcoxon: H1: DR > baseline
                # Remove zeros
                nonzero = diffs[diffs != 0]
                if len(nonzero) < 2:
                    p_val = 1.0
                else:
                    try:
                        stat, p_two = wilcoxon(nonzero, alternative='greater')
                        p_val = p_two
                    except:
                        p_val = 1.0
                
                row.append(f"{p_val:.4f}")
        rows.append(row)
    
    fname = 'table_wilcoxon.csv'
    with open(os.path.join(RESULTS_DIR, fname), 'w') as f:
        for row in rows: f.write(','.join(row) + '\n')
    print(f"  Saved {fname}")


def generate_boxplots(synthetic_results):
    """Generate boxplot figures for synthetic data."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    print("\n--- Generating Boxplots ---")
    
    for cm_name in CLUSTERING_NAMES:
        fig, axes = plt.subplots(1, 4, figsize=(24, 6), sharey=True)
        fig.suptitle(f'{cm_name} - ARI by DR Method and Data Type', fontsize=14)
        
        for idx, dtype in enumerate(['Circles', 'Moons', 'RSG', 'Repliclust']):
            ax = axes[idx]
            data_to_plot = []
            labels = ['None']
            
            # Baseline
            vals = synthetic_results.get(dtype, {}).get(cm_name, {}).get('No Reduction', [0])
            data_to_plot.append(vals)
            
            for dr in DR_METHOD_NAMES:
                for lv in REDUCTION_LEVELS:
                    key = f'{dr}_{lv}'
                    vals = synthetic_results.get(dtype, {}).get(cm_name, {}).get(key, [0])
                    data_to_plot.append(vals)
                    labels.append(f'{dr}\n{lv}')
            
            bp = ax.boxplot(data_to_plot, labels=labels, vert=True, patch_artist=True)
            ax.set_title(dtype)
            ax.set_ylabel('ARI' if idx == 0 else '')
            ax.tick_params(axis='x', rotation=90, labelsize=6)
            ax.set_ylim(-0.2, 1.1)
            
            # Color by DR method
            colors = ['gray'] + ['#1f77b4']*3 + ['#ff7f0e']*3 + ['#2ca02c']*3 + ['#d62728']*3 + ['#9467bd']*3
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
        
        plt.tight_layout()
        fname = f'boxplot_{cm_name.replace("-","").replace(" ","_").lower()}.png'
        plt.savefig(os.path.join(RESULTS_DIR, fname), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved {fname}")


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    total_start = time.time()
    
    # Parse args
    mode = sys.argv[1] if len(sys.argv) > 1 else 'all'
    
    if mode in ['all', 'real']:
        real_results = run_real_experiments()
        print("\n--- Generating Real-World Tables ---")
        generate_real_tables(real_results)
        generate_aggregate_tables(real_results)
        generate_wilcoxon_table(real_results)
    
    if mode in ['all', 'synthetic']:
        n_repeats = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        synthetic_results = run_synthetic_experiments(n_repeats=n_repeats)
        print("\n--- Generating Synthetic Tables ---")
        generate_synthetic_tables(synthetic_results)
        generate_boxplots(synthetic_results)
    
    total_elapsed = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"TOTAL TIME: {total_elapsed/60:.1f} minutes")
    print(f"{'='*70}")
