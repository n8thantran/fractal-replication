"""
Fast synthetic experiment runner. Skips d=200 to save time.
Uses faster MDS settings.
"""
import numpy as np
import json
import os
import time
import warnings
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

from generate_synthetic import generate_dataset, get_all_configs

RESULTS_DIR = '/workspace/results'
os.makedirs(RESULTS_DIR, exist_ok=True)

DR_METHOD_NAMES = ['PCA', 'Kernel PCA', 'VAE', 'Isomap', 'MDS']
CLUSTERING_NAMES = ['k-means', 'AHC', 'GMM', 'OPTICS']
REDUCTION_LEVELS = ['k-1', '25%', '50%']


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
        X_min, X_max = X.min(0), X.max(0)
        denom = X_max - X_min; denom[denom == 0] = 1
        X_s = (X - X_min) / denom
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        X_t = torch.FloatTensor(X_s).to(device)
        n = X_s.shape[0]; n_train = int(0.7 * n)
        idx = np.random.permutation(n)
        train_data = X_t[idx[:n_train]]
        if train_data.shape[0] < 2:
            return PCA(n_components=n_components).fit_transform(X)
        loader = DataLoader(TensorDataset(train_data), batch_size=64, shuffle=True, drop_last=False)
        model = VAE(X.shape[1], n_components).to(device)
        opt = torch.optim.Adam(model.parameters())
        model.train()
        for _ in range(50):  # Fewer epochs for speed
            for (batch,) in loader:
                if batch.shape[0] < 2: continue
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
            return MDS(n_components=n_components, random_state=10, n_init=1,
                       max_iter=100, normalized_stress='auto').fit_transform(X)
        except:
            return PCA(n_components=n_components).fit_transform(X)


def get_reduction_levels(n_features, n_clusters):
    k_minus_1 = max(n_clusters - 1, 2)
    pct_25 = max(int(np.round(0.25 * n_features)), 2)
    pct_50 = max(int(np.round(0.50 * n_features)), 2)
    return {'k-1': k_minus_1, '25%': pct_25, '50%': pct_50}


def find_best_ahc_params(X, y, k):
    best_ari, best_params = -2, {'affinity': 'euclidean', 'linkage': 'ward'}
    for linkage in ['complete', 'average', 'single', 'ward']:
        affs = ['euclidean'] if linkage == 'ward' else ['euclidean', 'manhattan', 'cosine']
        for aff in affs:
            try:
                labels = AgglomerativeClustering(n_clusters=k, 
                    metric=aff if linkage != 'ward' else 'euclidean', linkage=linkage).fit_predict(X)
                ari = adjusted_rand_score(y, labels)
                if ari > best_ari: best_ari, best_params = ari, {'affinity': aff, 'linkage': linkage}
            except: pass
    return best_params

def find_best_gmm_params(X, y, k):
    best_ari, best_params = -2, {'covariance_type': 'full'}
    for ct in ['spherical', 'tied', 'diag', 'full']:
        try:
            labels = GaussianMixture(n_components=k, covariance_type=ct, random_state=42, max_iter=200).fit_predict(X)
            ari = adjusted_rand_score(y, labels)
            if ari > best_ari: best_ari, best_params = ari, {'covariance_type': ct}
        except: pass
    return best_params

def find_best_optics_params(X, y, k):
    best_ari, best_params = -2, {'min_samples': 5, 'xi': 0.05}
    xi_values = np.arange(0.01, 1.01, 0.1)
    for ms in [5, 7, 10]:
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
                    if ari > best_ari: best_ari, best_params = ari, {'min_samples': ms, 'xi': float(xi_val)}
                except: pass
        except: pass
    return best_params


def cluster_single(X, y, k, method, params):
    try:
        if method == 'k-means':
            labels = KMeans(n_clusters=k, init='k-means++', n_init=100, random_state=42).fit_predict(X)
        elif method == 'AHC':
            labels = AgglomerativeClustering(n_clusters=k,
                metric=params.get('affinity', 'euclidean') if params.get('linkage', 'ward') != 'ward' else 'euclidean',
                linkage=params.get('linkage', 'ward')).fit_predict(X)
        elif method == 'GMM':
            labels = GaussianMixture(n_components=k, covariance_type=params.get('covariance_type', 'full'),
                                      random_state=42, max_iter=200).fit_predict(X)
        elif method == 'OPTICS':
            labels = OPTICS(min_samples=params.get('min_samples', 5), cluster_method='xi',
                             xi=params.get('xi', 0.05)).fit_predict(X)
        return adjusted_rand_score(y, labels)
    except:
        return 0.0


def run_single_dataset(X, y, k, skip_optics=False):
    """Run all DR + clustering combos on a single dataset."""
    red_levels = get_reduction_levels(X.shape[1], k)
    
    # Find best params on original data
    ahc_params = find_best_ahc_params(X, y, k)
    gmm_params = find_best_gmm_params(X, y, k)
    optics_params = find_best_optics_params(X, y, k) if not skip_optics else {'min_samples': 5, 'xi': 0.05}
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
                except Exception as e:
                    dr_cache[(dr_name, level_name)] = None
    
    results = {}
    clustering_list = CLUSTERING_NAMES if not skip_optics else ['k-means', 'AHC', 'GMM']
    
    for cm_name in clustering_list:
        params = params_map[cm_name]
        results[cm_name] = {}
        
        results[cm_name]['No Reduction'] = cluster_single(X, y, k, cm_name, params)
        
        for dr_name in DR_METHOD_NAMES:
            for level_name in REDUCTION_LEVELS:
                cond = f'{dr_name}_{level_name}'
                X_r = dr_cache.get((dr_name, level_name))
                if X_r is None:
                    results[cm_name][cond] = 0.0
                else:
                    results[cm_name][cond] = cluster_single(X_r, y, k, cm_name, params)
    
    return results


def main():
    n_repeats = 3
    skip_d200 = True  # Skip d=200 for speed (MDS too slow)
    
    configs = get_all_configs(n_repeats=n_repeats)
    if skip_d200:
        configs = [c for c in configs if c['d'] != 200]
    
    results_path = os.path.join(RESULTS_DIR, 'synthetic_raw_v2.json')
    
    if os.path.exists(results_path):
        with open(results_path) as f:
            all_results = json.load(f)
    else:
        all_results = {}
    
    # Group by type
    by_type = {}
    for c in configs:
        t = c['type']
        if t not in by_type:
            by_type[t] = []
        by_type[t].append(c)
    
    total_start = time.time()
    
    for dtype in ['Circles', 'Moons', 'RSG', 'Repliclust']:
        if dtype not in all_results:
            all_results[dtype] = []
        
        existing = len(all_results[dtype])
        needed = len(by_type.get(dtype, []))
        
        if existing >= needed:
            print(f"\n{dtype}: {existing}/{needed} done, skipping")
            continue
        
        print(f"\n{'='*50}")
        print(f"{dtype}: need {needed - existing} more datasets")
        print(f"{'='*50}")
        
        for i, config in enumerate(by_type[dtype]):
            if i < existing:
                continue
            
            seed = config['repeat'] * 1000 + hash(f"{config['k']}_{config['d']}_{config.get('n_per_cluster', 0)}") % 10000
            seed = abs(seed) % (2**31)
            
            t0 = time.time()
            try:
                X, y, k = generate_dataset(config, seed)
                skip_optics = X.shape[0] < 30
                result = run_single_dataset(X, y, k, skip_optics=skip_optics)
                result['_config'] = {
                    'k': config['k'], 'd': config['d'],
                    'n_per_cluster': config.get('n_per_cluster', 0),
                    'repeat': config['repeat']
                }
                all_results[dtype].append(result)
                dt = time.time() - t0
                nr_km = result.get('k-means', {}).get('No Reduction', -1)
                print(f"  [{i+1}/{needed}] k={config['k']} d={config['d']} nc={config.get('n_per_cluster','')} "
                      f"n={X.shape[0]} → {dt:.1f}s  km_NR={nr_km:.3f}")
            except Exception as e:
                print(f"  [{i+1}/{needed}] ERROR: {e}")
                all_results[dtype].append({'_error': str(e), '_config': {
                    'k': config['k'], 'd': config['d'],
                    'n_per_cluster': config.get('n_per_cluster', 0),
                    'repeat': config['repeat']
                }})
            
            if (i + 1) % 5 == 0:
                with open(results_path, 'w') as f:
                    json.dump(all_results, f)
        
        with open(results_path, 'w') as f:
            json.dump(all_results, f)
    
    # Compute averages
    avg_results = {}
    for dtype in ['Circles', 'Moons', 'RSG', 'Repliclust']:
        avg_results[dtype] = {}
        valid = [r for r in all_results.get(dtype, []) if '_error' not in r]
        if not valid:
            continue
        for cm in CLUSTERING_NAMES:
            avg_results[dtype][cm] = {}
            cm_results = [r[cm] for r in valid if cm in r]
            if not cm_results:
                continue
            all_conds = set()
            for r in cm_results:
                all_conds.update(k for k in r.keys() if not k.startswith('_'))
            for cond in sorted(all_conds):
                vals = [r[cond] for r in cm_results if cond in r]
                if vals:
                    avg_results[dtype][cm][cond] = round(np.mean(vals), 3)
    
    with open(os.path.join(RESULTS_DIR, 'synthetic_avg_v2.json'), 'w') as f:
        json.dump(avg_results, f, indent=2)
    
    total_time = time.time() - total_start
    print(f"\n{'='*50}")
    print(f"TOTAL TIME: {total_time/60:.1f} min")
    print(f"{'='*50}")
    
    # Print summary table
    print("\nSummary (k-means No Reduction averages):")
    for dtype in avg_results:
        if 'k-means' in avg_results[dtype]:
            nr = avg_results[dtype]['k-means'].get('No Reduction', 'N/A')
            print(f"  {dtype}: {nr}")


if __name__ == '__main__':
    main()
