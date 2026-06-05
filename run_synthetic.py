"""Fast synthetic experiments with reduced MDS and OPTICS overhead."""
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
            return MDS(n_components=n_components, random_state=10, n_init=4, 
                       max_iter=100, normalized_stress='auto').fit_transform(X)
        except:
            return PCA(n_components=n_components).fit_transform(X)


def get_reduction_levels(n_features, n_clusters):
    k_minus_1 = max(n_clusters - 1, 2)
    pct_25 = max(int(np.round(0.25 * n_features)), 2)
    pct_50 = max(int(np.round(0.50 * n_features)), 2)
    return {'k-1': k_minus_1, '25%': pct_25, '50%': pct_50}


def run_kmeans(X, k): return KMeans(n_clusters=k, init='k-means++', n_init=100, random_state=42).fit_predict(X)

def run_ahc(X, k, affinity='euclidean', linkage='ward'):
    if linkage == 'ward': affinity = 'euclidean'
    try:
        return AgglomerativeClustering(n_clusters=k, metric=affinity if linkage != 'ward' else 'euclidean', linkage=linkage).fit_predict(X)
    except:
        return AgglomerativeClustering(n_clusters=k).fit_predict(X)

def run_gmm(X, k, covariance_type='full'):
    try:
        return GaussianMixture(n_components=k, covariance_type=covariance_type, random_state=42, max_iter=200).fit_predict(X)
    except:
        return GaussianMixture(n_components=k, covariance_type='diag', random_state=42).fit_predict(X)

def run_optics(X, k, min_samples=5, xi=0.05):
    try:
        return OPTICS(min_samples=min_samples, cluster_method='xi', xi=xi).fit_predict(X)
    except:
        return -np.ones(X.shape[0], dtype=int)


def find_best_ahc_params(X, y, k):
    best_ari, best_params = -2, {'affinity': 'euclidean', 'linkage': 'ward'}
    for linkage in ['complete', 'average', 'single', 'ward']:
        affs = ['euclidean'] if linkage == 'ward' else ['euclidean', 'manhattan', 'cosine']
        for aff in affs:
            try:
                labels = run_ahc(X, k, affinity=aff, linkage=linkage)
                ari = adjusted_rand_score(y, labels)
                if ari > best_ari: best_ari, best_params = ari, {'affinity': aff, 'linkage': linkage}
            except: pass
    return best_params

def find_best_gmm_params(X, y, k):
    best_ari, best_params = -2, {'covariance_type': 'full'}
    for ct in ['spherical', 'tied', 'diag', 'full']:
        try:
            labels = run_gmm(X, k, covariance_type=ct)
            ari = adjusted_rand_score(y, labels)
            if ari > best_ari: best_ari, best_params = ari, {'covariance_type': ct}
        except: pass
    return best_params

def find_best_optics_params(X, y, k):
    best_ari, best_params = -2, {'min_samples': 5, 'xi': 0.05}
    xi_values = np.arange(0.01, 1.01, 0.1)  # Coarser grid: 10 values instead of 20
    for ms in [5, 7, 10]:  # 3 values instead of 6
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


def cluster_and_evaluate(X, y, k, method, params=None):
    if params is None: params = {}
    if method == 'k-means':
        labels = run_kmeans(X, k)
    elif method == 'AHC':
        labels = run_ahc(X, k, **params)
    elif method == 'GMM':
        labels = run_gmm(X, k, **params)
    elif method == 'OPTICS':
        labels = run_optics(X, k, **params)
    return adjusted_rand_score(y, labels)


# Synthetic data generators
def generate_circles(n_samples=500, n_noise_dims=20, noise_std=0.05, n_circles=5, rng=None):
    if rng is None: rng = np.random.RandomState()
    spc = n_samples // n_circles
    X_list, y_list = [], []
    for i in range(n_circles):
        r = (i + 1) / n_circles
        theta = rng.uniform(0, 2*np.pi, spc)
        x1 = r * np.cos(theta) + rng.normal(0, noise_std, spc)
        x2 = r * np.sin(theta) + rng.normal(0, noise_std, spc)
        X_list.append(np.column_stack([x1, x2]))
        y_list.append(np.full(spc, i))
    X = np.vstack(X_list)
    noise = rng.normal(0, 1, (X.shape[0], n_noise_dims))
    return np.hstack([X, noise]), np.concatenate(y_list), n_circles

def generate_moons(n_samples=500, n_noise_dims=20, noise=0.1, rng=None):
    from sklearn.datasets import make_moons
    if rng is None: rng = np.random.RandomState()
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=rng.randint(100000))
    noise_dims = rng.normal(0, 1, (X.shape[0], n_noise_dims))
    return np.hstack([X, noise_dims]), y, 2

def generate_rsg(n_samples=500, n_features=20, n_clusters=5, rng=None):
    if rng is None: rng = np.random.RandomState()
    spc = n_samples // n_clusters
    X_list, y_list = [], []
    centers = rng.uniform(-10, 10, (n_clusters, n_features))
    for i in range(n_clusters):
        scale = rng.uniform(0.3, 1.5)
        X_list.append(rng.normal(centers[i], scale, (spc, n_features)))
        y_list.append(np.full(spc, i))
    return np.vstack(X_list), np.concatenate(y_list), n_clusters

def generate_repliclust(n_samples=500, n_features=20, n_clusters=5, rng=None):
    if rng is None: rng = np.random.RandomState()
    spc = n_samples // n_clusters
    X_list, y_list = [], []
    for i in range(n_clusters):
        center = rng.uniform(-5, 5, n_features)
        A = rng.randn(n_features, n_features) * 0.5
        cov = A @ A.T + np.eye(n_features) * 0.1
        X_list.append(rng.multivariate_normal(center, cov, spc))
        y_list.append(np.full(spc, i))
    return np.vstack(X_list), np.concatenate(y_list), n_clusters


GENERATORS = {
    'Circles': generate_circles,
    'Moons': generate_moons,
    'RSG': generate_rsg,
    'Repliclust': generate_repliclust,
}


def run_synthetic(n_repeats=10):
    print("=" * 70)
    print("SYNTHETIC EXPERIMENTS")
    print("=" * 70)
    
    # Check for existing results
    results_path = os.path.join(RESULTS_DIR, 'synthetic_raw_results.json')
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
        print("Loaded existing results")
    else:
        results = {}
    
    for dtype, gen_func in GENERATORS.items():
        if dtype in results and len(results[dtype].get(CLUSTERING_NAMES[0], {}).get('No Reduction', [])) >= n_repeats:
            print(f"\n--- {dtype}: already done, skipping ---")
            continue
        
        print(f"\n--- {dtype} ---")
        results[dtype] = {cm: {} for cm in CLUSTERING_NAMES}
        
        for repeat in range(n_repeats):
            seed = repeat * 42 + 7
            rng = np.random.RandomState(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            
            X, y, k = gen_func(rng=rng)
            X = StandardScaler().fit_transform(X)
            red_levels = get_reduction_levels(X.shape[1], k)
            
            # Find best params
            ahc_params = find_best_ahc_params(X, y, k)
            gmm_params = find_best_gmm_params(X, y, k)
            optics_params = find_best_optics_params(X, y, k)
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
                
                cond = 'No Reduction'
                if cond not in results[dtype][cm_name]:
                    results[dtype][cm_name][cond] = []
                ari = cluster_and_evaluate(X, y, k, cm_name, params)
                results[dtype][cm_name][cond].append(round(ari, 4))
                
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
                        results[dtype][cm_name][cond].append(round(ari, 4))
            
            print(f"  Repeat {repeat+1}/{n_repeats} done")
        
        # Save after each data type
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    # Save averaged results
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


if __name__ == '__main__':
    import sys
    n_repeats = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    t0 = time.time()
    results = run_synthetic(n_repeats)
    print(f"\nTotal time: {(time.time()-t0)/60:.1f} minutes")
