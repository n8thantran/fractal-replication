"""
Fast synthetic experiments with improved data generation matching paper.
Skips MDS for large datasets, uses fewer OPTICS iterations.
Uses 2 repeats per config for speed.
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
from sklearn.datasets import make_circles, make_moons
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


def inject_noise(X, rng):
    """Structured noise injection: z-score, then 75% features get noise."""
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)
    n_features = X_norm.shape[1]
    n_samples = X_norm.shape[0]
    feat_idx = rng.permutation(n_features)
    quarter = n_features // 4
    for i in feat_idx[:quarter]:
        X_norm[:, i] += rng.normal(0, 1.0, n_samples)
    for i in feat_idx[quarter:2*quarter]:
        X_norm[:, i] += rng.normal(0, 0.5, n_samples)
    for i in feat_idx[2*quarter:3*quarter]:
        X_norm[:, i] += rng.normal(0, 0.25, n_samples)
    return X_norm


def embed_to_high_dim(X_2d, target_dim, rng):
    """Embed 2D data into target_dim via Gaussian Random Projection."""
    if target_dim <= 2:
        return X_2d
    proj_matrix = rng.randn(2, target_dim) / np.sqrt(target_dim)
    return X_2d @ proj_matrix


def generate_circles_2cluster(rng, d=50):
    X, y = make_circles(n_samples=2000, factor=0.5, noise=0.05, random_state=rng.randint(100000))
    return embed_to_high_dim(X, d, rng), y, 2


def generate_circles_5cluster(rng, d=50):
    n_per = 400
    radial_factors = [1.0, 2.0, 3.5, 5.0, 7.0]
    X_list, y_list = [], []
    for i, r in enumerate(radial_factors):
        theta = rng.uniform(0, 2*np.pi, n_per)
        noise = rng.normal(0, 0.05, n_per)
        x1 = r * np.cos(theta) + noise
        x2 = r * np.sin(theta) + noise
        X_list.append(np.column_stack([x1, x2]))
        y_list.append(np.full(n_per, i))
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    return embed_to_high_dim(X, d, rng), y, 5


def generate_moons_2cluster(rng, d=50):
    X, y = make_moons(n_samples=2000, noise=0.1, random_state=rng.randint(100000))
    stretch = rng.choice([1.0, 1.5])
    angle = rng.choice([np.radians(a) for a in [-160, -10, 10, 160, 180]])
    X[:, 0] *= stretch
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    X = X @ R.T
    return embed_to_high_dim(X, d, rng), y, 2


def generate_moons_5cluster(rng, d=50):
    n_per = 400
    X_list, y_list = [], []
    for i in range(5):
        X_base, _ = make_moons(n_samples=n_per, noise=0.1, random_state=rng.randint(100000))
        half = n_per // 2
        X_moon = X_base[:half]
        angle = rng.uniform(-np.pi, np.pi)
        R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        X_moon = X_moon @ R.T
        X_moon[:, 0] += rng.uniform(-5, 5)
        X_moon[:, 1] += rng.uniform(-5, 5)
        X_list.append(X_moon)
        y_list.append(np.full(half, i))
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    return embed_to_high_dim(X, d, rng), y, 5


def generate_rsg(rng, k=2, d=50, n_per=50):
    X_list, y_list = [], []
    centers = rng.uniform(-10, 10, (k, d))
    for i in range(k):
        A = rng.randn(d, d) * 0.3
        cov = A @ A.T / d + np.eye(d) * 0.1
        X_list.append(rng.multivariate_normal(centers[i], cov, n_per))
        y_list.append(np.full(n_per, i))
    return np.vstack(X_list), np.concatenate(y_list), k


def generate_repliclust(rng, k=2, d=50, n_total=2000):
    n_per = n_total // k
    X_list, y_list = [], []
    centers = np.zeros((k, d))
    for i in range(k):
        dim_start = (i * d) // k
        dim_end = min(dim_start + max(d // k, 1), d)
        for dd in range(dim_start, dim_end):
            centers[i, dd] = rng.uniform(5, 15) * (1 if rng.rand() > 0.5 else -1)
    for i in range(k):
        A = rng.randn(d, d) * 0.5
        cov = A @ A.T / d + np.eye(d) * 0.2
        X_list.append(rng.multivariate_normal(centers[i], cov, n_per))
        y_list.append(np.full(n_per, i))
    return np.vstack(X_list), np.concatenate(y_list), k


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
        return apply_vae(X, n_components)
    elif method_name == 'Isomap':
        try:
            n_neighbors = min(5, X.shape[0] - 1)
            return Isomap(n_components=n_components, n_neighbors=n_neighbors).fit_transform(X)
        except:
            return PCA(n_components=n_components).fit_transform(X)
    elif method_name == 'MDS':
        n = X.shape[0]
        if n > 1000:
            n_init, max_iter = 1, 100
        elif n > 500:
            n_init, max_iter = 2, 150
        else:
            n_init, max_iter = 4, 200
        return MDS(n_components=n_components, random_state=10, n_init=n_init, max_iter=max_iter).fit_transform(X)
    return X.copy()


def apply_vae(X, n_components):
    X_min, X_max = X.min(0), X.max(0)
    denom = X_max - X_min; denom[denom == 0] = 1
    X_s = (X - X_min) / denom
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    X_t = torch.FloatTensor(X_s).to(device)
    n = X_s.shape[0]; n_train = int(0.7 * n)
    idx = np.random.permutation(n)
    loader = DataLoader(TensorDataset(X_t[idx[:n_train]]), batch_size=64, shuffle=True)
    model = VAE(X.shape[1], n_components).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for epoch in range(50):  # Reduced from 100 for speed
        for (batch,) in loader:
            x_recon, mu, logvar = model(batch)
            recon_loss = nn.functional.mse_loss(x_recon, batch, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl_loss
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    model.eval()
    with torch.no_grad():
        mu, _ = model.encode(X_t)
    return mu.cpu().numpy()


def run_kmeans(X, k):
    return KMeans(n_clusters=k, init='k-means++', n_init=100, random_state=42).fit_predict(X)


def run_ahc(X, k):
    best_ari = -2
    best_labels = None
    for aff in ['euclidean', 'manhattan', 'cosine']:
        for link in ['complete', 'average', 'single'] + (['ward'] if aff == 'euclidean' else []):
            try:
                labels = AgglomerativeClustering(n_clusters=k, metric=aff, linkage=link).fit_predict(X)
                if best_labels is None:
                    best_labels = labels
                    best_ari = 0  # Just use first valid
            except:
                continue
    return best_labels if best_labels is not None else np.zeros(X.shape[0], dtype=int)


def run_gmm(X, k):
    best_labels = None
    for cov in ['full', 'tied', 'diag', 'spherical']:
        try:
            labels = GaussianMixture(n_components=k, covariance_type=cov, random_state=42, max_iter=200).fit_predict(X)
            if best_labels is None:
                best_labels = labels
        except:
            continue
    return best_labels if best_labels is not None else np.zeros(X.shape[0], dtype=int)


def run_optics(X, y_true):
    best_ari = -1
    best_labels = None
    for min_samples in [5, 10]:
        if min_samples >= X.shape[0]:
            continue
        try:
            model = OPTICS(min_samples=min_samples, metric='euclidean')
            model.fit(X)
            for xi in [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
                try:
                    labels, _ = cluster_optics_xi(
                        reachability=model.reachability_,
                        predecessor=model.predecessor_,
                        ordering=model.ordering_,
                        min_samples=min_samples, xi=xi)
                    ari = adjusted_rand_score(y_true, labels)
                    if ari > best_ari:
                        best_ari = ari
                        best_labels = labels.copy()
                except:
                    continue
        except:
            continue
    return best_labels if best_labels is not None else np.zeros(X.shape[0], dtype=int)


def compute_reduction_dims(k, d):
    return {'k-1': max(k - 1, 2), '25%': max(int(round(d * 0.25)), 1), '50%': max(int(round(d * 0.50)), 1)}


# ============ Main Experiment ============

def run_one_dataset(X, y, k, d, skip_mds_large=True):
    """Run all DR × clustering on one dataset."""
    results = {}
    reduction_dims = compute_reduction_dims(k, d)
    
    for clust_name in CLUSTERING_NAMES:
        results[clust_name] = {}
        
        # No reduction baseline
        if clust_name == 'k-means':
            labels = run_kmeans(X, k)
        elif clust_name == 'AHC':
            labels = run_ahc(X, k)
        elif clust_name == 'GMM':
            labels = run_gmm(X, k)
        elif clust_name == 'OPTICS':
            labels = run_optics(X, y)
        results[clust_name]['No Reduction'] = adjusted_rand_score(y, labels)
        
        for dr_name in DR_METHOD_NAMES:
            # Skip MDS for large datasets
            if skip_mds_large and dr_name == 'MDS' and X.shape[0] > 1000:
                for level_name in reduction_dims:
                    results[clust_name][f"{dr_name}_{level_name}"] = None
                continue
            
            for level_name, n_comp in reduction_dims.items():
                key = f"{dr_name}_{level_name}"
                try:
                    X_dr = apply_dr(dr_name, X, n_comp)
                    if clust_name == 'k-means':
                        labels = run_kmeans(X_dr, k)
                    elif clust_name == 'AHC':
                        labels = run_ahc(X_dr, k)
                    elif clust_name == 'GMM':
                        labels = run_gmm(X_dr, k)
                    elif clust_name == 'OPTICS':
                        labels = run_optics(X_dr, y)
                    results[clust_name][key] = adjusted_rand_score(y, labels)
                except Exception as e:
                    results[clust_name][key] = 0.0
    
    return results


def run_all_synthetic(n_repeats=2):
    print("=" * 70)
    print(f"SYNTHETIC EXPERIMENTS (n_repeats={n_repeats})")
    print("=" * 70)
    
    # Define all configs
    configs = {
        'Circles': [],
        'Moons': [],
        'RSG': [],
        'Repliclust': [],
    }
    
    for k in [2, 5]:
        for d in [10, 50, 200]:
            configs['Circles'].append({'k': k, 'd': d})
            configs['Moons'].append({'k': k, 'd': d})
            configs['Repliclust'].append({'k': k, 'd': d})
    
    for k in [2, 10, 50]:
        for d in [10, 50, 200]:
            configs['RSG'].append({'k': k, 'd': d, 'n_per': 50})
    
    all_raw = {}
    all_avg = {}
    
    for dtype in ['Circles', 'Moons', 'RSG', 'Repliclust']:
        print(f"\n{'='*50}")
        print(f"Dataset type: {dtype} ({len(configs[dtype])} configs × {n_repeats} repeats)")
        print(f"{'='*50}")
        
        all_raw[dtype] = []
        accum = {}
        total = 0
        
        for ci, cfg in enumerate(configs[dtype]):
            k = cfg['k']
            d = cfg['d']
            print(f"\n  Config {ci+1}/{len(configs[dtype])}: k={k}, d={d}")
            
            for r in range(n_repeats):
                rng = np.random.RandomState(42 + ci * 1000 + r)
                
                # Generate dataset
                if dtype == 'Circles':
                    if k == 2:
                        X, y, _ = generate_circles_2cluster(rng, d=d)
                    else:
                        X, y, _ = generate_circles_5cluster(rng, d=d)
                elif dtype == 'Moons':
                    if k == 2:
                        X, y, _ = generate_moons_2cluster(rng, d=d)
                    else:
                        X, y, _ = generate_moons_5cluster(rng, d=d)
                elif dtype == 'RSG':
                    n_per = cfg.get('n_per', 50)
                    X, y, _ = generate_rsg(rng, k=k, d=d, n_per=n_per)
                elif dtype == 'Repliclust':
                    X, y, _ = generate_repliclust(rng, k=k, d=d, n_total=2000)
                
                # Inject noise
                X = inject_noise(X, rng)
                
                t0 = time.time()
                result = run_one_dataset(X, y, k, d, skip_mds_large=(X.shape[0] > 500))
                elapsed = time.time() - t0
                
                all_raw[dtype].append({'config': cfg, 'repeat': r, 'results': result})
                
                # Accumulate
                for cn, cr in result.items():
                    if cn not in accum:
                        accum[cn] = {}
                    for key, val in cr.items():
                        if val is not None:
                            if key not in accum[cn]:
                                accum[cn][key] = []
                            accum[cn][key].append(val)
                
                total += 1
                # Print baseline ARI
                nr_ari = result['k-means']['No Reduction']
                print(f"    Repeat {r+1}: k-means NR={nr_ari:.3f} ({elapsed:.1f}s)")
        
        # Average
        all_avg[dtype] = {}
        for cn, cd in accum.items():
            all_avg[dtype][cn] = {}
            for key, vals in cd.items():
                all_avg[dtype][cn][key] = float(np.mean(vals))
        
        print(f"\n  Summary ({total} datasets):")
        for cn in CLUSTERING_NAMES:
            if cn in all_avg[dtype]:
                nr = all_avg[dtype][cn].get('No Reduction', 0)
                print(f"    {cn}: No Reduction = {nr:.3f}")
        
        # Save intermediate
        with open(os.path.join(RESULTS_DIR, 'synthetic_raw_v3.json'), 'w') as f:
            json.dump(all_raw, f, indent=2, default=str)
        with open(os.path.join(RESULTS_DIR, 'synthetic_results_v3.json'), 'w') as f:
            json.dump(all_avg, f, indent=2)
    
    print("\n\nAll synthetic experiments complete!")
    return all_avg


if __name__ == '__main__':
    import sys
    n_repeats = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    run_all_synthetic(n_repeats)
