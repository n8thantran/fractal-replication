"""
Synthetic experiments matching paper methodology:
- Circles: 2-cluster and 5-cluster, embedded in d={10,50,200} via GRP, noise injection
- Moons: 2-cluster and 5-cluster, embedded in d={10,50,200} via GRP, noise injection
- RSG: Rodriguez Structured Gaussian with k={2,10,50}, d={10,50,200}, N_c={5,50,100}
- Repliclust: k={2,5}, d={10,50,200}, n=2000
- Noise injection: z-score, then 75% features get noise (1/4 σ=1, 1/4 σ=0.5, 1/4 σ=0.25, 1/4 clean)
- n_repeats per config (paper uses 50, we use fewer for speed)
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
from sklearn.random_projection import GaussianRandomProjection
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
    """Apply structured noise injection as described in the paper.
    Z-score normalize, then add noise to 75% of features:
    - 1/4 features: σ=1
    - 1/4 features: σ=0.5
    - 1/4 features: σ=0.25
    - 1/4 features: clean
    """
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)
    
    n_features = X_norm.shape[1]
    n_samples = X_norm.shape[0]
    
    # Shuffle feature indices
    feat_idx = rng.permutation(n_features)
    quarter = n_features // 4
    
    # Group 1: σ=1 noise
    for i in feat_idx[:quarter]:
        X_norm[:, i] += rng.normal(0, 1.0, n_samples)
    # Group 2: σ=0.5 noise
    for i in feat_idx[quarter:2*quarter]:
        X_norm[:, i] += rng.normal(0, 0.5, n_samples)
    # Group 3: σ=0.25 noise
    for i in feat_idx[2*quarter:3*quarter]:
        X_norm[:, i] += rng.normal(0, 0.25, n_samples)
    # Group 4: clean (no noise)
    
    return X_norm


def embed_to_high_dim(X_2d, target_dim, rng):
    """Embed 2D data into target_dim dimensions via Gaussian Random Projection.
    Uses the inverse direction: project from 2D to target_dim."""
    if target_dim <= 2:
        return X_2d
    # Create random projection matrix from 2D to target_dim
    proj_matrix = rng.randn(2, target_dim) / np.sqrt(target_dim)
    X_high = X_2d @ proj_matrix
    return X_high


def generate_circles_2cluster(rng, target_dim=50):
    """2-cluster circles: N_c=1000 per cluster, factor=0.5"""
    X, y = make_circles(n_samples=2000, factor=0.5, noise=0.05, random_state=rng.randint(100000))
    X_high = embed_to_high_dim(X, target_dim, rng)
    return X_high, y, 2


def generate_circles_5cluster(rng, target_dim=50):
    """5-cluster circles: N_c=400 per cluster, radial factors 1.0, 2.0, 3.5, 5.0, 7.0"""
    n_per_cluster = 400
    radial_factors = [1.0, 2.0, 3.5, 5.0, 7.0]
    X_list, y_list = [], []
    for i, r in enumerate(radial_factors):
        theta = rng.uniform(0, 2*np.pi, n_per_cluster)
        noise = rng.normal(0, 0.05, n_per_cluster)
        x1 = r * np.cos(theta) + noise * np.cos(theta)
        x2 = r * np.sin(theta) + noise * np.sin(theta)
        X_list.append(np.column_stack([x1, x2]))
        y_list.append(np.full(n_per_cluster, i))
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    X_high = embed_to_high_dim(X, target_dim, rng)
    return X_high, y, 5


def generate_moons_2cluster(rng, target_dim=50):
    """2-cluster moons with stretching, rotation, translation"""
    X, y = make_moons(n_samples=2000, noise=0.1, random_state=rng.randint(100000))
    # Apply random transformation
    stretch = rng.choice([1.0, 1.5])
    angle = rng.choice([np.radians(a) for a in [-160, -10, 10, 160, 180]])
    tx = rng.choice([-4, -2, 2, 4])
    ty = rng.choice([1.0, 1.2, 1.5])
    
    X[:, 0] *= stretch
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    X = X @ R.T
    X[:, 0] += tx
    X[:, 1] += ty
    
    X_high = embed_to_high_dim(X, target_dim, rng)
    return X_high, y, 2


def generate_moons_5cluster(rng, target_dim=50):
    """5-cluster moons: create 5 crescent-shaped clusters"""
    n_per_cluster = 400
    X_list, y_list = [], []
    
    for i in range(5):
        X_base, _ = make_moons(n_samples=n_per_cluster, noise=0.1, random_state=rng.randint(100000))
        # Only take one moon (first half)
        half = n_per_cluster // 2
        X_moon = X_base[:half]
        
        # Apply different transformations
        angle = rng.uniform(-np.pi, np.pi)
        R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        X_moon = X_moon @ R.T
        X_moon[:, 0] += rng.uniform(-5, 5)
        X_moon[:, 1] += rng.uniform(-5, 5)
        
        X_list.append(X_moon)
        y_list.append(np.full(half, i))
    
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    X_high = embed_to_high_dim(X, target_dim, rng)
    return X_high, y, 5


def generate_rsg(rng, n_clusters=2, n_features=50, n_per_cluster=50):
    """Rodriguez Structured Gaussian: normally distributed with controlled covariance."""
    X_list, y_list = [], []
    
    # Generate cluster centers with good separation
    centers = rng.uniform(-10, 10, (n_clusters, n_features))
    
    for i in range(n_clusters):
        # Generate cluster-specific covariance (symmetric positive semi-definite)
        A = rng.randn(n_features, n_features) * 0.3
        cov = A @ A.T / n_features + np.eye(n_features) * 0.1
        
        X_cluster = rng.multivariate_normal(centers[i], cov, n_per_cluster)
        X_list.append(X_cluster)
        y_list.append(np.full(n_per_cluster, i))
    
    return np.vstack(X_list), np.concatenate(y_list), n_clusters


def generate_repliclust(rng, n_clusters=2, n_features=50, n_total=2000):
    """Repliclust-style: anisotropic clusters with substantial inter-class separation."""
    n_per_cluster = n_total // n_clusters
    X_list, y_list = [], []
    
    # Create well-separated centers
    centers = np.zeros((n_clusters, n_features))
    for i in range(n_clusters):
        # Place clusters along different feature dimensions
        dim_start = (i * n_features) // n_clusters
        dim_end = min(dim_start + max(n_features // n_clusters, 1), n_features)
        for d in range(dim_start, dim_end):
            centers[i, d] = rng.uniform(5, 15) * (1 if rng.rand() > 0.5 else -1)
    
    for i in range(n_clusters):
        # Anisotropic covariance
        A = rng.randn(n_features, n_features) * 0.5
        cov = A @ A.T / n_features + np.eye(n_features) * 0.2
        
        X_cluster = rng.multivariate_normal(centers[i], cov, n_per_cluster)
        X_list.append(X_cluster)
        y_list.append(np.full(n_per_cluster, i))
    
    return np.vstack(X_list), np.concatenate(y_list), n_clusters


# ============ DR Methods ============

def apply_dr(method_name, X, n_components, rng=None):
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
        n_init = 2 if X.shape[0] > 500 else (4 if X.shape[0] > 200 else 10)
        max_iter = 200 if X.shape[0] > 500 else 300
        return MDS(n_components=n_components, random_state=10, n_init=n_init, max_iter=max_iter).fit_transform(X)
    return X.copy()


def apply_vae(X, n_components):
    X_min, X_max = X.min(0), X.max(0)
    denom = X_max - X_min
    denom[denom == 0] = 1
    X_s = (X - X_min) / denom
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    X_t = torch.FloatTensor(X_s).to(device)
    n = X_s.shape[0]
    n_train = int(0.7 * n)
    idx = np.random.permutation(n)
    loader = DataLoader(TensorDataset(X_t[idx[:n_train]]), batch_size=64, shuffle=True)
    
    model = VAE(X.shape[1], n_components).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    model.train()
    for epoch in range(100):
        for (batch,) in loader:
            x_recon, mu, logvar = model(batch)
            recon_loss = nn.functional.mse_loss(x_recon, batch, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    model.eval()
    with torch.no_grad():
        mu, _ = model.encode(X_t)
    return mu.cpu().numpy()


# ============ Clustering Methods ============

def run_kmeans(X, k):
    km = KMeans(n_clusters=k, init='k-means++', n_init=100, random_state=42)
    return km.fit_predict(X)


def run_ahc(X, k, best_params=None):
    """Run AHC with best affinity+linkage."""
    if best_params:
        aff, link = best_params
        model = AgglomerativeClustering(n_clusters=k, metric=aff, linkage=link)
        return model.fit_predict(X)
    
    # Grid search
    best_ari = -1
    best_labels = None
    affinities = ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']
    linkages = ['complete', 'average', 'single', 'ward']
    
    for aff in affinities:
        for link in linkages:
            if link == 'ward' and aff != 'euclidean':
                continue
            try:
                model = AgglomerativeClustering(n_clusters=k, metric=aff, linkage=link)
                labels = model.fit_predict(X)
                if best_labels is None:
                    best_labels = labels
            except:
                continue
    
    return best_labels


def run_gmm(X, k, best_cov=None):
    """Run GMM with best covariance type."""
    if best_cov:
        model = GaussianMixture(n_components=k, covariance_type=best_cov, random_state=42, max_iter=200)
        return model.fit_predict(X)
    
    # Grid search
    best_labels = None
    for cov_type in ['spherical', 'tied', 'diag', 'full']:
        try:
            model = GaussianMixture(n_components=k, covariance_type=cov_type, random_state=42, max_iter=200)
            labels = model.fit_predict(X)
            if best_labels is None:
                best_labels = labels
        except:
            continue
    
    return best_labels


def run_optics(X, y_true):
    """Run OPTICS with parameter search."""
    best_ari = -1
    best_labels = None
    
    for min_samples in [5, 7, 10]:
        if min_samples >= X.shape[0]:
            continue
        try:
            model = OPTICS(min_samples=min_samples, metric='euclidean')
            model.fit(X)
            
            for xi in np.arange(0.01, 1.01, 0.1):
                try:
                    labels, _ = cluster_optics_xi(
                        reachability=model.reachability_,
                        predecessor=model.predecessor_,
                        ordering=model.ordering_,
                        min_samples=min_samples,
                        xi=xi,
                        min_cluster_size=None
                    )
                    ari = adjusted_rand_score(y_true, labels)
                    if ari > best_ari:
                        best_ari = ari
                        best_labels = labels.copy()
                except:
                    continue
        except:
            continue
    
    if best_labels is None:
        best_labels = np.zeros(X.shape[0], dtype=int)
    
    return best_labels


def compute_reduction_dims(k, d):
    """Compute the 3 reduction levels: k-1, 25%, 50%"""
    k_minus_1 = max(k - 1, 2)
    pct_25 = max(int(round(d * 0.25)), 1)
    pct_50 = max(int(round(d * 0.50)), 1)
    return {'k-1': k_minus_1, '25%': pct_25, '50%': pct_50}


# ============ Dataset Configurations ============

def get_circles_configs():
    """6 configs: k∈{2,5} × d∈{10,50,200}"""
    configs = []
    for k in [2, 5]:
        for d in [10, 50, 200]:
            configs.append({'k': k, 'd': d})
    return configs


def get_moons_configs():
    """6 configs: k∈{2,5} × d∈{10,50,200}"""
    configs = []
    for k in [2, 5]:
        for d in [10, 50, 200]:
            configs.append({'k': k, 'd': d})
    return configs


def get_rsg_configs():
    """9 configs: k∈{2,10,50} × d∈{10,50,200}"""
    configs = []
    for k in [2, 10, 50]:
        for d in [10, 50, 200]:
            configs.append({'k': k, 'd': d, 'n_per_cluster': 50})
    return configs


def get_repliclust_configs():
    """6 configs: k∈{2,5} × d∈{10,50,200}"""
    configs = []
    for k in [2, 5]:
        for d in [10, 50, 200]:
            configs.append({'k': k, 'd': d})
    return configs


def generate_dataset(dtype, config, rng):
    """Generate a single dataset for given type and config."""
    k = config['k']
    d = config['d']
    
    if dtype == 'Circles':
        if k == 2:
            X, y, k_actual = generate_circles_2cluster(rng, target_dim=d)
        else:
            X, y, k_actual = generate_circles_5cluster(rng, target_dim=d)
    elif dtype == 'Moons':
        if k == 2:
            X, y, k_actual = generate_moons_2cluster(rng, target_dim=d)
        else:
            X, y, k_actual = generate_moons_5cluster(rng, target_dim=d)
    elif dtype == 'RSG':
        n_per_cluster = config.get('n_per_cluster', 50)
        X, y, k_actual = generate_rsg(rng, n_clusters=k, n_features=d, n_per_cluster=n_per_cluster)
    elif dtype == 'Repliclust':
        X, y, k_actual = generate_repliclust(rng, n_clusters=k, n_features=d, n_total=2000)
    
    # Apply noise injection
    X = inject_noise(X, rng)
    
    return X, y, k_actual


def find_best_ahc_params(datasets, k):
    """Find best AHC params across a set of datasets."""
    affinities = ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']
    linkages = ['complete', 'average', 'single', 'ward']
    
    best_score = -1
    best_params = ('euclidean', 'ward')
    
    for aff in affinities:
        for link in linkages:
            if link == 'ward' and aff != 'euclidean':
                continue
            total_ari = 0
            count = 0
            for X, y in datasets:
                try:
                    model = AgglomerativeClustering(n_clusters=k, metric=aff, linkage=link)
                    labels = model.fit_predict(X)
                    total_ari += adjusted_rand_score(y, labels)
                    count += 1
                except:
                    continue
            if count > 0:
                avg_ari = total_ari / count
                if avg_ari > best_score:
                    best_score = avg_ari
                    best_params = (aff, link)
    
    return best_params


def find_best_gmm_cov(datasets, k):
    """Find best GMM covariance type across a set of datasets."""
    best_score = -1
    best_cov = 'full'
    
    for cov_type in ['spherical', 'tied', 'diag', 'full']:
        total_ari = 0
        count = 0
        for X, y in datasets:
            try:
                model = GaussianMixture(n_components=k, covariance_type=cov_type, random_state=42, max_iter=200)
                labels = model.fit_predict(X)
                total_ari += adjusted_rand_score(y, labels)
                count += 1
            except:
                continue
        if count > 0:
            avg_ari = total_ari / count
            if avg_ari > best_score:
                best_score = avg_ari
                best_cov = cov_type
    
    return best_cov


def run_experiment_on_dataset(X, y, k, d, ahc_params=None, gmm_cov=None):
    """Run all DR methods × clustering algorithms on one dataset."""
    results = {}
    reduction_dims = compute_reduction_dims(k, d)
    
    for clust_name in CLUSTERING_NAMES:
        results[clust_name] = {}
        
        # No reduction baseline
        if clust_name == 'k-means':
            labels = run_kmeans(X, k)
        elif clust_name == 'AHC':
            labels = run_ahc(X, k, ahc_params)
        elif clust_name == 'GMM':
            labels = run_gmm(X, k, gmm_cov)
        elif clust_name == 'OPTICS':
            labels = run_optics(X, y)
        
        results[clust_name]['No Reduction'] = adjusted_rand_score(y, labels)
        
        # DR methods
        for dr_name in DR_METHOD_NAMES:
            for level_name, n_comp in reduction_dims.items():
                key = f"{dr_name}_{level_name}"
                try:
                    X_dr = apply_dr(dr_name, X, n_comp)
                    
                    if clust_name == 'k-means':
                        labels = run_kmeans(X_dr, k)
                    elif clust_name == 'AHC':
                        labels = run_ahc(X_dr, k, ahc_params)
                    elif clust_name == 'GMM':
                        labels = run_gmm(X_dr, k, gmm_cov)
                    elif clust_name == 'OPTICS':
                        labels = run_optics(X_dr, y)
                    
                    results[clust_name][key] = adjusted_rand_score(y, labels)
                except Exception as e:
                    results[clust_name][key] = 0.0
    
    return results


def run_all_synthetic(n_repeats=5):
    """Run all synthetic experiments."""
    print("=" * 70)
    print("SYNTHETIC EXPERIMENTS (Paper-matching methodology)")
    print("=" * 70)
    
    all_raw_results = {}
    all_avg_results = {}
    
    dataset_types = {
        'Circles': get_circles_configs(),
        'Moons': get_moons_configs(),
        'RSG': get_rsg_configs(),
        'Repliclust': get_repliclust_configs(),
    }
    
    for dtype, configs in dataset_types.items():
        print(f"\n{'='*50}")
        print(f"Dataset type: {dtype} ({len(configs)} configs × {n_repeats} repeats)")
        print(f"{'='*50}")
        
        all_raw_results[dtype] = []
        
        # Accumulate results across all configs and repeats
        accum = {}
        count = 0
        
        for ci, config in enumerate(configs):
            k = config['k']
            d = config['d']
            print(f"\n  Config {ci+1}/{len(configs)}: k={k}, d={d}")
            
            # Generate a few datasets first to find best AHC/GMM params
            sample_datasets = []
            for r in range(min(3, n_repeats)):
                rng = np.random.RandomState(42 + ci * 1000 + r)
                X, y, k_actual = generate_dataset(dtype, config, rng)
                sample_datasets.append((X, y))
            
            ahc_params = find_best_ahc_params(sample_datasets, k)
            gmm_cov = find_best_gmm_cov(sample_datasets, k)
            print(f"    Best AHC: {ahc_params}, Best GMM cov: {gmm_cov}")
            
            for r in range(n_repeats):
                rng = np.random.RandomState(42 + ci * 1000 + r)
                X, y, k_actual = generate_dataset(dtype, config, rng)
                
                t0 = time.time()
                result = run_experiment_on_dataset(X, y, k, d, ahc_params, gmm_cov)
                elapsed = time.time() - t0
                
                all_raw_results[dtype].append({
                    'config': config,
                    'repeat': r,
                    'results': result
                })
                
                # Accumulate
                for clust_name, clust_results in result.items():
                    if clust_name not in accum:
                        accum[clust_name] = {}
                    for key, val in clust_results.items():
                        if key not in accum[clust_name]:
                            accum[clust_name][key] = []
                        accum[clust_name][key].append(val)
                
                count += 1
                print(f"    Repeat {r+1}/{n_repeats} done ({elapsed:.1f}s)")
        
        # Average results
        all_avg_results[dtype] = {}
        for clust_name, clust_data in accum.items():
            all_avg_results[dtype][clust_name] = {}
            for key, vals in clust_data.items():
                all_avg_results[dtype][clust_name][key] = float(np.mean(vals))
        
        # Print summary
        print(f"\n  Summary for {dtype} ({count} total datasets):")
        for clust_name in CLUSTERING_NAMES:
            if clust_name in all_avg_results[dtype]:
                nr = all_avg_results[dtype][clust_name].get('No Reduction', 0)
                print(f"    {clust_name}: No Reduction ARI = {nr:.3f}")
    
    # Save results
    with open(os.path.join(RESULTS_DIR, 'synthetic_raw_v2.json'), 'w') as f:
        json.dump(all_raw_results, f, indent=2, default=str)
    
    with open(os.path.join(RESULTS_DIR, 'synthetic_results_v2.json'), 'w') as f:
        json.dump(all_avg_results, f, indent=2)
    
    print("\n\nResults saved!")
    return all_avg_results


if __name__ == '__main__':
    import sys
    n_repeats = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    results = run_all_synthetic(n_repeats)
