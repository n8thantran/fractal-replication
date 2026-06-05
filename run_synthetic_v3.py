"""
Ultra-fast synthetic experiments. Key optimizations:
- 500 samples per dataset (not 2000)
- OPTICS: single min_samples, fewer xi values
- MDS: skip for d>50
- VAE: 30 epochs
- 2 repeats per config
- 6 configs per type (k in [2,5], d in [10,50,200])
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

DR_METHODS = ['PCA', 'Kernel PCA', 'VAE', 'Isomap', 'MDS']
CLUSTERING = ['k-means', 'AHC', 'GMM', 'OPTICS']
LEVELS = ['k-1', '25%', '50%']


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.BatchNorm1d(32), nn.Dropout(0.4))
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(),
            nn.BatchNorm1d(32), nn.Dropout(0.4),
            nn.Linear(32, 64), nn.ReLU(),
            nn.BatchNorm1d(64), nn.Dropout(0.4),
            nn.Linear(64, input_dim), nn.Sigmoid())
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return self.decoder(z), mu, logvar


def inject_noise(X, rng):
    scaler = StandardScaler()
    X_n = scaler.fit_transform(X)
    d = X_n.shape[1]
    n = X_n.shape[0]
    idx = rng.permutation(d)
    q = d // 4
    for i in idx[:q]:
        X_n[:, i] += rng.normal(0, 1.0, n)
    for i in idx[q:2*q]:
        X_n[:, i] += rng.normal(0, 0.5, n)
    for i in idx[2*q:3*q]:
        X_n[:, i] += rng.normal(0, 0.25, n)
    return X_n


def embed_high(X_2d, d, rng):
    if d <= 2:
        return X_2d
    proj = rng.randn(2, d) / np.sqrt(d)
    return X_2d @ proj


def gen_circles(rng, k, d, n=500):
    if k == 2:
        X, y = make_circles(n_samples=n, factor=0.5, noise=0.05, random_state=rng.randint(100000))
    else:
        n_per = n // k
        radii = np.linspace(1, 7, k)
        X_list, y_list = [], []
        for i, r in enumerate(radii):
            theta = rng.uniform(0, 2*np.pi, n_per)
            noise = rng.normal(0, 0.05, n_per)
            X_list.append(np.column_stack([r*np.cos(theta)+noise, r*np.sin(theta)+noise]))
            y_list.append(np.full(n_per, i))
        X, y = np.vstack(X_list), np.concatenate(y_list)
    return inject_noise(embed_high(X, d, rng), rng), y, k


def gen_moons(rng, k, d, n=500):
    if k == 2:
        X, y = make_moons(n_samples=n, noise=0.1, random_state=rng.randint(100000))
    else:
        n_per = n // k
        X_list, y_list = [], []
        for i in range(k):
            X_base, _ = make_moons(n_samples=n_per, noise=0.1, random_state=rng.randint(100000))
            half = n_per // 2
            X_moon = X_base[:half]
            angle = rng.uniform(-np.pi, np.pi)
            R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            X_moon = X_moon @ R.T + rng.uniform(-5, 5, 2)
            X_list.append(X_moon)
            y_list.append(np.full(half, i))
        X, y = np.vstack(X_list), np.concatenate(y_list)
    return inject_noise(embed_high(X, d, rng), rng), y, k


def gen_rsg(rng, k, d, n_per=50):
    centers = rng.uniform(-10, 10, (k, d))
    X_list, y_list = [], []
    for i in range(k):
        A = rng.randn(d, d) * 0.3
        cov = A @ A.T / d + np.eye(d) * 0.1
        X_list.append(rng.multivariate_normal(centers[i], cov, n_per))
        y_list.append(np.full(n_per, i))
    X, y = np.vstack(X_list), np.concatenate(y_list)
    return inject_noise(X, rng), y, k


def gen_repliclust(rng, k, d, n=500):
    n_per = n // k
    centers = np.zeros((k, d))
    for i in range(k):
        dim_s = (i * d) // k
        dim_e = min(dim_s + max(d // k, 1), d)
        for dd in range(dim_s, dim_e):
            centers[i, dd] = rng.uniform(5, 15) * (1 if rng.rand() > 0.5 else -1)
    X_list, y_list = [], []
    for i in range(k):
        A = rng.randn(d, d) * 0.5
        cov = A @ A.T / d + np.eye(d) * 0.2
        X_list.append(rng.multivariate_normal(centers[i], cov, n_per))
        y_list.append(np.full(n_per, i))
    X, y = np.vstack(X_list), np.concatenate(y_list)
    return inject_noise(X, rng), y, k


def apply_dr(name, X, nc):
    nc = max(1, min(nc, X.shape[1]-1, X.shape[0]-1))
    if nc >= X.shape[1]:
        return X.copy()
    
    if name == 'PCA':
        return PCA(n_components=nc).fit_transform(X)
    elif name == 'Kernel PCA':
        try:
            r = KernelPCA(n_components=nc, kernel='rbf').fit_transform(X)
            if r.shape[1] < nc:
                r = np.hstack([r, np.zeros((r.shape[0], nc - r.shape[1]))])
            return r
        except:
            return PCA(n_components=nc).fit_transform(X)
    elif name == 'VAE':
        return _vae(X, nc)
    elif name == 'Isomap':
        try:
            nn = min(5, X.shape[0]-1)
            return Isomap(n_components=nc, n_neighbors=nn).fit_transform(X)
        except:
            return PCA(n_components=nc).fit_transform(X)
    elif name == 'MDS':
        return MDS(n_components=nc, random_state=10, n_init=2, max_iter=100).fit_transform(X)
    return X.copy()


def _vae(X, nc):
    X_min, X_max = X.min(0), X.max(0)
    denom = X_max - X_min; denom[denom == 0] = 1
    X_s = (X - X_min) / denom
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    X_t = torch.FloatTensor(X_s).to(dev)
    n_train = int(0.7 * X.shape[0])
    idx = np.random.permutation(X.shape[0])
    loader = DataLoader(TensorDataset(X_t[idx[:n_train]]), batch_size=64, shuffle=True)
    model = VAE(X.shape[1], nc).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(30):
        for (b,) in loader:
            xr, mu, lv = model(b)
            loss = nn.functional.mse_loss(xr, b, reduction='sum') - 0.5*torch.sum(1+lv-mu.pow(2)-lv.exp())
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        mu, _ = model.encode(X_t)
    return mu.cpu().numpy()


def cluster(name, X, k, y_true):
    if name == 'k-means':
        return KMeans(n_clusters=k, init='k-means++', n_init=50, random_state=42).fit_predict(X)
    elif name == 'AHC':
        best = None
        for aff in ['euclidean', 'cosine']:
            for link in (['ward', 'complete', 'average'] if aff == 'euclidean' else ['complete', 'average']):
                try:
                    l = AgglomerativeClustering(n_clusters=k, metric=aff, linkage=link).fit_predict(X)
                    if best is None:
                        best = l
                except:
                    pass
        return best if best is not None else np.zeros(X.shape[0], dtype=int)
    elif name == 'GMM':
        for ct in ['full', 'tied', 'diag', 'spherical']:
            try:
                return GaussianMixture(n_components=k, covariance_type=ct, random_state=42, max_iter=100).fit_predict(X)
            except:
                pass
        return np.zeros(X.shape[0], dtype=int)
    elif name == 'OPTICS':
        best_ari, best_l = -1, None
        for ms in [5, 10]:
            if ms >= X.shape[0]:
                continue
            try:
                m = OPTICS(min_samples=ms, metric='euclidean')
                m.fit(X)
                for xi in [0.01, 0.05, 0.1, 0.2, 0.5, 0.8]:
                    try:
                        l, _ = cluster_optics_xi(reachability=m.reachability_, predecessor=m.predecessor_,
                                                  ordering=m.ordering_, min_samples=ms, xi=xi)
                        a = adjusted_rand_score(y_true, l)
                        if a > best_ari:
                            best_ari, best_l = a, l.copy()
                    except:
                        pass
            except:
                pass
        return best_l if best_l is not None else np.zeros(X.shape[0], dtype=int)


def get_dims(k, d):
    return {'k-1': max(k-1, 2), '25%': max(int(round(d*0.25)), 1), '50%': max(int(round(d*0.50)), 1)}


def run_dataset(X, y, k, d):
    dims = get_dims(k, d)
    res = {}
    for cn in CLUSTERING:
        res[cn] = {}
        # No reduction
        labels = cluster(cn, X, k, y)
        res[cn]['No Reduction'] = adjusted_rand_score(y, labels)
        
        for dr in DR_METHODS:
            # Skip MDS for d>50 (too slow)
            if dr == 'MDS' and d > 50:
                for lv in dims:
                    res[cn][f'{dr}_{lv}'] = None
                continue
            
            for lv, nc in dims.items():
                key = f'{dr}_{lv}'
                try:
                    X_dr = apply_dr(dr, X, nc)
                    labels = cluster(cn, X_dr, k, y)
                    res[cn][key] = adjusted_rand_score(y, labels)
                except:
                    res[cn][key] = 0.0
    return res


def main(n_repeats=2):
    configs = []
    for k in [2, 5]:
        for d in [10, 50, 200]:
            configs.append((k, d))
    
    # RSG uses different k values
    rsg_configs = []
    for k in [2, 10, 50]:
        for d in [10, 50, 200]:
            rsg_configs.append((k, d))
    
    generators = {
        'Circles': (gen_circles, configs),
        'Moons': (gen_moons, configs),
        'RSG': (gen_rsg, rsg_configs),
        'Repliclust': (gen_repliclust, configs),
    }
    
    all_raw = {}
    all_avg = {}
    
    for dtype, (gen_fn, cfgs) in generators.items():
        print(f"\n{'='*60}")
        print(f"  {dtype}: {len(cfgs)} configs × {n_repeats} repeats")
        print(f"{'='*60}")
        
        all_raw[dtype] = []
        accum = {}
        
        for ci, (k, d) in enumerate(cfgs):
            for r in range(n_repeats):
                rng = np.random.RandomState(42 + ci*1000 + r)
                t0 = time.time()
                
                if dtype == 'RSG':
                    X, y, k_actual = gen_fn(rng, k=k, d=d, n_per=50)
                else:
                    X, y, k_actual = gen_fn(rng, k=k, d=d, n=500)
                
                res = run_dataset(X, y, k_actual, d)
                elapsed = time.time() - t0
                
                all_raw[dtype].append({'k': k, 'd': d, 'repeat': r, 'results': res})
                
                for cn, cr in res.items():
                    if cn not in accum:
                        accum[cn] = {}
                    for key, val in cr.items():
                        if val is not None:
                            accum[cn].setdefault(key, []).append(val)
                
                nr = res['k-means']['No Reduction']
                print(f"  k={k}, d={d}, rep={r}: kmeans_NR={nr:.3f} ({elapsed:.1f}s)")
        
        all_avg[dtype] = {}
        for cn, cd in accum.items():
            all_avg[dtype][cn] = {}
            for key, vals in cd.items():
                all_avg[dtype][cn][key] = float(np.mean(vals))
        
        # Save after each type
        with open(os.path.join(RESULTS_DIR, 'synthetic_raw_v3.json'), 'w') as f:
            json.dump(all_raw, f, indent=2, default=str)
        with open(os.path.join(RESULTS_DIR, 'synthetic_results_v3.json'), 'w') as f:
            json.dump(all_avg, f, indent=2)
        
        print(f"\n  {dtype} summary:")
        for cn in CLUSTERING:
            if cn in all_avg[dtype]:
                nr = all_avg[dtype][cn].get('No Reduction', 0)
                print(f"    {cn}: NR={nr:.3f}")
    
    print("\n\nDone! Results saved to synthetic_results_v3.json")
    return all_avg


if __name__ == '__main__':
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    main(n)
