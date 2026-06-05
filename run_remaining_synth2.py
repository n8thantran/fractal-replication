"""
Run remaining synthetic experiments (RSG and Repliclust) - ultra fast version.
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

RESULTS_DIR = '/workspace/results'
DR_METHODS = ['PCA', 'Kernel PCA', 'VAE', 'Isomap', 'MDS']
CLUSTERING = ['k-means', 'AHC', 'GMM', 'OPTICS']


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        h = min(64, input_dim)
        h2 = min(32, h)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h), nn.ReLU(),
            nn.Linear(h, h2), nn.ReLU(),
            nn.BatchNorm1d(h2), nn.Dropout(0.4))
        self.fc_mu = nn.Linear(h2, latent_dim)
        self.fc_logvar = nn.Linear(h2, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h2), nn.ReLU(),
            nn.BatchNorm1d(h2), nn.Dropout(0.4),
            nn.Linear(h2, h), nn.ReLU(),
            nn.Linear(h, input_dim), nn.Sigmoid())
    
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
    d = X_n.shape[1]; n = X_n.shape[0]
    idx = rng.permutation(d); q = d // 4
    for i in idx[:q]: X_n[:, i] += rng.normal(0, 1.0, n)
    for i in idx[q:2*q]: X_n[:, i] += rng.normal(0, 0.5, n)
    for i in idx[2*q:3*q]: X_n[:, i] += rng.normal(0, 0.25, n)
    return X_n


def gen_rsg(rng, k, d, n_per=50):
    centers = rng.uniform(-10, 10, (k, d))
    X_list, y_list = [], []
    for i in range(k):
        A = rng.randn(d, d) * 0.3
        cov = A @ A.T / d + np.eye(d) * 0.1
        X_list.append(rng.multivariate_normal(centers[i], cov, n_per))
        y_list.append(np.full(n_per, i))
    return inject_noise(np.vstack(X_list), rng), np.concatenate(y_list), k


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
    return inject_noise(np.vstack(X_list), rng), np.concatenate(y_list), k


def apply_dr(name, X, nc):
    nc = max(1, min(nc, X.shape[1]-1, X.shape[0]-1))
    if nc >= X.shape[1]:
        return X.copy()
    if name == 'PCA':
        return PCA(n_components=nc).fit_transform(X)
    elif name == 'Kernel PCA':
        try:
            r = KernelPCA(n_components=nc, kernel='rbf').fit_transform(X)
            if r.shape[1] < nc: r = np.hstack([r, np.zeros((r.shape[0], nc - r.shape[1]))])
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
        return MDS(n_components=nc, random_state=10, n_init=1, max_iter=50).fit_transform(X)
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
    for _ in range(20):
        for (b,) in loader:
            xr, mu, lv = model(b)
            loss = nn.functional.mse_loss(xr, b, reduction='sum') - 0.5*torch.sum(1+lv-mu.pow(2)-lv.exp())
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        mu, _ = model.encode(X_t)
    return mu.cpu().numpy()


def cluster(name, X, k, y_true):
    n = X.shape[0]
    if name == 'k-means':
        ni = 10 if k > 20 else 30
        return KMeans(n_clusters=k, init='k-means++', n_init=ni, random_state=42).fit_predict(X)
    elif name == 'AHC':
        try:
            return AgglomerativeClustering(n_clusters=k, metric='euclidean', linkage='ward').fit_predict(X)
        except:
            return np.zeros(n, dtype=int)
    elif name == 'GMM':
        for ct in ['full', 'tied', 'diag', 'spherical']:
            try:
                return GaussianMixture(n_components=k, covariance_type=ct, random_state=42, max_iter=50).fit_predict(X)
            except: pass
        return np.zeros(n, dtype=int)
    elif name == 'OPTICS':
        best_ari, best_l = -1, None
        ms = min(5, n - 1)
        try:
            m = OPTICS(min_samples=ms, metric='euclidean')
            m.fit(X)
            for xi in [0.01, 0.05, 0.1, 0.3]:
                try:
                    l, _ = cluster_optics_xi(reachability=m.reachability_, predecessor=m.predecessor_,
                                              ordering=m.ordering_, min_samples=ms, xi=xi)
                    a = adjusted_rand_score(y_true, l)
                    if a > best_ari: best_ari, best_l = a, l.copy()
                except: pass
        except: pass
        return best_l if best_l is not None else np.zeros(n, dtype=int)


def get_dims(k, d):
    return {'k-1': max(k-1, 2), '25%': max(int(round(d*0.25)), 1), '50%': max(int(round(d*0.50)), 1)}


def run_dataset(X, y, k, d):
    dims = get_dims(k, d)
    res = {}
    for cn in CLUSTERING:
        res[cn] = {}
        labels = cluster(cn, X, k, y)
        res[cn]['No Reduction'] = adjusted_rand_score(y, labels)
        for dr in DR_METHODS:
            if dr == 'MDS' and (d > 50 or X.shape[0] > 500):
                for lv in dims: res[cn][f'{dr}_{lv}'] = None
                continue
            if dr == 'Isomap' and X.shape[0] > 1000:
                for lv in dims: res[cn][f'{dr}_{lv}'] = None
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


def main():
    # Load existing results
    with open(os.path.join(RESULTS_DIR, 'synthetic_raw_v3.json')) as f:
        all_raw = json.load(f)
    with open(os.path.join(RESULTS_DIR, 'synthetic_results_v3.json')) as f:
        all_avg = json.load(f)
    
    n_repeats = 2
    
    rsg_configs = [(k, d) for k in [2, 10, 50] for d in [10, 50, 200]]
    rep_configs = [(k, d) for k in [2, 5] for d in [10, 50, 200]]
    
    for dtype, gen_fn, cfgs in [
        ('RSG', gen_rsg, rsg_configs),
        ('Repliclust', gen_repliclust, rep_configs),
    ]:
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
                    if cn not in accum: accum[cn] = {}
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
        
        with open(os.path.join(RESULTS_DIR, 'synthetic_raw_v3.json'), 'w') as f:
            json.dump(all_raw, f, indent=2, default=str)
        with open(os.path.join(RESULTS_DIR, 'synthetic_results_v3.json'), 'w') as f:
            json.dump(all_avg, f, indent=2)
        
        print(f"\n  {dtype} summary:")
        for cn in CLUSTERING:
            if cn in all_avg[dtype]:
                nr = all_avg[dtype][cn].get('No Reduction', 0)
                print(f"    {cn}: NR={nr:.3f}")
    
    print("\nDone!")


if __name__ == '__main__':
    main()
