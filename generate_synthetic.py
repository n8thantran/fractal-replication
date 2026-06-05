"""
Generate synthetic datasets matching the paper's description:
- Circles: 2 and 5 clusters, embedded in 10/50/200 dims
- Moons: 2 and 5 clusters, with transformations, embedded in 10/50/200 dims  
- RSG: Rodriguez Structured Gaussian with k∈{2,10,50}, d∈{10,50,200}, Nc∈{5,50,100}
- Repliclust: 2 and 5 clusters, in 10/50/200 dims

Each config gets n_repeats datasets. Noise injection on 75% of features.
"""
import numpy as np
from sklearn.random_projection import GaussianRandomProjection
from sklearn.datasets import make_moons, make_circles
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


def inject_noise(X, rng):
    """Structured noise injection: 75% of features get noise, 25% untouched.
    1/4 features: N(0,1), 1/4: N(0,0.5), 1/4: N(0,0.25), 1/4: no noise."""
    n, d = X.shape
    idx = rng.permutation(d)
    q = d // 4
    # First quarter: sigma=1
    X[:, idx[:q]] += rng.normal(0, 1, (n, q))
    # Second quarter: sigma=0.5
    X[:, idx[q:2*q]] += rng.normal(0, 0.5, (n, 2*q - q))
    # Third quarter: sigma=0.25
    X[:, idx[2*q:3*q]] += rng.normal(0, 0.25, (n, 3*q - 2*q))
    # Fourth quarter: no noise
    return X


def generate_circles_2(rng, n_per_cluster=1000, target_dim=50):
    """2-cluster circles using make_circles with factor=0.5"""
    n = 2 * n_per_cluster
    X, y = make_circles(n_samples=n, factor=0.5, noise=0.05, random_state=rng.randint(100000))
    # Embed into target_dim via Gaussian Random Projection
    if target_dim > 2:
        # Pad to target_dim first, then project
        X_high = np.zeros((n, target_dim))
        X_high[:, :2] = X
        # Use GRP to mix dimensions
        grp = GaussianRandomProjection(n_components=target_dim, random_state=rng.randint(100000))
        X_high = grp.fit_transform(X_high)
    else:
        X_high = X
    return X_high, y, 2


def generate_circles_5(rng, n_per_cluster=400, target_dim=50):
    """5-cluster concentric rings with radial factors 1.0, 2.0, 3.5, 5.0, 7.0"""
    X_list, y_list = [], []
    factors = [1.0, 2.0, 3.5, 5.0, 7.0]
    for i, f in enumerate(factors):
        theta = rng.uniform(0, 2*np.pi, n_per_cluster)
        x1 = f * np.cos(theta) + rng.normal(0, 0.05, n_per_cluster)
        x2 = f * np.sin(theta) + rng.normal(0, 0.05, n_per_cluster)
        X_list.append(np.column_stack([x1, x2]))
        y_list.append(np.full(n_per_cluster, i))
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    n = X.shape[0]
    if target_dim > 2:
        X_high = np.zeros((n, target_dim))
        X_high[:, :2] = X
        grp = GaussianRandomProjection(n_components=target_dim, random_state=rng.randint(100000))
        X_high = grp.fit_transform(X_high)
    else:
        X_high = X
    return X_high, y, 5


def generate_moons_2(rng, n_per_cluster=1000, target_dim=50):
    """2-cluster moons with stretching, rotation, translation"""
    n = 2 * n_per_cluster
    X, y = make_moons(n_samples=n, noise=0.1, random_state=rng.randint(100000))
    # Apply random transformation
    stretch = rng.choice([1.0, 1.5])
    angle = rng.choice([-160, -10, 10, 160, 180]) * np.pi / 180
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    X = X * np.array([stretch, 1.0])
    X = X @ R.T
    tx = rng.choice([-4, -2, 0, 2, 4])
    ty = rng.choice([1.0, 1.2, 1.5])
    X[:, 0] += tx
    X[:, 1] += ty
    
    if target_dim > 2:
        X_high = np.zeros((n, target_dim))
        X_high[:, :2] = X
        grp = GaussianRandomProjection(n_components=target_dim, random_state=rng.randint(100000))
        X_high = grp.fit_transform(X_high)
    else:
        X_high = X
    return X_high, y, 2


def generate_moons_5(rng, n_per_cluster=400, target_dim=50):
    """5-cluster moons with different transformations"""
    X_list, y_list = [], []
    stretches = [1.0, 1.5, 1.0, 1.5, 1.0]
    angles_deg = [0, 160, -10, 10, 180]
    x_shifts = [0, 3, -3, 6, -6]
    y_shifts = [0, 1.5, -1.5, 3.0, -3.0]
    
    for i in range(5):
        X_moon, _ = make_moons(n_samples=2*n_per_cluster, noise=0.1, random_state=rng.randint(100000))
        # Take first half for cluster i
        X_m = X_moon[:n_per_cluster]
        angle = angles_deg[i] * np.pi / 180
        R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        X_m = X_m * np.array([stretches[i], 1.0])
        X_m = X_m @ R.T
        X_m[:, 0] += x_shifts[i]
        X_m[:, 1] += y_shifts[i]
        X_list.append(X_m)
        y_list.append(np.full(n_per_cluster, i))
    
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    n = X.shape[0]
    if target_dim > 2:
        X_high = np.zeros((n, target_dim))
        X_high[:, :2] = X
        grp = GaussianRandomProjection(n_components=target_dim, random_state=rng.randint(100000))
        X_high = grp.fit_transform(X_high)
    else:
        X_high = X
    return X_high, y, 5


def generate_rsg(rng, k=5, d=50, n_per_cluster=50, alpha=None):
    """Rodriguez Structured Gaussian: clusters with structured covariance."""
    if alpha is None:
        # Tune alpha to avoid trivial clustering
        alpha = rng.uniform(0.3, 0.7)
    
    X_list, y_list = [], []
    # Generate cluster centers spread apart
    centers = rng.uniform(-5, 5, (k, d))
    # Scale centers by sqrt(d) to maintain separation
    centers *= (1 - alpha) * np.sqrt(d) / max(np.sqrt(d), 1)
    
    for i in range(k):
        # Generate structured covariance
        # Random rotation matrix
        A = rng.randn(d, d)
        Q, _ = np.linalg.qr(A)
        # Random eigenvalues (structured)
        eigvals = rng.uniform(0.1, 2.0, d)
        eigvals = np.sort(eigvals)[::-1]
        cov = Q @ np.diag(eigvals) @ Q.T
        # Mix with identity based on alpha
        cov = alpha * cov + (1 - alpha) * np.eye(d)
        
        try:
            samples = rng.multivariate_normal(centers[i], cov, n_per_cluster)
        except:
            samples = rng.multivariate_normal(centers[i], np.eye(d), n_per_cluster)
        X_list.append(samples)
        y_list.append(np.full(n_per_cluster, i))
    
    return np.vstack(X_list), np.concatenate(y_list), k


def generate_repliclust(rng, k=2, d=50, n_per_cluster=1000):
    """Repliclust-style: anisotropic clusters with centroid separation."""
    X_list, y_list = [], []
    
    # Generate well-separated centers
    centers = np.zeros((k, d))
    for i in range(k):
        centers[i] = rng.randn(d) * 3
    
    for i in range(k):
        # Anisotropic covariance
        A = rng.randn(d, d) * 0.3
        cov = A @ A.T + np.eye(d) * 0.1
        # Make some dimensions more spread
        spread_dims = rng.choice(d, size=max(1, d//3), replace=False)
        for sd in spread_dims:
            cov[sd, sd] += rng.uniform(1, 5)
        
        try:
            samples = rng.multivariate_normal(centers[i], cov, n_per_cluster)
        except:
            samples = rng.multivariate_normal(centers[i], np.eye(d) * 2, n_per_cluster)
        X_list.append(samples)
        y_list.append(np.full(n_per_cluster, i))
    
    return np.vstack(X_list), np.concatenate(y_list), k


def get_all_configs(n_repeats=10):
    """Generate all synthetic dataset configurations matching the paper."""
    configs = []
    
    # Circles: 2 and 5 clusters, dims 10/50/200
    for k in [2, 5]:
        for d in [10, 50, 200]:
            n_per = 1000 if k == 2 else 400
            for rep in range(n_repeats):
                configs.append({
                    'type': 'Circles',
                    'gen_func': generate_circles_2 if k == 2 else generate_circles_5,
                    'k': k, 'd': d, 'n_per_cluster': n_per, 'repeat': rep
                })
    
    # Moons: 2 and 5 clusters, dims 10/50/200
    for k in [2, 5]:
        for d in [10, 50, 200]:
            n_per = 1000 if k == 2 else 400
            for rep in range(n_repeats):
                configs.append({
                    'type': 'Moons',
                    'gen_func': generate_moons_2 if k == 2 else generate_moons_5,
                    'k': k, 'd': d, 'n_per_cluster': n_per, 'repeat': rep
                })
    
    # RSG: k∈{2,10,50}, d∈{10,50,200}, Nc∈{5,50,100}
    for k in [2, 10, 50]:
        for d in [10, 50, 200]:
            for nc in [5, 50, 100]:
                for rep in range(n_repeats):
                    configs.append({
                        'type': 'RSG',
                        'gen_func': generate_rsg,
                        'k': k, 'd': d, 'n_per_cluster': nc, 'repeat': rep
                    })
    
    # Repliclust: 2 and 5 clusters, dims 10/50/200
    for k in [2, 5]:
        for d in [10, 50, 200]:
            n_per = 1000 if k == 2 else 400
            for rep in range(n_repeats):
                configs.append({
                    'type': 'Repliclust',
                    'gen_func': generate_repliclust,
                    'k': k, 'd': d, 'n_per_cluster': n_per, 'repeat': rep
                })
    
    return configs


def generate_dataset(config, seed):
    """Generate a single dataset from config."""
    rng = np.random.RandomState(seed)
    dtype = config['type']
    
    if dtype == 'Circles':
        X, y, k = config['gen_func'](rng, n_per_cluster=config['n_per_cluster'], target_dim=config['d'])
    elif dtype == 'Moons':
        X, y, k = config['gen_func'](rng, n_per_cluster=config['n_per_cluster'], target_dim=config['d'])
    elif dtype == 'RSG':
        X, y, k = generate_rsg(rng, k=config['k'], d=config['d'], n_per_cluster=config['n_per_cluster'])
    elif dtype == 'Repliclust':
        X, y, k = generate_repliclust(rng, k=config['k'], d=config['d'], n_per_cluster=config['n_per_cluster'])
    
    # Z-score normalize
    X = StandardScaler().fit_transform(X)
    
    # Inject structured noise
    X = inject_noise(X, rng)
    
    # Re-normalize after noise
    X = StandardScaler().fit_transform(X)
    
    return X, y, k


if __name__ == '__main__':
    # Quick test
    rng = np.random.RandomState(42)
    
    for gen_name, gen_func, kwargs in [
        ('Circles-2', generate_circles_2, {'rng': rng, 'n_per_cluster': 1000, 'target_dim': 50}),
        ('Circles-5', generate_circles_5, {'rng': rng, 'n_per_cluster': 400, 'target_dim': 50}),
        ('Moons-2', generate_moons_2, {'rng': rng, 'n_per_cluster': 1000, 'target_dim': 50}),
        ('Moons-5', generate_moons_5, {'rng': rng, 'n_per_cluster': 400, 'target_dim': 50}),
        ('RSG', generate_rsg, {'rng': rng, 'k': 5, 'd': 50, 'n_per_cluster': 50}),
        ('Repliclust', generate_repliclust, {'rng': rng, 'k': 5, 'd': 50, 'n_per_cluster': 400}),
    ]:
        X, y, k = gen_func(**kwargs)
        print(f'{gen_name}: X={X.shape}, k={k}, classes={np.unique(y)}')
    
    # Test noise injection
    X_test = np.zeros((100, 20))
    X_noisy = inject_noise(X_test.copy(), rng)
    print(f'\nNoise test: original std={X_test.std():.3f}, noisy std={X_noisy.std():.3f}')
