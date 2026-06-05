"""
Dimensionality reduction methods: PCA, Kernel PCA, VAE, Isomap, MDS.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap, MDS
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')


def apply_pca(X, n_components):
    """PCA with scikit-learn defaults."""
    n_components = min(n_components, X.shape[1], X.shape[0])
    if n_components < 1:
        n_components = 1
    pca = PCA(n_components=n_components)
    return pca.fit_transform(X)


def apply_kernel_pca(X, n_components):
    """Kernel PCA with RBF kernel, scikit-learn defaults."""
    n_components = min(n_components, X.shape[1], X.shape[0])
    if n_components < 1:
        n_components = 1
    kpca = KernelPCA(n_components=n_components, kernel='rbf')
    try:
        result = kpca.fit_transform(X)
        # Check for degenerate output
        if result.shape[1] < n_components:
            # Pad with zeros
            pad = np.zeros((result.shape[0], n_components - result.shape[1]))
            result = np.hstack([result, pad])
        return result
    except Exception:
        # Fallback to PCA
        return apply_pca(X, n_components)


def apply_isomap(X, n_components):
    """Isomap with scikit-learn defaults."""
    n_components = min(n_components, X.shape[1], X.shape[0] - 1)
    if n_components < 1:
        n_components = 1
    try:
        iso = Isomap(n_components=n_components)
        return iso.fit_transform(X)
    except Exception:
        return apply_pca(X, n_components)


def apply_mds(X, n_components):
    """MDS with random_state=10, n_init=50."""
    n_components = min(n_components, X.shape[1], X.shape[0])
    if n_components < 1:
        n_components = 1
    try:
        mds = MDS(n_components=n_components, random_state=10, n_init=50, normalized_stress='auto')
        return mds.fit_transform(X)
    except Exception:
        return apply_pca(X, n_components)


# ===== VAE =====
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder: input -> 64 -> 32 -> BN -> Dropout(0.4) -> mu, logvar
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.4),
        )
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)
        
        # Decoder: latent -> 32 -> 64 -> BN -> Dropout(0.4) -> output (sigmoid)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.4),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.4),
            nn.Linear(64, input_dim),
            nn.Sigmoid(),
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


def vae_loss(x_recon, x, mu, logvar):
    """MSE reconstruction + KL divergence."""
    mse = nn.functional.mse_loss(x_recon, x, reduction='sum')
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return mse + kl


def apply_vae(X, n_components):
    """Apply VAE dimensionality reduction."""
    n_components = min(n_components, X.shape[1])
    if n_components < 1:
        n_components = 1
    
    # Min-max scale to [0, 1] for sigmoid output
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    denom = X_max - X_min
    denom[denom == 0] = 1
    X_scaled = (X - X_min) / denom
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 70/30 train-validation split
    n = X_scaled.shape[0]
    n_train = int(0.7 * n)
    indices = np.random.permutation(n)
    train_idx = indices[:n_train]
    
    X_tensor = torch.FloatTensor(X_scaled).to(device)
    train_data = TensorDataset(X_tensor[train_idx])
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, drop_last=False)
    
    model = VAE(X.shape[1], n_components).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    
    model.train()
    for epoch in range(100):
        for batch in train_loader:
            x = batch[0]
            x_recon, mu, logvar = model(x)
            loss = vae_loss(x_recon, x, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Get embeddings (z_mean) for all data
    model.eval()
    with torch.no_grad():
        mu, _ = model.encode(X_tensor)
        embeddings = mu.cpu().numpy()
    
    return embeddings


# ===== Main interface =====
DR_METHODS = {
    'PCA': apply_pca,
    'Kernel PCA': apply_kernel_pca,
    'VAE': apply_vae,
    'Isomap': apply_isomap,
    'MDS': apply_mds,
}


def get_reduction_levels(n_features, n_clusters):
    """Get the three reduction levels: k-1, 25%, 50%."""
    k_minus_1 = max(n_clusters - 1, 2)
    pct_25 = max(int(np.round(0.25 * n_features)), 2)
    pct_50 = max(int(np.round(0.50 * n_features)), 2)
    return {
        'k-1': k_minus_1,
        '25%': pct_25,
        '50%': pct_50,
    }


def apply_dr(method_name, X, n_components):
    """Apply a dimensionality reduction method."""
    if n_components >= X.shape[1]:
        return X.copy()
    return DR_METHODS[method_name](X, n_components)
