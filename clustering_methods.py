"""
Clustering methods: k-means, AHC, GMM, OPTICS.
With parameter search for AHC, GMM, and OPTICS.
"""

import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, OPTICS
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
import warnings
warnings.filterwarnings('ignore')


def run_kmeans(X, n_clusters, **kwargs):
    """K-means with kmeans++ init, n_init=100."""
    km = KMeans(n_clusters=n_clusters, init='k-means++', n_init=100, random_state=42)
    labels = km.fit_predict(X)
    return labels


def run_ahc(X, n_clusters, affinity='euclidean', linkage='ward', **kwargs):
    """Agglomerative Hierarchical Clustering."""
    # Ward only works with euclidean
    if linkage == 'ward' and affinity != 'euclidean':
        affinity = 'euclidean'
    
    try:
        ahc = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric=affinity if linkage != 'ward' else 'euclidean',
            linkage=linkage,
        )
        labels = ahc.fit_predict(X)
        return labels
    except Exception:
        ahc = AgglomerativeClustering(n_clusters=n_clusters)
        return ahc.fit_predict(X)


def run_gmm(X, n_clusters, covariance_type='full', **kwargs):
    """Gaussian Mixture Model."""
    try:
        gmm = GaussianMixture(
            n_components=n_clusters,
            covariance_type=covariance_type,
            random_state=42,
            max_iter=200,
        )
        labels = gmm.fit_predict(X)
        return labels
    except Exception:
        gmm = GaussianMixture(n_components=n_clusters, covariance_type='diag', random_state=42)
        return gmm.fit_predict(X)


def run_optics(X, n_clusters, min_samples=5, min_cluster_size=0.05, **kwargs):
    """OPTICS with xi method."""
    try:
        optics = OPTICS(
            min_samples=min_samples,
            cluster_method='xi',
            xi=min_cluster_size,
        )
        labels = optics.fit_predict(X)
        return labels
    except Exception:
        return -np.ones(X.shape[0], dtype=int)


def find_best_ahc_params(X, y, n_clusters):
    """Find best AHC affinity+linkage combination."""
    best_ari = -2
    best_params = {'affinity': 'euclidean', 'linkage': 'ward'}
    
    affinities = ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']
    linkages = ['complete', 'average', 'single', 'ward']
    
    for linkage in linkages:
        for affinity in affinities:
            if linkage == 'ward' and affinity != 'euclidean':
                continue
            try:
                labels = run_ahc(X, n_clusters, affinity=affinity, linkage=linkage)
                ari = adjusted_rand_score(y, labels)
                if ari > best_ari:
                    best_ari = ari
                    best_params = {'affinity': affinity, 'linkage': linkage}
            except Exception:
                continue
    
    return best_params, best_ari


def find_best_gmm_params(X, y, n_clusters):
    """Find best GMM covariance type."""
    best_ari = -2
    best_params = {'covariance_type': 'full'}
    
    for cov_type in ['spherical', 'tied', 'diag', 'full']:
        try:
            labels = run_gmm(X, n_clusters, covariance_type=cov_type)
            ari = adjusted_rand_score(y, labels)
            if ari > best_ari:
                best_ari = ari
                best_params = {'covariance_type': cov_type}
        except Exception:
            continue
    
    return best_params, best_ari


def find_best_optics_params(X, y, n_clusters):
    """Find best OPTICS parameters."""
    best_ari = -2
    best_params = {'min_samples': 5, 'min_cluster_size': 0.05}
    
    for min_samples in range(5, 11):
        for xi_val in np.arange(0.01, 1.01, 0.05):
            try:
                labels = run_optics(X, n_clusters, min_samples=min_samples, min_cluster_size=xi_val)
                ari = adjusted_rand_score(y, labels)
                if ari > best_ari:
                    best_ari = ari
                    best_params = {'min_samples': min_samples, 'min_cluster_size': xi_val}
            except Exception:
                continue
    
    return best_params, best_ari


# ===== Main interface =====
CLUSTERING_METHODS = {
    'k-means': run_kmeans,
    'AHC': run_ahc,
    'GMM': run_gmm,
    'OPTICS': run_optics,
}

PARAM_SEARCH = {
    'AHC': find_best_ahc_params,
    'GMM': find_best_gmm_params,
    'OPTICS': find_best_optics_params,
}


def cluster_and_evaluate(X, y, n_clusters, method_name, params=None):
    """Run clustering and return ARI."""
    if params is None:
        params = {}
    
    func = CLUSTERING_METHODS[method_name]
    labels = func(X, n_clusters, **params)
    ari = adjusted_rand_score(y, labels)
    return ari
