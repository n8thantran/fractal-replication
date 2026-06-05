"""
Data loader for UCI real-world datasets used in the paper.
20 datasets from UCI ML Repository as specified in Table 1.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
import os
import warnings
import urllib.request
import tempfile
warnings.filterwarnings('ignore')


# Dataset specifications from Table 1
DATASET_SPECS = {
    'Breast tissue':      {'objects': 106,  'features': 9,   'clusters': 6},
    'Breast Wisconsin':   {'objects': 569,  'features': 30,  'clusters': 2},
    'Ecoli':              {'objects': 336,  'features': 7,   'clusters': 8},
    'Glass':              {'objects': 214,  'features': 9,   'clusters': 7},
    'Haberman':           {'objects': 306,  'features': 3,   'clusters': 2},
    'Ionosphere':         {'objects': 351,  'features': 34,  'clusters': 2},
    'Iris':               {'objects': 150,  'features': 4,   'clusters': 3},
    'Movement libras':    {'objects': 360,  'features': 90,  'clusters': 15},
    'Musk':               {'objects': 476,  'features': 166, 'clusters': 2},
    'Parkinsons':         {'objects': 195,  'features': 22,  'clusters': 2},
    'Segmentation':       {'objects': 2310, 'features': 19,  'clusters': 7},
    'Sonar all':          {'objects': 208,  'features': 60,  'clusters': 2},
    'Spectf':             {'objects': 267,  'features': 44,  'clusters': 2},
    'Transfusion':        {'objects': 748,  'features': 4,   'clusters': 2},
    'Vehicle':            {'objects': 846,  'features': 18,  'clusters': 4},
    'Vertebral column':   {'objects': 310,  'features': 6,   'clusters': 3},
    'Vowel context':      {'objects': 990,  'features': 10,  'clusters': 11},
    'Wine':               {'objects': 178,  'features': 13,  'clusters': 3},
    'Wine quality red':   {'objects': 1599, 'features': 11,  'clusters': 6},
    'Yeast':              {'objects': 1484, 'features': 8,   'clusters': 10},
}


def _load_breast_tissue():
    """Breast Tissue from Excel file."""
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00192/BreastTissue.xls'
    tmp = tempfile.NamedTemporaryFile(suffix='.xls', delete=False)
    urllib.request.urlretrieve(url, tmp.name)
    df = pd.read_excel(tmp.name, sheet_name='Data')
    X = df.iloc[:, 2:].values.astype(float)
    y = LabelEncoder().fit_transform(df.iloc[:, 1].values)
    os.unlink(tmp.name)
    return X, y


def _load_breast_wisconsin():
    data = load_breast_cancer()
    return data.data, data.target


def _load_ecoli():
    from ucimlrepo import fetch_ucirepo
    d = fetch_ucirepo(id=39)
    X = d.data.features.values.astype(float)
    y = LabelEncoder().fit_transform(d.data.targets.values.ravel().astype(str))
    return X, y


def _load_glass():
    from ucimlrepo import fetch_ucirepo
    d = fetch_ucirepo(id=42)
    X = d.data.features.values.astype(float)
    y = LabelEncoder().fit_transform(d.data.targets.values.ravel().astype(str))
    return X, y


def _load_haberman():
    from ucimlrepo import fetch_ucirepo
    d = fetch_ucirepo(id=43)
    X = d.data.features.values.astype(float)
    y = LabelEncoder().fit_transform(d.data.targets.values.ravel().astype(str))
    return X, y


def _load_ionosphere():
    from ucimlrepo import fetch_ucirepo
    d = fetch_ucirepo(id=52)
    X = d.data.features.values.astype(float)
    y = LabelEncoder().fit_transform(d.data.targets.values.ravel().astype(str))
    return X, y


def _load_iris():
    data = load_iris()
    return data.data, data.target


def _load_movement_libras():
    """Movement Libras from URL."""
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/libras/movement_libras.data'
    df = pd.read_csv(url, header=None)
    X = df.iloc[:, :-1].values.astype(float)
    y = LabelEncoder().fit_transform(df.iloc[:, -1].values)
    return X, y


def _load_musk():
    """Musk (Version 1) - clean1.data."""
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/musk/clean1.data.Z'
    tmp = tempfile.NamedTemporaryFile(suffix='.Z', delete=False)
    urllib.request.urlretrieve(url, tmp.name)
    import subprocess
    subprocess.run(['uncompress', '-f', tmp.name], check=True)
    decompressed = tmp.name[:-2]
    df = pd.read_csv(decompressed, header=None)
    # First 2 columns are molecule/conformation names, last column is class
    X = df.iloc[:, 2:-1].values.astype(float)
    y = LabelEncoder().fit_transform(df.iloc[:, -1].values)
    os.unlink(decompressed)
    return X, y


def _load_parkinsons():
    from ucimlrepo import fetch_ucirepo
    d = fetch_ucirepo(id=174)
    X = d.data.features.values.astype(float)
    y = LabelEncoder().fit_transform(d.data.targets.values.ravel().astype(str))
    return X, y


def _load_segmentation():
    """Image Segmentation - combine train and test."""
    url_train = 'https://archive.ics.uci.edu/ml/machine-learning-databases/image/segmentation.data'
    url_test = 'https://archive.ics.uci.edu/ml/machine-learning-databases/image/segmentation.test'
    df_train = pd.read_csv(url_train, header=None, skiprows=5)
    df_test = pd.read_csv(url_test, header=None, skiprows=5)
    combined = pd.concat([df_train, df_test], ignore_index=True)
    X = combined.iloc[:, 1:].values.astype(float)
    y = LabelEncoder().fit_transform(combined.iloc[:, 0].values)
    return X, y


def _load_sonar():
    from ucimlrepo import fetch_ucirepo
    d = fetch_ucirepo(id=151)
    X = d.data.features.values.astype(float)
    y = LabelEncoder().fit_transform(d.data.targets.values.ravel().astype(str))
    return X, y


def _load_spectf():
    """SPECTF Heart - combine train and test (44 features)."""
    url_train = 'https://archive.ics.uci.edu/ml/machine-learning-databases/spect/SPECTF.train'
    url_test = 'https://archive.ics.uci.edu/ml/machine-learning-databases/spect/SPECTF.test'
    df_train = pd.read_csv(url_train, header=None)
    df_test = pd.read_csv(url_test, header=None)
    combined = pd.concat([df_train, df_test], ignore_index=True)
    X = combined.iloc[:, 1:].values.astype(float)
    y = LabelEncoder().fit_transform(combined.iloc[:, 0].values)
    return X, y


def _load_transfusion():
    from ucimlrepo import fetch_ucirepo
    d = fetch_ucirepo(id=176)
    X = d.data.features.values.astype(float)
    y = LabelEncoder().fit_transform(d.data.targets.values.ravel().astype(str))
    return X, y


def _load_vehicle():
    """Vehicle - remove erroneous '204' class entry."""
    from ucimlrepo import fetch_ucirepo
    d = fetch_ucirepo(id=149)
    X = d.data.features.values
    y_raw = d.data.targets.values.ravel().astype(str)
    # Filter out the '204' erroneous entry
    mask = y_raw != '204'
    X = X[mask].astype(float)
    y = LabelEncoder().fit_transform(y_raw[mask])
    return X, y


def _load_vertebral_column():
    from ucimlrepo import fetch_ucirepo
    d = fetch_ucirepo(id=212)
    X = d.data.features.values.astype(float)
    y = LabelEncoder().fit_transform(d.data.targets.values.ravel().astype(str))
    return X, y


def _load_vowel_context():
    """Vowel Context - columns 3-12 are features, column 13 is class."""
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/vowel/vowel-context.data'
    df = pd.read_csv(url, sep='\\s+', header=None)
    # Columns: 0=train/test, 1=speaker, 2=sex, 3-12=features, 13=class
    X = df.iloc[:, 3:13].values.astype(float)
    y = LabelEncoder().fit_transform(df.iloc[:, 13].values)
    return X, y


def _load_wine():
    data = load_wine()
    return data.data, data.target


def _load_wine_quality_red():
    """Wine Quality - Red wine only."""
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
    df = pd.read_csv(url, sep=';')
    X = df.iloc[:, :-1].values.astype(float)
    y = LabelEncoder().fit_transform(df.iloc[:, -1].values)
    return X, y


def _load_yeast():
    from ucimlrepo import fetch_ucirepo
    d = fetch_ucirepo(id=110)
    X = d.data.features.values.astype(float)
    y = LabelEncoder().fit_transform(d.data.targets.values.ravel().astype(str))
    return X, y


# Map dataset names to loader functions
LOADERS = {
    'Breast tissue':    _load_breast_tissue,
    'Breast Wisconsin': _load_breast_wisconsin,
    'Ecoli':            _load_ecoli,
    'Glass':            _load_glass,
    'Haberman':         _load_haberman,
    'Ionosphere':       _load_ionosphere,
    'Iris':             _load_iris,
    'Movement libras':  _load_movement_libras,
    'Musk':             _load_musk,
    'Parkinsons':       _load_parkinsons,
    'Segmentation':     _load_segmentation,
    'Sonar all':        _load_sonar,
    'Spectf':           _load_spectf,
    'Transfusion':      _load_transfusion,
    'Vehicle':          _load_vehicle,
    'Vertebral column': _load_vertebral_column,
    'Vowel context':    _load_vowel_context,
    'Wine':             _load_wine,
    'Wine quality red': _load_wine_quality_red,
    'Yeast':            _load_yeast,
}


def download_and_cache_datasets(cache_dir='/workspace/data/uci', force=False):
    """Download all UCI datasets and cache them locally."""
    os.makedirs(cache_dir, exist_ok=True)
    datasets = {}
    
    for name in DATASET_SPECS:
        cache_path = os.path.join(cache_dir, f"{name.replace(' ', '_')}.npz")
        if os.path.exists(cache_path) and not force:
            data = np.load(cache_path)
            datasets[name] = (data['X'], data['y'])
            continue
        
        try:
            X, y = LOADERS[name]()
            if X is not None:
                # Handle missing values
                if np.any(np.isnan(X)):
                    col_medians = np.nanmedian(X, axis=0)
                    for j in range(X.shape[1]):
                        mask = np.isnan(X[:, j])
                        X[mask, j] = col_medians[j]
                
                np.savez(cache_path, X=X, y=y)
                datasets[name] = (X, y)
                print(f"✓ {name}: {X.shape[0]} objects, {X.shape[1]} features, {len(np.unique(y))} clusters")
            else:
                print(f"✗ {name}: Failed to load")
        except Exception as e:
            print(f"✗ {name}: {e}")
    
    return datasets


def load_all_real_datasets(cache_dir='/workspace/data/uci'):
    """Load all real-world datasets, applying z-score normalization."""
    datasets = download_and_cache_datasets(cache_dir)
    normalized = {}
    
    for name, (X, y) in datasets.items():
        scaler = StandardScaler()
        X_norm = scaler.fit_transform(X)
        n_clusters = DATASET_SPECS[name]['clusters']
        normalized[name] = {
            'X': X_norm,
            'y': y,
            'n_clusters': n_clusters,
            'n_features': X.shape[1],
            'n_objects': X.shape[0]
        }
    
    return normalized


if __name__ == '__main__':
    # Force re-download all datasets
    datasets = download_and_cache_datasets(force=True)
    print(f"\nLoaded {len(datasets)} datasets:")
    for name in DATASET_SPECS:
        if name in datasets:
            X, y = datasets[name]
            spec = DATASET_SPECS[name]
            match_obj = "✓" if X.shape[0] == spec['objects'] else f"✗({X.shape[0]})"
            match_feat = "✓" if X.shape[1] == spec['features'] else f"✗({X.shape[1]})"
            match_clust = "✓" if len(np.unique(y)) == spec['clusters'] else f"✗({len(np.unique(y))})"
            print(f"  {name:25s}: obj={match_obj:>8} feat={match_feat:>8} clust={match_clust:>8}")
        else:
            print(f"  {name:25s}: MISSING")
