# Replication Report: Assessing the Impact of Dimensionality Reduction on Clustering Performance

## Summary

This replicates the paper "Assessing the impact of dimensionality reduction on clustering performance" which systematically evaluates 5 dimensionality reduction (DR) methods × 4 clustering algorithms × 3 reduction levels on both synthetic and real-world data, using Adjusted Rand Index (ARI) as the evaluation metric.

## What Was Implemented

### Core Components
1. **Data Pipeline** (`data_loader.py`): Downloads and preprocesses 20 UCI real-world datasets with z-score normalization
2. **DR Methods** (`dr_methods.py`): PCA, Kernel PCA (RBF), Variational Autoencoder, Isomap, MDS
3. **Clustering Methods** (`clustering_methods.py`): k-means (k-means++, n_init=100), Agglomerative Hierarchical Clustering (best affinity/linkage via grid search), Gaussian Mixture Models (best covariance type via grid search), OPTICS (xi method, min_samples/xi grid search)
4. **Experiment Runner** (`run_remaining_real.py`, `run_synthetic_v3.py`): Runs all DR×clustering×reduction-level combinations
5. **Results Generator** (`generate_results.py`): Produces all tables and figures from cached JSON results

### Synthetic Data
- 4 dataset types: Circles, Moons, RSG (Rodriguez et al.), Repliclust
- 10 repeats per type with 500 samples, 20 informative + 20 noise features
- All DR methods applied at 3 reduction levels: max(k-1,2), 25%, 50% of features

### Real-World Data (20 UCI Datasets)
Breast_tissue, Breast_Wisconsin, Ecoli, Glass, Haberman, Ionosphere, Iris, Movement_libras, Musk, Parkinsons, Segmentation, Sonar_all, Spectf, Transfusion, Vehicle, Vertebral_column, Vowel_context, Wine, Wine_quality_red, Yeast

## Tables and Figures Produced

### Tables (in `results/tables/`)
| Paper Table | File | Description |
|-------------|------|-------------|
| A.1 | `table_A1_synthetic_Circles.csv/.txt` | ARI for Circles synthetic data |
| A.2 | `table_A2_synthetic_Moons.csv/.txt` | ARI for Moons synthetic data |
| A.3 | `table_A3_synthetic_RSG.csv/.txt` | ARI for RSG synthetic data |
| A.4 | `table_A4_synthetic_Repliclust.csv/.txt` | ARI for Repliclust synthetic data |
| A.5 | `table_A5_real_kmeans.csv/.txt` | Real-world k-means ARI per dataset |
| A.6 | `table_A6_real_AHC.csv/.txt` | Real-world AHC ARI per dataset |
| A.7 | `table_A7_real_GMM.csv/.txt` | Real-world GMM ARI per dataset |
| A.8 | `table_A8_real_OPTICS.csv/.txt` | Real-world OPTICS ARI per dataset |
| A.9 | `table_A9_wilcoxon.csv/.txt` | Wilcoxon signed-rank test (p-values) |
| 2 | `table_2_aggregate_kmeans.csv/.txt` | k-means: Win%, Avg% change |
| 3 | `table_3_aggregate_AHC.csv/.txt` | AHC: Win%, Avg% change |
| 4 | `table_4_aggregate_GMM.csv/.txt` | GMM: Win%, Avg% change |
| 5 | `table_5_aggregate_OPTICS.csv/.txt` | OPTICS: Win%, Avg% change |

### Figures (in `results/figures/`)
- `figure_2_boxplot_kmeans.png` — Boxplots of k-means ARI across DR methods (real data)
- `figure_3_boxplot_AHC.png` — Boxplots of AHC ARI across DR methods (real data)
- `figure_4_boxplot_GMM.png` — Boxplots of GMM ARI across DR methods (real data)
- `figure_5_boxplot_OPTICS.png` — Boxplots of OPTICS ARI across DR methods (real data)
- `heatmap_*.png` — Heatmaps of ARI per dataset × DR method

## Key Results

### Agreement with Paper Findings

1. **DR rarely improves clustering on real-world data**: Confirmed. For AHC, GMM, and OPTICS, no DR method consistently outperforms the baseline. Only Kernel PCA (k-1) significantly improves k-means (Wilcoxon p=0.012).

2. **Kernel PCA is the best DR method for k-means**: Confirmed. Kernel PCA(k-1) achieves the highest mean ARI improvement (+0.026 absolute, +15% relative) with 50% win rate on real-world data.

3. **VAE consistently degrades clustering**: Confirmed. VAE hurts performance across all clustering methods, with large negative average changes (-9% to -29% on real data).

4. **Isomap helps on synthetic data**: Confirmed. Isomap often gives the best performance on synthetic datasets for k-means.

5. **Wilcoxon test**: Only Kernel PCA(k-1) for k-means is statistically significant (p=0.012). Paper reports similar (PCA and Kernel PCA significant for k-means). Our result is more conservative.

### Quantitative Comparison (k-means, No Reduction)
- Mean absolute difference from paper ARI values: **0.052** across 20 datasets
- Notable discrepancies: Parkinsons (-0.22), Ecoli (-0.15), Haberman (-0.10)
- Exact matches (±0.01): Glass, Sonar, Spectf, Wine, Musk

## How to Reproduce

```bash
# Generate tables/figures from cached results (fast, ~10 seconds):
bash reproduce.sh

# Re-run real-world experiments from scratch (~2-4 hours):
bash reproduce.sh --real

# Re-run synthetic experiments (~1-2 hours):
bash reproduce.sh --synth

# Re-run everything (~4-6 hours):
bash reproduce.sh --full
```

## Limitations / Approximations

1. **Synthetic data**: Uses 10 repeats (paper uses 50 repeats with more configurations). Our RSG generator is a simplified version of Rodriguez et al.'s approach.
2. **VAE architecture**: The paper gives high-level description; we implemented a standard VAE with encoder=64→32→latent, decoder mirrors, 100 epochs, Adam optimizer.
3. **OPTICS xi grid**: We search xi in [0.01, 0.1, 0.2, ..., 1.0] (11 values × 3 min_samples) vs potentially finer grid in paper.
4. **Some ARI discrepancies**: Likely due to differences in exact parameter search spaces, random seeds, and potentially different preprocessing of some UCI datasets.

## File Structure

```
/workspace/
├── reproduce.sh              # Main reproduction script
├── data_loader.py            # UCI dataset loading + preprocessing
├── dr_methods.py             # PCA, KPCA, VAE, Isomap, MDS
├── clustering_methods.py     # k-means, AHC, GMM, OPTICS
├── run_remaining_real.py     # Real-world experiment runner
├── run_synthetic_v3.py       # Synthetic experiment runner
├── generate_results.py       # Table/figure generator
├── data/uci/                 # Cached UCI datasets (.npz)
├── results/
│   ├── real_world_results.json      # Raw real-world results
│   ├── synthetic_results_v3.json    # Averaged synthetic results
│   ├── synthetic_raw_v3.json        # Per-repeat synthetic results
│   ├── tables/               # All CSV and TXT tables
│   └── figures/              # All PNG figures
├── REPORT.md                 # This report
└── PROGRESS.md               # Development progress log
```
