# Progress Tracker

## Current Phase: Running experiments (Turn 100 checkpoint)

## Paper Summary
- **Title**: Assessing the impact of dimensionality reduction on clustering performance
- **Goal**: Systematic comparison of 5 DR methods × 4 clustering algorithms × 3 reduction levels on synthetic + real-world data
- **Metric**: Adjusted Rand Index (ARI)

## Implementation Plan
- [x] 1. Setup environment and install dependencies
- [x] 2. Download and prepare 20 UCI real-world datasets (all cached in /workspace/data/uci/)
- [x] 3. Implement DR methods: PCA, Kernel PCA, VAE, Isomap, MDS
- [x] 4. Implement clustering: k-means, AHC, GMM, OPTICS with param search
- [x] 5. Implement evaluation pipeline (run_all_experiments.py - comprehensive)
- [x] 6. OPTICS optimization: fit once per min_samples, vary xi post-hoc
- [ ] 7. Run real-world data experiments → Tables A.5-A.8
- [ ] 8. Run synthetic data experiments → Tables A.1-A.4
- [ ] 9. Compute aggregate statistics → Tables 2-5
- [ ] 10. Compute Wilcoxon test → Table A.9
- [ ] 11. Generate boxplot figures
- [ ] 12. Write REPORT.md and reproduce.sh

## Key Decisions & Hyperparameters
- **PCA**: scikit-learn default, only n_components changes
- **Kernel PCA**: kernel='rbf', scikit-learn defaults
- **VAE**: Encoder 64→32→BN→Dropout(0.4)→mu/logvar, Decoder mirrors, Adam, MSE+KL, batch=64, 100 epochs, 70/30 split, sigmoid output
- **Isomap**: scikit-learn default, only n_components changes
- **MDS**: random_state=10, n_init=50 (reduced to 10 for large datasets)
- **K-means**: kmeans++, n_init=100
- **AHC**: Best affinity+linkage per dataset (grid search over 16 combos)
- **GMM**: Best covariance type per dataset (grid search over 4 types)
- **OPTICS**: xi method, min_samples 5-10, xi 0.01-1.0 step 0.05, fit once per min_samples
- **DR levels**: max(k-1, 2), 25%, 50% of original dims
- **Preprocessing**: z-score normalization
- **Synthetic**: 10 repeats per dataset type, 500 samples, 20 features/noise dims

## Completed Work
- **data_loader.py**: Loads all 20 UCI datasets, z-score normalization. Tested, working.
- **dr_methods.py**: PCA, Kernel PCA, VAE, Isomap, MDS. Tested on Iris.
- **clustering_methods.py**: k-means, AHC, GMM, OPTICS with param search. Optimized OPTICS.
- **run_all_experiments.py**: Comprehensive pipeline for real + synthetic experiments. Generates all tables and figures.
- All 20 datasets cached in /workspace/data/uci/*.npz
- Old files from previous paper cleaned up

## Failed Approaches
- **OPTICS param search with fine grid**: Too slow (38,400 fits). Fixed by fitting once per min_samples and varying xi post-hoc.
- **MDS with n_init=50 on large datasets**: Too slow. Reduced to n_init=10 for n>1000.

## Evaluation Coverage
### Tables to reproduce:
- Tables A.1-A.4: Avg ARI for Circles/Moons/RSG/Repliclust synthetic data
- Tables A.5-A.8: ARI for 20 real-world datasets (k-means, AHC, GMM, OPTICS)
- Tables 2-5: Aggregate win rates and avg improvements
- Table A.9: Wilcoxon signed-rank test
- Boxplot figures

## Files
- /workspace/data_loader.py - Dataset loading (20 UCI datasets)
- /workspace/dr_methods.py - DR methods (PCA, KPCA, VAE, Isomap, MDS)
- /workspace/clustering_methods.py - Clustering methods with param search
- /workspace/run_all_experiments.py - Main experiment runner
- /workspace/run_real_experiments.py - Original real-world pipeline (superseded)
