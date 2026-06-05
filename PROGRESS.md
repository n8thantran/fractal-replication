# Progress Tracker

## Current Phase: Optimizing and running real-world experiments

## Paper Summary
- **Title**: Assessing the impact of dimensionality reduction on clustering performance
- **Goal**: Systematic comparison of 5 DR methods × 4 clustering algorithms × 3 reduction levels on synthetic + real-world data
- **Metric**: Adjusted Rand Index (ARI)

## Implementation Plan
- [x] 1. Setup environment and install dependencies
- [x] 2. Download and prepare 20 UCI real-world datasets (all cached in /workspace/data/uci/)
- [x] 3. Implement DR methods: PCA, Kernel PCA, VAE, Isomap, MDS (dr_methods.py)
- [x] 4. Implement clustering: k-means, AHC, GMM, OPTICS with param search (clustering_methods.py)
- [x] 5. Implement evaluation pipeline (run_real_experiments.py)
- [ ] 6. **OPTIMIZE OPTICS** - parameter search too slow, need coarser grid
- [ ] 7. Run real-world data experiments → Tables A.5-A.8
- [ ] 8. Generate synthetic datasets (Circles, Moons, RSG, Repliclust)
- [ ] 9. Run synthetic data experiments → Tables A.1-A.4
- [ ] 10. Compute aggregate statistics → Tables 2-5
- [ ] 11. Compute Wilcoxon test → Table A.9
- [ ] 12. Generate boxplot figures
- [ ] 13. Write REPORT.md and reproduce.sh

## Key Decisions & Hyperparameters
- **PCA**: scikit-learn default, only n_components changes
- **Kernel PCA**: kernel='rbf', scikit-learn defaults
- **VAE**: Encoder 64→32→BN→Dropout(0.4)→mu/logvar, Decoder mirrors, Adam, MSE, batch=64, 100 epochs, 70/30 split, sigmoid output
- **Isomap**: scikit-learn default, only n_components changes
- **MDS**: random_state=10, n_init=50
- **K-means**: kmeans++, n_init=100
- **AHC**: Best affinity+linkage per dataset type (grid search)
- **GMM**: Best covariance type per dataset type (grid search)
- **OPTICS**: xi method, min_samples 5-10, min_cluster_size 0-1 step 0.05, best per type
- **DR levels**: max(k-1, 2), 25%, 50% of original dims
- **Preprocessing**: z-score normalization

## Completed Work
- **data_loader.py**: Loads all 20 UCI datasets, z-score normalization. Tested, working.
- **dr_methods.py**: PCA, Kernel PCA, VAE, Isomap, MDS. Tested on Iris.
- **clustering_methods.py**: k-means, AHC, GMM, OPTICS with param search. Working but OPTICS search too slow.
- **run_real_experiments.py**: Main pipeline. Timed out at 1hr due to OPTICS.
- All 20 datasets cached in /workspace/data/uci/*.npz

## Failed Approaches
- **OPTICS param search with fine grid**: 6 min_samples × 20 xi values = 120 fits per dataset. On Segmentation (n=2310), each OPTICS fit takes ~1s, so 120 fits = ~2 min per dataset. But with 20 datasets × 16 conditions (baseline + 15 DR combos), this becomes 20 × 16 × 120 = 38,400 OPTICS fits. TOO SLOW.
- **Fix**: Use coarser grid for OPTICS, or fit OPTICS once per min_samples and vary xi post-hoc (OPTICS stores reachability, xi extraction is fast).

## Evaluation Coverage
### Tables to reproduce:
- Tables A.1-A.4: Avg ARI for Circles/Moons/RSG/Repliclust synthetic data
- Tables A.5-A.8: ARI for 20 real-world datasets (k-means, AHC, GMM, OPTICS)
- Tables 2-5: Aggregate win rates and avg win/loss percentages
- Table A.9: Wilcoxon signed-rank test
- Boxplot figures

## Workspace Notes
- Old files from previous paper (fractal_init.py, model.py, train.py, lra_datasets.py, generate_figures.py) still in workspace - will clean up at end
- Results directory has old files from previous paper - will clean up
