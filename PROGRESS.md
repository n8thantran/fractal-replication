# Progress Tracker

## Current Phase: Planning and Initial Setup

## Paper Summary
- **Title**: Assessing the impact of dimensionality reduction on clustering performance
- **Goal**: Systematic comparison of 5 DR methods × 4 clustering algorithms × 3 reduction levels on synthetic + real-world data
- **Metric**: Adjusted Rand Index (ARI)

## Implementation Plan
- [ ] 1. Setup environment and install dependencies
- [ ] 2. Download and prepare 20 UCI real-world datasets
- [ ] 3. Implement DR methods: PCA, Kernel PCA, VAE, Isomap, MDS
- [ ] 4. Implement clustering: k-means, AHC, GMM, OPTICS (with param search)
- [ ] 5. Implement evaluation pipeline (ARI computation)
- [ ] 6. Run real-world data experiments → Tables A.5-A.8
- [ ] 7. Generate synthetic datasets (Circles, Moons, RSG, Repliclust)
- [ ] 8. Run synthetic data experiments → Tables A.1-A.4
- [ ] 9. Compute aggregate statistics → Tables 2-5
- [ ] 10. Compute Wilcoxon test → Table A.9
- [ ] 11. Generate boxplot figures
- [ ] 12. Write REPORT.md and reproduce.sh

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
(none yet)

## Failed Approaches
(none yet)

## Evaluation Coverage
### Tables to reproduce:
- Tables A.1-A.4: Avg ARI for Circles/Moons/RSG/Repliclust synthetic data
- Tables A.5-A.8: ARI for 20 real-world datasets (k-means, AHC, GMM, OPTICS)
- Tables 2-5: Aggregate win rates and avg win/loss percentages
- Table A.9: Wilcoxon signed-rank test
- Boxplot figures
