# FRACTAL Paper Replication Progress

## Current Phase
All experiments complete. Finalizing deliverables (REPORT.md, reproduce.sh, results).

## Paper Summary
FRACTAL: fractional-order measure in HiPPO → Jacobi polynomial basis → diagonal SSM with multi-α filter bank.

### Key Results (Table 1 - LRA Benchmark)
| Task | FRACTAL (paper) | Our Result | Epochs (ours/paper) | Notes |
|------|----------------|------------|---------------------|-------|
| ListOps | 61.85 | 14.21% | 4/40 | 10k samples, barely started learning |
| Text | 89.10 | 74.70% | 10/40 | 5k/25k samples, clearly learning |
| Image | 87.30 | 65.93% | 30/200 | Full data, strong learning curve |
| Retrieval | 91.19 | skip | - | Requires paired data |
| Pathfinder | 94.80 | skip | - | Requires special data |
| Path-X | 98.39 | skip | - | Requires special data |

### Hyperparameters (from appendix in paper.tex)
- Model dim H=256, State dim N=64, Layers=6
- α config: [0,0,0.3,0.3,0.5,0.5,0.9,0.9] → K=8 channels
- Batch sizes: ListOps=32, Text=16, Retrieval=32, Image=50, Pathfinder=32, PathX=16
- LR=0.001, Weight decay=0.05
- Epochs: ListOps=40, Text=40, Retrieval=30, Image=200, Pathfinder=200, PathX=200
- Optimizer: AdamW (β1=0.9, β2=0.999)
- LR schedule: linear warmup 10%, cosine decay
- Gradient clip: global norm 1.0
- Mixed precision: bfloat16 forward, float32 gradients
- GLU gating with SiLU activation
- Pre-norm (LayerNorm) before SSM
- ZOH discretization

## Implementation Plan
- [x] Read paper thoroughly
- [x] Implement A(α) matrix computation (Gauss-Jacobi quadrature) ✓
- [x] Implement B(α) vector computation (closed-form) ✓
- [x] Implement diagonal SSM layer with ZOH discretization ✓
- [x] Implement parallel scan (associative scan in PyTorch) ✓
- [x] Implement FRACTAL layer (GLU gating, pre-norm) ✓
- [x] Implement full FRACTAL model (stacked layers, encoder/decoder) ✓
- [x] Implement LRA data loading (ListOps, IMDB, sCIFAR-10) ✓
- [x] Training loop with proper hyperparameters ✓
- [x] Smoke test: 1-epoch image task works (25.4% test acc) ✓
- [x] Generate figures: A matrix heatmap, memory measure, eigenvalue verification, filter bank ✓
- [x] sCIFAR-10 training: 65.93% test acc (30 epochs) ✓
- [x] Text/IMDB training: 74.70% test acc (10 epochs, 5k samples) ✓
- [x] ListOps training: 14.21% val acc (4 epochs, 10k samples) ✓
- [x] Training curves plot ✓
- [x] reproduce.sh written ✓
- [x] REPORT.md written ✓
- [ ] Update REPORT.md with final results
- [ ] Verify reproduce.sh --quick works
- [ ] Final commit and push

## Key Decisions
- Using PyTorch (paper used JAX)
- For ill-conditioned V: use diagonal eigenvalues directly (1,2,...,N)
- α config: K=8 blocks with [0,0,0.3,0.3,0.5,0.5,0.9,0.9]
- Running reduced epochs due to compute constraints (30 vs 200 for image)

## Completed Work
- **fractal_init.py**: A(α) and B(α) computation, tested and working
  - A(α=0) matches HiPPO-LegS exactly
  - Diagonal always n+1, lower triangular verified
  - B(α=0) = sqrt(2n+1) verified
- **model.py**: Full FRACTAL model with:
  - DiagonalSSMLayer with ZOH discretization
  - Parallel scan (associative scan)
  - FRACTAL layer with GLU gating, pre-norm
  - Multi-α filter bank (K=8 channels)
  - Full model with embedding/encoder for all LRA tasks
- **lra_datasets.py**: Dataset loaders for ListOps, IMDB, sCIFAR-10
  - ListOps: parsed from LRA tfrecord-like format or generated
  - IMDB: from HuggingFace datasets or local
  - sCIFAR-10: from torchvision, grayscale flattened to 1024
- **train.py**: Training loop with:
  - AdamW optimizer, cosine schedule with warmup
  - Mixed precision (bfloat16)
  - Gradient clipping (max_norm=1.0)
  - Best model checkpointing
- **generate_figures.py**: Generates all paper figures
- **reproduce.sh**: Orchestrates all experiments
- **REPORT.md**: Summary document
- **results/**: Contains figures, training logs, result JSONs, training curves

## Figures Generated (in results/)
1. figure2_A_matrix_heatmap.png - A(α) matrix structure for multiple α values
2. figure2_A_matrix_annotated.png - Annotated version showing diagonal/lower-triangular
3. B_vector_comparison.png - B(α) vectors for different α
4. figure1_memory_measure.png - Fractional memory measure dμ_α
5. eigenvalue_verification.png - Eigenvalues vs theoretical -(n+1)
6. alpha_diversity_filters.png - Multi-α filter bank impulse responses
7. training_curves.png - Training curves for all 3 tasks

## Experiment Results
### sCIFAR-10 (30 epochs, full data)
- Test: 65.93%, Best val: 65.52%
- Clear learning curve, still improving at epoch 30
- Paper: 87.30% at 200 epochs

### Text/IMDB (10 epochs, 5k samples)
- Test: 74.70%, Best val: 72.10%
- Clear learning, not yet converged
- Paper: 89.10% at 40 epochs with 25k samples

### ListOps (4 epochs, 10k samples)
- Best val: 14.21% (near random for 10-class)
- Very slow convergence expected - paper needs 40 epochs with 96k samples
- Paper: 61.85%

## Failed Approaches
- LRA benchmark data download: tried multiple URLs, all failed; used fallback datasets
- V matrix diagonalization: V was ill-conditioned, used diagonal eigenvalues directly
- Direct parallel scan: initially had shape issues, fixed by proper broadcasting
- ListOps full data (96k): each epoch takes ~28 min, impractical for our compute budget
