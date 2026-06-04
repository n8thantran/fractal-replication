# FRACTAL Paper Replication Progress

## Current Phase
sCIFAR-10 training running: 7/30 epochs done, 56.06% val acc. ~2.4 hours remaining.
GPU occupied. Next: ListOps after sCIFAR-10 finishes.

## Paper Summary
FRACTAL: fractional-order measure in HiPPO → Jacobi polynomial basis → diagonal SSM with multi-α filter bank.

### Key Results to Reproduce (Table 1 - LRA Benchmark)
| Task | FRACTAL | S5 |
|------|---------|-----|
| ListOps | 61.85 | 61.10 |
| Text | 89.10 | 88.72 |
| Retrieval | 91.19 | 91.27 |
| Image | 87.30 | 87.59 |
| Pathfinder | 94.80 | 95.04 |
| Path-X | 98.39 | 98.62 |
| **Avg** | **87.11** | **87.04** |

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
- [x] sCIFAR-10 training started (30 epochs) - IN PROGRESS (7/30, 56.06%)
- [x] reproduce.sh written ✓
- [x] REPORT.md written ✓
- [ ] sCIFAR-10 training complete → save results
- [ ] ListOps experiment (15 epochs)
- [ ] Text/IMDB experiment (10 epochs) 
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
  - Full model with embedding/linear encoder and classification head
  - ~5.77M parameters
- **lra_datasets.py**: Data loading for ListOps, IMDB, sCIFAR-10, Retrieval
- **train.py**: Training loop with cosine schedule, AMP, gradient clipping
- **generate_figures.py**: Generates all paper figures
- **reproduce.sh**: Script to run all experiments
- **REPORT.md**: Report template (needs final results)
- **results/**: Contains figure PNGs and training logs

## Training Results So Far
- sCIFAR-10 (Image): 7/30 epochs → 56.06% val acc (paper target: 87.30% at 200 epochs)
  - Epoch 1: 26.10%, Epoch 2: 37.88%, Epoch 3: 39.94%
  - Epoch 4: 45.68%, Epoch 5: 49.02%, Epoch 6: 51.70%, Epoch 7: 56.06%
  - Steady improvement, ~374s/epoch
  - Extrapolating: should reach ~70-75% at 30 epochs

## Failed Approaches
- LRA dataset download from Google Cloud failed (403 Forbidden) → generated synthetic ListOps, used torchvision CIFAR-10, used IMDB from datasets library instead
- torch.compile failed with complex number conjugation issue → disabled compilation
- annot_kws parameter in matplotlib imshow → removed, using text annotations instead
- bs=32 and bs=16 for image task caused OOM → using bs=50 (paper default)

## Evaluation Coverage
### Addressed:
- Main result: LRA benchmark Table 1 - sCIFAR-10 IN PROGRESS
- Figure 2: A matrix heatmap visualization ✓
- Figure 1: Memory measure visualization ✓ 
- Eigenvalue verification ✓
- Filter bank diversity visualization ✓
- Mathematical correctness of A(α), B(α) ✓

### Not addressed (compute-limited):
- Full 200-epoch training for any task
- ListOps, Text, Retrieval, Pathfinder, Path-X experiments
- Ablation studies (Table 2)
- Comparison with other SSM baselines

## Background Training Process
- PID 6305: python train.py --task image --epochs 30 --batch_size 50 --d_model 256 --state_dim 64 --n_layers 6 --num_workers 4
- Log: results/image_training_30ep.log
- Started at ~03:11, expected completion ~06:00
