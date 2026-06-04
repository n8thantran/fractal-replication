# FRACTAL Paper Replication Progress

## Current Phase
Training infrastructure complete. Need to run experiments and generate visualizations.
Key bottleneck: full training is too slow (~20h for sCIFAR-10 at 200 epochs).
Strategy: Run reduced epochs, generate visualizations, demonstrate model works.

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

### Hyperparameters (from commented-out appendix in paper.tex)
- Model dim H=256, State dim N=64, Layers=6
- α config: [0,0,0.3,0.3,0.5,0.5,0.9,0.9] → K=8 channels
- Batch sizes: ListOps=32, Text=16, Retrieval=32, Image=50, Pathfinder=32, PathX=16
- LR=0.001 for all tasks, Weight decay=0.05
- Epochs: ListOps=40, Text=40, Retrieval=30, Image=200, Pathfinder=200, PathX=200
- Optimizer: AdamW (β1=0.9, β2=0.999)
- LR schedule: linear warmup 10%, cosine decay
- Gradient clip: global norm 1.0
- Mixed precision: bfloat16 forward, float32 gradients
- GLU gating with SiLU activation
- Pre-norm (LayerNorm) before SSM
- ZOH discretization
- Complex diagonal: Λ = -Λ_real + iΛ_imag

## Implementation Plan
- [x] Read paper thoroughly
- [x] Implement A(α) matrix computation (Gauss-Jacobi quadrature) ✓
- [x] Implement B(α) vector computation (closed-form) ✓
- [x] Implement diagonal SSM layer with ZOH discretization ✓ (in model.py)
- [x] Implement parallel scan (associative scan in PyTorch) ✓ (in model.py)
- [x] Implement FRACTAL layer (GLU gating, pre-norm) ✓ (in model.py)
- [x] Implement full FRACTAL model (stacked layers, encoder/decoder) ✓ (in model.py)
- [x] Implement LRA data loading (ListOps, IMDB, sCIFAR-10) ✓ (lra_datasets.py)
- [x] Training loop with proper hyperparameters ✓ (train.py)
- [x] Smoke test: 1-epoch image task works (25.4% test acc) ✓
- [ ] Run sCIFAR-10 experiment (reduced epochs ~30-50)
- [ ] Run ListOps experiment (reduced epochs ~15-20)
- [ ] Run Text/IMDB experiment (reduced epochs ~10-15)
- [ ] Generate A matrix heatmap visualization (Figure 2) — EASY, from init code
- [ ] Generate memory measure visualization (Figure 1) — medium
- [ ] Write reproduce.sh and REPORT.md

## Key Decisions
- Using PyTorch (paper used JAX)
- For ill-conditioned V: use diagonal eigenvalues directly (1,2,...,N)
- B̃_init may need normalization; paper notes random init converges to same final accuracy
- α config: K=8 blocks with [0,0,0.3,0.3,0.5,0.5,0.9,0.9]

## Completed Work
- **fractal_init.py**: A(α) and B(α) computation, tested and working
  - A(α=0) matches HiPPO-LegS exactly
  - Diagonal always n+1, lower triangular verified
  - B(α=0) = sqrt(2n+1) verified
- **model.py**: Full FRACTAL model with:
  - DiagonalSSMLayer with ZOH discretization and parallel scan
  - FRACTALLayer with GLU gating, pre-norm, skip connections
  - FRACTALModel for classification (stacked layers + pooling + classifier)
  - Tested: forward pass works, shapes correct, ~4M params (full config)
- **lra_datasets.py**: Data loaders for ListOps (synthetic), IMDB (HuggingFace), sCIFAR-10
- **train.py**: Training loop with AdamW, cosine schedule, warmup, AMP, gradient clipping

## Failed Approaches
- Direct eigendecomposition for large N: V is ill-conditioned (cond ~10^76 for N=64)
  - Fall back to normalized B for B̃ initialization
  - Paper says "asymptotic performance is comparable" with random init
- LRA dataset download from Google Cloud: 403 Forbidden
  - Used alternatives: synthetic ListOps, HuggingFace IMDB, torchvision CIFAR-10
- torch.compile on model: fails due to complex tensor view_as_real conjugate issue
  - Not critical; model runs at acceptable speed without compilation

## Performance Estimates (full model: d=256, N=64, layers=6)
- sCIFAR (bs=50, L=1024): ~0.4s/step, 900 steps/epoch, ~6min/epoch, ~20h for 200 epochs
- ListOps (bs=32, L=2048): would be ~2x slower due to longer sequences
- Text (bs=16, L=4096): would be ~4x slower due to 4096 length

## Plan for Remaining Work
1. Run sCIFAR-10 for ~30 epochs (feasible in ~3h) — should show meaningful learning
2. Run ListOps for ~10-15 epochs (feasible in ~2-3h)
3. Generate Figure 2 (A matrix heatmap) — purely from fractal_init.py, no training needed
4. Generate Figure 1 (memory measure) — requires impulse response computation
5. Write reproduce.sh, REPORT.md
6. Final cleanup and push

## Evaluation Coverage
- Main result: LRA benchmark Table 1 (6 tasks) — will have 2-3 tasks with reduced epochs
- Numerical verification: A matrix heatmap (Figure 2) — NOT YET, EASY to do
- Memory measure visualization (Figure 1) — NOT YET
- Ablation (Table 2): multi-α vs single α — could do quickly if time permits
