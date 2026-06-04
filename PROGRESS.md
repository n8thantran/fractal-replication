# FRACTAL Paper Replication Progress

## Current Phase
Model implemented and tested. Now building LRA data loading and training loop.
LRA dataset download from Google Cloud failed (403 Forbidden) — need alternative approach.

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
- [ ] Implement LRA data loading (feasible tasks)
- [ ] Training loop with proper hyperparameters
- [ ] Run experiments on at least ListOps, Text, Image
- [ ] Generate A matrix heatmap visualization (Figure 2)
- [ ] Generate memory measure visualization (Figure 1)
- [ ] Write reproduce.sh and REPORT.md

## Key Decisions
- Using PyTorch (paper used JAX)
- For ill-conditioned V: use diagonal eigenvalues directly (1,2,...,N)
- B̃_init may need normalization; paper notes random init converges to same final accuracy
- α config: K=8 blocks with [0,0,0.3,0.3,0.5,0.5,0.9,0.9]

## Completed Work
- fractal_init.py: A(α) and B(α) computation, tested and working
  - A(α=0) matches HiPPO-LegS exactly
  - Diagonal always n+1, lower triangular verified
  - B(α=0) = sqrt(2n+1) verified
- model.py: Full FRACTAL model with:
  - DiagonalSSMLayer with ZOH discretization and parallel scan
  - FRACTALLayer with GLU gating, pre-norm, skip connections
  - FRACTALModel for classification (stacked layers + pooling + classifier)
  - FRACTALModelLinear for linear recurrence mode
  - Tested: forward pass works, shapes correct

## Failed Approaches
- Direct eigendecomposition for large N: V is ill-conditioned (cond ~10^11 for N=16)
  - Won't use V^{-1}B directly for large N; will need alternative approach
  - Paper says "asymptotic performance is comparable" with random init
- LRA dataset download from https://storage.googleapis.com/long-range-arena/lra_release.gz
  - Returns 403 Forbidden
  - Need alternative: generate ListOps synthetically, IMDB from HF, CIFAR from torchvision

## Data Loading Strategy (revised)
For LRA tasks:
1. **ListOps** (seq_len=2048): Generate synthetically - well defined task
2. **Text/IMDB** (seq_len=4096): HuggingFace stanfordnlp/imdb, byte-level tokenization
3. **Image/sCIFAR** (seq_len=1024): torchvision CIFAR-10, grayscale, flatten
4. **Retrieval** (seq_len=4000): AAN dataset - may need to skip or simplify
5. **Pathfinder** (seq_len=1024): Hard to generate - may skip
6. **Path-X** (seq_len=16384): Hard to generate - may skip

Priority: ListOps, Text, Image (3/6 tasks)

## Evaluation Coverage
- Main result: LRA benchmark Table 1 (6 tasks) - IN PROGRESS
- Numerical verification: A matrix heatmap (Figure 2) - NOT YET
- Memory measure visualization (Figure 1) - NOT YET
