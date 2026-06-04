# FRACTAL Paper Replication Progress

## Current Phase
**COMPLETE** - All deliverables ready. Final verification and commit needed.

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
- [x] Implement A(α) matrix computation (Gauss-Jacobi quadrature)
- [x] Implement B(α) vector computation (closed-form)
- [x] Implement diagonal SSM layer with ZOH discretization
- [x] Implement parallel scan (associative scan in PyTorch)
- [x] Implement FRACTAL layer (GLU gating, pre-norm)
- [x] Implement full FRACTAL model (stacked layers, encoder/decoder)
- [x] Implement LRA data loading (ListOps, IMDB, sCIFAR-10)
- [x] Training loop with proper hyperparameters
- [x] Smoke test: 1-epoch image task works
- [x] Generate figures: A matrix heatmap, memory measure, eigenvalue verification, filter bank
- [x] sCIFAR-10 training: 65.93% test acc (30 epochs)
- [x] Text/IMDB training: 74.70% test acc (10 epochs, 5k samples)
- [x] ListOps training: 14.21% val acc (4 epochs, 10k samples)
- [x] Training curves plot
- [x] reproduce.sh written and verified (--quick mode works in ~40s)
- [x] REPORT.md written
- [x] Results JSON files with full-run data
- [x] Final commit and push

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
- **model.py**: Full FRACTAL model
  - DiagonalSSMLayer: ZOH discretization, parallel scan, multi-α filter bank
  - FRACTALLayer: pre-norm, SSM, GLU gating with SiLU
  - FRACTALModel: embedding, stacked layers, pooling, classification head
- **lra_datasets.py**: LRA dataset loaders
  - ListOps: from HuggingFace, tokenized, padded to 2048
  - Text/IMDB: from HuggingFace, byte-level encoding, padded to 1024
  - Image/sCIFAR-10: from torchvision, flattened to 1024 sequence
- **train.py**: Training script
  - AdamW with separate LR for SSM params
  - Cosine schedule with linear warmup
  - Mixed precision (bfloat16)
  - Gradient clipping
  - Checkpoint saving
- **generate_figures.py**: Paper figure reproduction
  - Figure 2: A(α) matrix heatmap
  - Figure 1: Memory measure visualization
  - Eigenvalue verification
  - Filter bank diversity visualization
- **reproduce.sh**: Reproduction script
  - --quick mode: smoke test in ~40s
  - Default mode: reduced-epoch experiments (~3-4 hours)

## Failed Approaches
- V matrix inversion for diagonalization: extremely ill-conditioned (rcond ~1e-77)
  - Solution: use eigenvalues directly as diagonal approximation
- Full LRA data download: lra_release.gz was empty/corrupted
  - Solution: use HuggingFace datasets for ListOps/IMDB, torchvision for CIFAR-10
- Full-epoch training: would take 50+ GPU hours
  - Solution: reduced epochs, clearly documented gap

## Evaluation Coverage
### Addressed:
- Core method: fractional-order HiPPO A(α), B(α) matrices (Section 3)
- Multi-α filter bank architecture (Section 4)
- Diagonal SSM with ZOH discretization (Section 4)
- GLU gating, pre-norm architecture (Section 4)
- LRA benchmark: 3 of 6 tasks (Image, Text, ListOps)
- Paper figures: A matrix structure, memory measure, eigenvalue properties

### Not addressed:
- Retrieval, Pathfinder, Path-X tasks (data not available)
- Full-epoch training (compute constraints)
- Exact numerical match to paper results (reduced epochs + PyTorch vs JAX)
- Speech Commands experiment (Table 2)
- Ablation studies (Table 3)
