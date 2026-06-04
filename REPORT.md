# FRACTAL Paper Replication Report

## Paper
**FRACTAL: Fractional Measures for Long-Range Sequence Modeling**

The paper introduces FRACTAL, a state-space model (SSM) that generalizes the HiPPO framework through fractional calculus. Key contributions:
1. A new family of A(α) matrices parameterized by fractional order α, computed via Gauss-Jacobi quadrature
2. Multi-α filter banks that combine diverse temporal dynamics
3. Competitive performance on the Long Range Arena (LRA) benchmark

## What Was Implemented

### Core Components
1. **`fractal_init.py`**: Computes A(α) and B(α) matrices using Gauss-Jacobi quadrature (Eq. 3 in paper)
   - Verified A(α=0) matches HiPPO-LegS exactly (lower-triangular, diagonal = n+1)
   - B(α=0) = sqrt(2n+1) verified against known HiPPO-LegS result
   - Tested for α ∈ {0, 0.3, 0.5, 0.9}

2. **`model.py`**: Full FRACTAL architecture
   - `DiagonalSSMLayer`: Diagonal state-space model with ZOH discretization
   - `parallel_scan`: Associative scan for efficient recurrence computation
   - `FRACTALLayer`: Complete layer with multi-α filter bank, GLU gating, LayerNorm
   - `FRACTALModel`: Stacked layers with encoder (embedding/linear) and classification head
   - Multi-α config: K=8 channels with α ∈ [0, 0, 0.3, 0.3, 0.5, 0.5, 0.9, 0.9]

3. **`lra_datasets.py`**: LRA benchmark data loading
   - ListOps: Synthetic generation (seq_len=2048)
   - Text/IMDB: Byte-level encoding from HuggingFace datasets (seq_len=1024)
   - sCIFAR-10: Sequential pixel classification (seq_len=1024)
   - Retrieval: AAN dataset (seq_len=4000)

4. **`train.py`**: Training infrastructure
   - AdamW optimizer with cosine LR schedule + linear warmup
   - Mixed precision (bfloat16) training
   - Gradient clipping (global norm 1.0)
   - All paper hyperparameters configurable via CLI

5. **`generate_figures.py`**: Paper figure reproduction
   - Figure 2: A(α) matrix heatmaps for α = {0, 0.3, 0.5, 0.9}
   - Figure 1: Memory measure visualization
   - Eigenvalue spectrum verification
   - Filter bank diversity across α values

### Model Parameters
- d_model=256, state_dim=64, n_layers=6
- ~5.77M parameters (consistent with paper's efficient design)

## Experiments Run

### 1. sCIFAR-10 (Sequential Image Classification)
- **Config**: bs=50, lr=0.001, wd=0.05, 30 epochs (paper uses 200)
- **Result**: [UPDATING - training in progress]
- **Paper target**: 87.30%
- **Note**: Running 15% of full training schedule due to compute constraints

### 2. Paper Figures
All figures generated successfully:
- `results/figure2_A_matrix_heatmap.png` — A(α) matrix structure
- `results/figure2_A_matrix_annotated.png` — Annotated small A matrices
- `results/B_vector_comparison.png` — B vector across α values
- `results/figure1_memory_measure.png` — Memory measure visualization
- `results/eigenvalue_verification.png` — Eigenvalue spectrum
- `results/alpha_diversity_filters.png` — Filter bank diversity

## Key Result Files
- `results/image_results.json` — sCIFAR-10 training results
- `results/listops_results.json` — ListOps training results (if completed)
- `results/*.png` — All generated figures
- `checkpoints/image_best.pt` — Best sCIFAR-10 model checkpoint

## How to Reproduce
```bash
# Quick test (few epochs)
bash reproduce.sh --quick

# Standard run (reduced epochs, ~5 hours)
bash reproduce.sh
```

## Implementation vs Paper Differences
1. **Framework**: PyTorch instead of JAX/Flax
2. **Training duration**: Reduced epochs (30 vs 200 for image, 15 vs 40 for ListOps) due to compute constraints
3. **Parallel scan**: Custom sequential implementation (JAX has built-in `jax.lax.associative_scan`)
4. **Data**: Synthetic ListOps generation instead of pre-generated LRA dataset (original URL returned 403)

## What's Still Incomplete
- Full 200-epoch training for convergence to paper's reported numbers
- Pathfinder and Path-X tasks (very long sequences, need more compute)
- Retrieval task (dual-input architecture needed)
- Ablation studies from Section 4.2

## Verification of Mathematical Correctness
The core mathematical contribution is verified:
- A(α=0) exactly reproduces HiPPO-LegS matrix
- A(α) is lower-triangular for all α as expected
- Diagonal elements are always (n+1) regardless of α
- B(α=0) = sqrt(2n+1) matches the known closed-form
- Eigenvalue spectrum shows all eigenvalues are real and negative (stable)
- Different α values produce visually distinct filter responses
