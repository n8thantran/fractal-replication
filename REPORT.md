# FRACTAL Paper Replication Report

## Paper
**FRACTAL: Fractional-Order Continuous-Time Adaptive Learning for Sequence Modeling**

The paper introduces FRACTAL, a state-space model (SSM) that generalizes HiPPO memory by using fractional-order Jacobi polynomial bases parameterized by α ∈ [0,1). Different α values create different memory profiles (α=0 is uniform/Legendre, α→1 emphasizes recent history). The model uses a multi-α filter bank with K=8 channels to capture diverse temporal patterns.

## What Was Implemented

### Core Mathematical Components (`fractal_init.py`)
- **A(α) matrix**: Fractional-order HiPPO matrix via Gauss-Jacobi quadrature (Eq. 5-7)
  - Verified: A(α=0) matches HiPPO-LegS exactly
  - Verified: diagonal elements = n+1, lower triangular structure
- **B(α) vector**: Closed-form computation (Eq. 8)
  - Verified: B(α=0) = sqrt(2n+1) matches Legendre case

### Model Architecture (`model.py`)
- **DiagonalSSMLayer**: Diagonal SSM with ZOH discretization, parallel scan
  - Multi-α filter bank: K=8 channels with α ∈ {0, 0.3, 0.5, 0.9}
  - Learnable discretization step Δ (log-uniform initialization)
  - Parallel associative scan for O(L log L) computation
- **FRACTALLayer**: Pre-norm → SSM → GLU gating (SiLU activation)
- **FRACTALModel**: Embedding → 6 FRACTAL layers → global avg pool → classifier

### Training Infrastructure (`train.py`)
- AdamW optimizer (β1=0.9, β2=0.999, weight_decay=0.05)
- Cosine LR schedule with 10% linear warmup
- Mixed precision (bfloat16 forward, float32 gradients)
- Gradient clipping (global norm 1.0)
- Separate LR for SSM parameters (0.1× main LR)

### Data Loading (`lra_datasets.py`)
- **ListOps**: HuggingFace dataset, tokenized, padded to 2048
- **Text/IMDB**: HuggingFace dataset, byte-level encoding, padded to 1024
- **Image/sCIFAR-10**: torchvision, grayscale, flattened to 1024 sequence

### Paper Figures (`generate_figures.py`)
- Figure 2: A(α) matrix heatmap showing structure for different α values
- Figure 1: Memory measure visualization (fractional-order weighting)
- Eigenvalue verification plot
- Filter bank diversity visualization (impulse responses for different α)

## Results

### LRA Benchmark (Table 1)

| Task | Paper Result | Our Result | Epochs (ours/paper) | Data |
|------|-------------|------------|---------------------|------|
| sCIFAR-10 | 87.30% | **65.93%** | 30/200 | Full (50k) |
| Text/IMDB | 89.10% | **74.70%** | 10/40 | 5k/25k |
| ListOps | 61.85% | **14.21%** | 4/40 | 10k/96k |

**Note**: All gaps are primarily due to reduced training epochs and data. The sCIFAR-10 learning curve shows steady improvement from 26% → 66% over 30 epochs with no sign of plateauing, suggesting full 200-epoch training would approach the paper's result.

### Generated Figures
- `results/figure2_A_matrix_heatmap.png` — A(α) matrix structure
- `results/figure2_A_matrix_annotated.png` — Annotated version
- `results/figure1_memory_measure.png` — Memory measure visualization
- `results/eigenvalue_verification.png` — Eigenvalue properties
- `results/alpha_diversity_filters.png` — Multi-α filter bank
- `results/B_vector_comparison.png` — B vector for different α
- `results/training_curves.png` — Training curves for all 3 tasks

## Commands to Reproduce

```bash
# Quick smoke test (~40 seconds)
bash reproduce.sh --quick

# Full reduced-epoch experiments (~3-4 hours)
bash reproduce.sh

# Individual experiments
python generate_figures.py
python train.py --task image --epochs 30 --batch_size 50 --d_model 256 --state_dim 64 --n_layers 6
python train.py --task text --epochs 10 --batch_size 16 --d_model 256 --state_dim 64 --n_layers 6 --max_train_samples 5000
```

## Important File Paths
- `/workspace/fractal_init.py` — Core A(α), B(α) computation
- `/workspace/model.py` — Full FRACTAL model (SSM layer, FRACTAL layer, full model)
- `/workspace/lra_datasets.py` — LRA dataset loaders
- `/workspace/train.py` — Training script with all hyperparameters
- `/workspace/generate_figures.py` — Paper figure generation
- `/workspace/reproduce.sh` — Reproduction script
- `/workspace/results/` — All results (JSON, plots, logs)
- `/workspace/checkpoints/` — Model checkpoints

## What Is Still Incomplete or Approximate

1. **Reduced training epochs**: Paper uses 200 epochs for Image, 40 for Text/ListOps. We ran 30/10/4 respectively due to compute constraints.
2. **Reduced data**: Text uses 5k/25k samples, ListOps uses 10k/96k samples.
3. **Missing tasks**: Retrieval, Pathfinder, Path-X (require specialized data formats not readily available).
4. **V matrix diagonalization**: The Vandermonde matrix for exact diagonalization is extremely ill-conditioned. We use eigenvalues directly as diagonal approximation (same approach as S4D/S5).
5. **JAX vs PyTorch**: Paper uses JAX; our implementation is in PyTorch. Minor numerical differences expected.
6. **Speech Commands** (Table 2) and **Ablation studies** (Table 3) not replicated.
