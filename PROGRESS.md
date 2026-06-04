# FRACTAL Paper Replication Progress

## Current Phase
Starting implementation. Paper has been fully read and understood.

## Paper Summary
FRACTAL introduces a fractional-order measure into the HiPPO framework for state space models.
Key idea: use power-law measure μ(t)(x) = (1-α)/t^(1-α) * (t-x)^(-α) instead of uniform (LegS).
This gives Jacobi polynomial basis with parameters (-α, 0).

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

### Key Claims to Verify
1. A(α) matrix: lower triangular, diagonal = n+1 (invariant to α), off-diagonal computed via Gauss-Jacobi quadrature
2. B(α) vector: closed-form B_n = sqrt((2n+1-α)/(1-α)) * C(n-α, n)
3. Fractional filter bank with multiple α channels improves performance
4. A matrix heatmap visualization showing structure
5. Memory measure visualization

## Architecture Details (from paper)
- Based on S5 architecture (diagonal SSM with parallel scan)
- Model dim H=256, state dim N=64 per block
- K channels with α linearly spaced in [0, 0.9]
- LTI relaxation: drop 1/t factor, use learnable Δ
- ZOH discretization: Ā = exp(-ΔA), B̄ = A^{-1}(Ā - I)B
- Diagonalization: A = VΛV^{-1}, Λ_real = diag(1,2,...,N)
- Complex augmentation: Λ = -Λ_real + iΛ_imag
- B̃_init = V^{-1} * B(α)
- GLU gating: z_out = (W_out * y) ⊙ σ(W_gate * z_in)
- Parallel scan (associative scan) for training
- Pre-norm (LayerNorm) before SSM

## Implementation Plan
- [x] Read paper thoroughly
- [ ] Implement A(α) matrix computation (Gauss-Jacobi quadrature)
- [ ] Implement B(α) vector computation (closed-form)
- [ ] Implement eigendecomposition and B̃ initialization
- [ ] Implement diagonal SSM layer with ZOH discretization
- [ ] Implement parallel scan (associative scan in PyTorch)
- [ ] Implement FRACTAL layer (GLU gating, pre-norm)
- [ ] Implement full FRACTAL model (stacked layers, encoder/decoder)
- [ ] Implement LRA data loading (all 6 tasks)
- [ ] Training loop with proper hyperparameters
- [ ] Run experiments on at least ListOps and Text tasks
- [ ] Generate A matrix heatmap visualization
- [ ] Generate memory measure visualization
- [ ] Write reproduce.sh and REPORT.md

## Key Decisions
- Using PyTorch (paper used JAX but we have PyTorch + CUDA available)
- Will implement associative scan in PyTorch
- For LRA datasets: use HuggingFace datasets or download directly
- Hyperparams: follow S5 defaults where paper doesn't specify (lr, batch size, etc.)

## Completed Work
- Paper reading complete

## Failed Approaches
(none yet)

## Evaluation Coverage
- Main result: LRA benchmark Table 1 (6 tasks)
- Numerical verification: A matrix heatmap (Figure 2)
- Memory measure visualization (Figure 1)
