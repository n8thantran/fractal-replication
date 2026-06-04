"""
FRACTAL Model: Full implementation of the Fractional SSM architecture.

Architecture:
- Embedding layer (task-specific)
- Stack of FRACTAL layers (each: LayerNorm -> SSM -> GLU gate)
- Classification head

Each SSM layer:
- Diagonal complex state: Λ = -Λ_real + iΛ_imag
- ZOH discretization: Ā = exp(ΔΛ), B̄ = Λ^{-1}(Ā - I)B̃
- Parallel scan for training
- Multi-α filter bank: K channels with different α values
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial


def associative_scan(gates, values):
    """
    Parallel associative scan for linear recurrence.
    
    Given gates a_t and values b_t, computes:
        h_t = a_t * h_{t-1} + b_t
    
    Uses the Blelloch parallel scan algorithm.
    
    Args:
        gates: (batch, length, state_dim) - complex
        values: (batch, length, state_dim) - complex
    Returns:
        outputs: (batch, length, state_dim) - h_t for each t
    """
    # Sequential fallback for correctness (will optimize later)
    B, L, N = gates.shape
    outputs = torch.zeros_like(values)
    h = torch.zeros(B, N, dtype=values.dtype, device=values.device)
    for t in range(L):
        h = gates[:, t] * h + values[:, t]
        outputs[:, t] = h
    return outputs


def parallel_scan(gates, values):
    """
    Efficient parallel scan using log(L) steps.
    
    For the recurrence h_t = a_t * h_{t-1} + b_t,
    the associative operator is: (a2, b2) * (a1, b1) = (a2*a1, a2*b1 + b2)
    
    Args:
        gates: (batch, length, state_dim) - complex  
        values: (batch, length, state_dim) - complex
    Returns:
        outputs: (batch, length, state_dim)
    """
    B, L, N = gates.shape
    
    # Pad to power of 2
    log2L = int(math.ceil(math.log2(L)))
    L_pad = 2 ** log2L
    
    if L_pad > L:
        pad_gates = torch.ones(B, L_pad - L, N, dtype=gates.dtype, device=gates.device)
        pad_values = torch.zeros(B, L_pad - L, N, dtype=values.dtype, device=values.device)
        gates = torch.cat([gates, pad_gates], dim=1)
        values = torch.cat([values, pad_values], dim=1)
    
    # Up sweep (reduction)
    a = gates.clone()
    b = values.clone()
    
    # Store intermediate results for down sweep
    a_intermediates = []
    b_intermediates = []
    
    for d in range(log2L):
        stride = 2 ** d
        a_intermediates.append(a.clone())
        b_intermediates.append(b.clone())
        
        # Apply: (a[i], b[i]) = (a[i] * a[i-stride], a[i] * b[i-stride] + b[i])
        # Only for indices where i % (2*stride) == 2*stride - 1
        step = 2 * stride
        # Vectorized: operate on pairs
        idx = torch.arange(stride - 1, L_pad, step, device=gates.device)
        idx_partner = idx + stride
        
        valid = idx_partner < L_pad
        idx = idx[valid]
        idx_partner = idx_partner[valid]
        
        a_left = a[:, idx]   # a[i - stride]
        b_left = b[:, idx]   # b[i - stride]
        a_right = a[:, idx_partner]  # a[i]
        b_right = b[:, idx_partner]  # b[i]
        
        a[:, idx_partner] = a_right * a_left
        b[:, idx_partner] = a_right * b_left + b_right
    
    # Down sweep
    # Set the last element's "prefix" to identity
    # Actually, let's use a simpler approach: just do the standard parallel scan
    # The above is the reduce phase. Now we need the down sweep.
    
    # Actually, let me use a cleaner implementation based on the standard algorithm
    # Reset and use the simple log(L) approach
    
    # Method: iterative doubling
    # h[t] = a[t]*h[t-1] + b[t]
    # After k steps: h[t] depends on h[t-2^k]
    
    a = gates.clone()
    b = values.clone()
    
    for d in range(log2L):
        stride = 2 ** d
        a_shifted = torch.zeros_like(a)
        b_shifted = torch.zeros_like(b)
        
        a_shifted[:, stride:] = a[:, :-stride] if stride < L_pad else torch.zeros_like(a[:, :0])
        b_shifted[:, stride:] = b[:, :-stride] if stride < L_pad else torch.zeros_like(b[:, :0])
        
        # For positions >= stride: combine with shifted version
        # (a[t], b[t]) * (a[t-stride], b[t-stride]) = (a[t]*a[t-stride], a[t]*b[t-stride] + b[t])
        new_b = a * b_shifted + b
        new_a = a * a_shifted
        
        # Only update positions >= stride (identity for others doesn't change)
        mask = torch.zeros(L_pad, dtype=torch.bool, device=gates.device)
        mask[stride:] = True
        
        a = torch.where(mask.unsqueeze(0).unsqueeze(-1), new_a, a)
        b = torch.where(mask.unsqueeze(0).unsqueeze(-1), new_b, b)
    
    return b[:, :L]


class FRACTALInit:
    """Compute FRACTAL initialization parameters."""
    
    @staticmethod
    def compute_B_vector(N, alpha):
        """B_n = sqrt((2n+1-α)/(1-α)) * C(n-α, n)"""
        from scipy.special import gamma
        B = np.zeros(N)
        for n in range(N):
            gamma_n = np.sqrt((2*n + 1 - alpha) / (1 - alpha))
            binom_coeff = gamma(n + 1 - alpha) / (gamma(1 - alpha) * gamma(n + 1))
            B[n] = gamma_n * binom_coeff
        return B
    
    @staticmethod
    def compute_A_matrix(N, alpha, num_quad_points=128):
        """Compute A(α) via Gauss-Jacobi quadrature."""
        from scipy.special import gamma, eval_jacobi, roots_jacobi
        
        A = np.zeros((N, N))
        for n in range(N):
            A[n, n] = n + 1
        
        if alpha == 0:
            for n in range(N):
                for k in range(n):
                    A[n, k] = np.sqrt((2*n+1) * (2*k+1))
            return A
        
        nodes, weights = roots_jacobi(num_quad_points, -alpha, 0)
        
        P_vals = np.zeros((N, num_quad_points))
        P_derivs = np.zeros((N, num_quad_points))
        
        for n in range(N):
            P_vals[n] = eval_jacobi(n, -alpha, 0, nodes)
            if n > 0:
                P_derivs[n] = (n - alpha + 1) / 2 * eval_jacobi(n-1, -alpha+1, 1, nodes)
        
        norms_sq = np.zeros(N)
        for k in range(N):
            a, b = -alpha, 0
            norms_sq[k] = (2**(a+b+1) / (2*k + a + b + 1) * 
                           gamma(k + a + 1) * gamma(k + b + 1) / 
                           (gamma(k + 1) * gamma(k + a + b + 1)))
        
        gamma_n = np.sqrt((2*np.arange(N) + 1 - alpha) / (1 - alpha))
        
        for n in range(N):
            L_Pn = P_vals[n] + (1 + nodes) * P_derivs[n]
            for k in range(n):
                inner_product = np.sum(weights * L_Pn * P_vals[k])
                A[n, k] = (gamma_n[n] / gamma_n[k]) * inner_product / norms_sq[k]
        
        return A
    
    @staticmethod
    def get_init_params(N_per_channel, alpha_list):
        """
        Get initialization parameters for multi-channel FRACTAL.
        
        Args:
            N_per_channel: state dimension per channel
            alpha_list: list of α values for each channel
        Returns:
            Lambda_real: (total_N,) real eigenvalues
            B_tilde_real: (total_N,) real part of B̃
        """
        K = len(alpha_list)
        total_N = N_per_channel * K
        
        Lambda_real_all = []
        B_tilde_all = []
        
        for alpha in alpha_list:
            # Eigenvalues are always 1, 2, ..., N regardless of α
            Lambda_real = np.arange(1, N_per_channel + 1, dtype=np.float64)
            Lambda_real_all.append(Lambda_real)
            
            # Compute B(α) vector
            B = FRACTALInit.compute_B_vector(N_per_channel, alpha)
            
            # For B̃ = V^{-1} B, try eigendecomposition but fall back to 
            # using B directly (normalized) if ill-conditioned
            try:
                from scipy import linalg
                A = FRACTALInit.compute_A_matrix(N_per_channel, alpha)
                eigenvalues, V = linalg.eig(A)
                idx = np.argsort(eigenvalues.real)
                V = V[:, idx]
                V_inv = linalg.inv(V)
                B_tilde = (V_inv @ B).real
                
                # Check if values are reasonable (not too large due to ill-conditioning)
                if np.max(np.abs(B_tilde)) > 1e6:
                    # Fall back to normalized B
                    B_tilde = B / np.linalg.norm(B) * np.sqrt(N_per_channel)
            except:
                B_tilde = B / np.linalg.norm(B) * np.sqrt(N_per_channel)
            
            B_tilde_all.append(B_tilde)
        
        Lambda_real_all = np.concatenate(Lambda_real_all)
        B_tilde_all = np.concatenate(B_tilde_all)
        
        return Lambda_real_all, B_tilde_all


class DiagonalSSM(nn.Module):
    """
    Diagonal State Space Model with FRACTAL initialization.
    
    State equation (continuous):
        dx/dt = Λx + B̃u
        y = C̃x + Du
    
    Discretized (ZOH):
        x_k = Ā x_{k-1} + B̄ u_k
        y_k = C̃ x_k + D u_k
    
    where Ā = exp(ΔΛ), B̄ = Λ^{-1}(Ā - I)B̃
    """
    
    def __init__(self, d_model, state_dim, alpha_list, dt_min=0.001, dt_max=0.1, use_parallel_scan=True):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim  # per channel
        self.K = len(alpha_list)
        self.total_state_dim = state_dim * self.K
        self.use_parallel_scan = use_parallel_scan
        
        # Get FRACTAL initialization
        Lambda_real_init, B_tilde_init = FRACTALInit.get_init_params(state_dim, alpha_list)
        
        # Learnable parameters
        # Lambda = -Lambda_real + i*Lambda_imag (diagonal of state matrix)
        # Lambda_real is initialized to eigenvalues, Lambda_imag to random
        self.Lambda_log_real = nn.Parameter(
            torch.log(torch.tensor(Lambda_real_init, dtype=torch.float32))
        )
        # Initialize imaginary part: uniform random like S5
        Lambda_imag_init = np.random.uniform(0, 2*np.pi, size=self.total_state_dim)
        self.Lambda_imag = nn.Parameter(
            torch.tensor(Lambda_imag_init, dtype=torch.float32)
        )
        
        # Learnable log(Δ) for ZOH discretization
        log_dt = np.random.uniform(np.log(dt_min), np.log(dt_max), size=(d_model,))
        self.log_dt = nn.Parameter(torch.tensor(log_dt, dtype=torch.float32))
        
        # B̃: (d_model, total_state_dim) - complex, initialized from FRACTAL
        # Each input dimension maps to the full state
        B_real_init = np.tile(B_tilde_init, (d_model, 1)) / np.sqrt(d_model)
        B_imag_init = np.zeros_like(B_real_init)
        self.B_real = nn.Parameter(torch.tensor(B_real_init, dtype=torch.float32))
        self.B_imag = nn.Parameter(torch.tensor(B_imag_init, dtype=torch.float32))
        
        # C̃: (d_model, total_state_dim) - complex
        C_real_init = np.random.randn(d_model, self.total_state_dim) / np.sqrt(self.total_state_dim)
        C_imag_init = np.random.randn(d_model, self.total_state_dim) / np.sqrt(self.total_state_dim)
        self.C_real = nn.Parameter(torch.tensor(C_real_init, dtype=torch.float32))
        self.C_imag = nn.Parameter(torch.tensor(C_imag_init, dtype=torch.float32))
        
        # D: skip connection (d_model,)
        self.D = nn.Parameter(torch.ones(d_model, dtype=torch.float32))
    
    def _get_discrete_params(self):
        """Compute discretized parameters via ZOH."""
        # Continuous diagonal: Lambda = -exp(log_real) + i*Lambda_imag
        Lambda_real = -torch.exp(self.Lambda_log_real)  # (total_state_dim,)
        Lambda = torch.complex(Lambda_real, self.Lambda_imag)  # (total_state_dim,)
        
        # Timescale
        dt = torch.exp(self.log_dt)  # (d_model,)
        
        # ZOH discretization
        # Ā = exp(Δ * Λ) - for diagonal, this is element-wise
        # We need dt to be broadcast: (d_model, 1) * (1, total_state_dim) -> not quite right
        # Actually in S5/FRACTAL, Δ is per-input-dim, and each state has its own Lambda
        # The discretization is: for each input dim h, Ā_h = exp(dt_h * Lambda)
        # But that makes Ā different per input dim, which is unusual.
        # 
        # In S5: Δ is a scalar (or per-state), and the system is MIMO:
        # x_{k+1} = Ā x_k + B̄ u_k where u_k is H-dim, x_k is N-dim
        # Ā = exp(Δ Λ) is N×N diagonal
        # B̄ = Λ^{-1}(Ā - I) B̃ where B̃ is N×H
        
        # Use a single Δ (mean of log_dt) or per-state Δ
        # Actually, let's follow S5 more carefully: Δ is (1,) or (N,)
        # Here we use dt as (d_model,) and broadcast appropriately
        
        # Simpler approach: dt is scalar-like, applied uniformly
        dt_mean = dt.mean()
        
        # Ā = exp(dt * Lambda), shape: (total_state_dim,)
        A_bar = torch.exp(dt_mean * Lambda)
        
        # B̄ = Lambda^{-1} (Ā - I) B̃, where B̃ is (d_model, total_state_dim)
        B_complex = torch.complex(self.B_real, self.B_imag)  # (d_model, total_state_dim)
        
        # Lambda^{-1} (Ā - I) is element-wise for diagonal Lambda
        scale = (A_bar - 1.0) / Lambda  # (total_state_dim,)
        B_bar = scale.unsqueeze(0) * B_complex  # (d_model, total_state_dim)
        
        # C̃
        C_complex = torch.complex(self.C_real, self.C_imag)  # (d_model, total_state_dim)
        
        return A_bar, B_bar, C_complex, dt_mean
    
    def forward(self, u):
        """
        Forward pass.
        
        Args:
            u: (batch, length, d_model) - input sequence
        Returns:
            y: (batch, length, d_model) - output sequence
        """
        B, L, H = u.shape
        
        A_bar, B_bar, C_complex, dt = self._get_discrete_params()
        
        # u: (B, L, H) -> compute B̄ᵀ u for each timestep
        # B_bar: (H, N_total) complex
        # Bu: (B, L, N_total) = u @ B_bar (complex matmul)
        u_complex = u.to(torch.complex64)
        Bu = u_complex @ B_bar  # (B, L, N_total)
        
        # Parallel scan: x_t = A_bar * x_{t-1} + Bu_t
        # gates: A_bar repeated for each timestep -> (B, L, N_total)
        gates = A_bar.unsqueeze(0).unsqueeze(0).expand(B, L, -1)
        
        if self.use_parallel_scan:
            x = parallel_scan(gates, Bu)  # (B, L, N_total)
        else:
            x = associative_scan(gates, Bu)
        
        # Output: y_t = Re(C̃ x_t) + D u_t
        # C_complex: (H, N_total)
        # x: (B, L, N_total)
        # y: (B, L, H) = x @ C_complex.T (take real part)
        y = torch.matmul(x, C_complex.T.conj()).real  # (B, L, H)
        
        # Skip connection
        y = y + self.D.unsqueeze(0).unsqueeze(0) * u  # (B, L, H)
        
        return y


class FRACTALLayer(nn.Module):
    """
    Single FRACTAL layer: LayerNorm -> SSM -> GLU gate
    
    z_out = (W_out * ssm(norm(z_in))) ⊙ σ(W_gate * z_in)
    Residual connection added.
    """
    
    def __init__(self, d_model, state_dim, alpha_list, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = DiagonalSSM(d_model, state_dim, alpha_list)
        self.out_proj = nn.Linear(d_model, d_model)
        self.gate_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: (batch, length, d_model)
        Returns:
            x: (batch, length, d_model)
        """
        residual = x
        z = self.norm(x)
        
        # SSM branch
        y = self.ssm(z)
        y = self.out_proj(y)
        
        # Gate branch (SiLU activation)
        gate = F.silu(self.gate_proj(z))
        
        # GLU combination
        out = y * gate
        out = self.dropout(out)
        
        return residual + out


class FRACTALModel(nn.Module):
    """
    Full FRACTAL model for sequence classification.
    
    Architecture:
    - Task-specific embedding
    - Stack of FRACTAL layers
    - Global average pooling (or special pooling for retrieval)
    - Classification head
    """
    
    def __init__(
        self,
        d_model=256,
        state_dim=64,
        n_layers=6,
        n_classes=10,
        vocab_size=None,
        max_seq_len=2048,
        alpha_list=None,
        dropout=0.0,
        task_type='classification',
        embedding_type='linear',  # 'linear', 'embedding', 'patch'
        input_dim=1,
        prenorm=True,
    ):
        super().__init__()
        self.d_model = d_model
        self.task_type = task_type
        self.prenorm = prenorm
        
        if alpha_list is None:
            alpha_list = [0, 0, 0.3, 0.3, 0.5, 0.5, 0.9, 0.9]
        
        # Embedding
        if embedding_type == 'embedding':
            self.encoder = nn.Embedding(vocab_size, d_model)
        elif embedding_type == 'linear':
            self.encoder = nn.Linear(input_dim, d_model)
        elif embedding_type == 'patch':
            # For image tasks - simple linear projection
            self.encoder = nn.Linear(input_dim, d_model)
        
        # Optional positional encoding (not used in S5/FRACTAL typically)
        # self.pos_enc = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
        
        # FRACTAL layers
        self.layers = nn.ModuleList([
            FRACTALLayer(d_model, state_dim, alpha_list, dropout=dropout)
            for _ in range(n_layers)
        ])
        
        # Final norm
        self.final_norm = nn.LayerNorm(d_model)
        
        # Classification head
        if task_type == 'retrieval':
            self.classifier = nn.Linear(2 * d_model, n_classes)
        else:
            self.classifier = nn.Linear(d_model, n_classes)
    
    def forward(self, x, x2=None):
        """
        Args:
            x: input tensor
                - For embedding: (batch, length) long tensor
                - For linear: (batch, length, input_dim) float tensor
            x2: second input for retrieval task
        Returns:
            logits: (batch, n_classes)
        """
        # Encode
        x = self.encoder(x)
        
        # Apply FRACTAL layers
        for layer in self.layers:
            x = layer(x)
        
        # Final norm
        x = self.final_norm(x)
        
        # Pool
        x = x.mean(dim=1)  # Global average pooling
        
        if self.task_type == 'retrieval' and x2 is not None:
            x2 = self.encoder(x2)
            for layer in self.layers:
                x2 = layer(x2)
            x2 = self.final_norm(x2)
            x2 = x2.mean(dim=1)
            x = torch.cat([x, x2], dim=-1)
        
        # Classify
        logits = self.classifier(x)
        return logits


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick test
    print("Testing FRACTAL model...")
    
    # Test parallel scan
    B, L, N = 2, 16, 4
    gates = torch.randn(B, L, N, dtype=torch.complex64) * 0.5
    values = torch.randn(B, L, N, dtype=torch.complex64)
    
    out_seq = associative_scan(gates, values)
    out_par = parallel_scan(gates, values)
    
    err = (out_seq - out_par).abs().max().item()
    print(f"Parallel scan error: {err:.2e}")
    assert err < 1e-4, f"Parallel scan mismatch: {err}"
    print("✓ Parallel scan matches sequential")
    
    # Test model forward pass
    model = FRACTALModel(
        d_model=64,
        state_dim=16,
        n_layers=2,
        n_classes=10,
        vocab_size=256,
        alpha_list=[0, 0.5, 0.9],
        embedding_type='embedding',
    )
    
    x = torch.randint(0, 256, (4, 128))
    logits = model(x)
    print(f"Model output shape: {logits.shape}")
    print(f"Parameters: {count_parameters(model):,}")
    assert logits.shape == (4, 10)
    print("✓ Model forward pass works")
    
    # Test with linear embedding (for image)
    model2 = FRACTALModel(
        d_model=64,
        state_dim=16,
        n_layers=2,
        n_classes=10,
        input_dim=1,
        alpha_list=[0, 0.5],
        embedding_type='linear',
    )
    
    x = torch.randn(4, 1024, 1)
    logits = model2(x)
    print(f"Linear model output shape: {logits.shape}")
    assert logits.shape == (4, 10)
    print("✓ Linear model forward pass works")
    
    print("\nAll tests passed!")
