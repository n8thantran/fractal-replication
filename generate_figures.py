"""
Generate visualizations from the FRACTAL paper:
- Figure 2: A(α) matrix heatmap for different α values
- Figure 1: Memory measure / impulse response comparison
- Additional: B(α) vector visualization
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fractal_init import compute_A_matrix, compute_B_vector
import os

os.makedirs('results', exist_ok=True)


def generate_figure2_heatmap():
    """
    Figure 2: Heatmap of A(α) matrices for α ∈ {0, 0.3, 0.5, 0.9}.
    Shows the lower-triangular structure with diagonal n+1.
    """
    N = 32  # Moderate size for visualization
    alphas = [0.0, 0.3, 0.5, 0.9]
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    for i, alpha in enumerate(alphas):
        A = compute_A_matrix(N, alpha)
        # Negate to show -A (the SSM convention is dx/dt = -Ax + Bu)
        A_neg = -A
        
        # Use symmetric colormap centered at 0
        vmax = max(abs(A_neg.min()), abs(A_neg.max()))
        im = axes[i].imshow(A_neg, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='equal')
        axes[i].set_title(f'$-A(\\alpha={alpha})$', fontsize=14)
        axes[i].set_xlabel('Column index $k$')
        if i == 0:
            axes[i].set_ylabel('Row index $n$')
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    
    plt.suptitle('FRACTAL: HiPPO A matrices for different fractional orders α', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('results/figure2_A_matrix_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Generated results/figure2_A_matrix_heatmap.png")
    
    # Also save individual smaller matrices for inspection
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    N_small = 8
    for i, alpha in enumerate(alphas):
        A = compute_A_matrix(N_small, alpha)
        vmax_s = max(abs((-A).min()), abs((-A).max()))
        im = axes[i].imshow(-A, cmap='RdBu_r', vmin=-vmax_s, vmax=vmax_s, aspect='equal')
        axes[i].set_title(f'$-A(\\alpha={alpha})$, N={N_small}', fontsize=12)
        
        # Add text annotations for small matrix
        for row in range(N_small):
            for col in range(N_small):
                val = -A[row, col]
                text = f'{val:.1f}' if abs(val) > 0.05 else ''
                axes[i].text(col, row, text, ha='center', va='center', fontsize=7,
                           color='white' if abs(val) > vmax*0.5 else 'black')
        
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    
    plt.suptitle('FRACTAL: A matrices (annotated, N=8)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('results/figure2_A_matrix_annotated.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Generated results/figure2_A_matrix_annotated.png")


def generate_B_vector_plot():
    """Visualize B(α) vectors for different α values."""
    N = 64
    alphas = [0.0, 0.3, 0.5, 0.9]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for alpha in alphas:
        B = compute_B_vector(N, alpha)
        ax.plot(range(N), B, label=f'α={alpha}', linewidth=2)
    
    ax.set_xlabel('State index n', fontsize=12)
    ax.set_ylabel('B(α)_n', fontsize=12)
    ax.set_title('FRACTAL: B(α) vectors for different fractional orders', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/B_vector_comparison.png', dpi=150)
    plt.close()
    print("✓ Generated results/B_vector_comparison.png")


def generate_figure1_memory():
    """
    Figure 1: Memory measure / impulse response for different α values.
    
    The memory measure shows how the SSM retains information over time.
    For continuous system dx/dt = -Ax + Bu, y = Cx:
    - Impulse response h(t) = C * exp(-At) * B
    
    For the diagonal approximation:
    - h(t) = sum_n C_n * exp(-λ_n * t) * B_n
    
    Different α values give different memory profiles.
    """
    N = 64
    alphas = [0.0, 0.3, 0.5, 0.9]
    T = 200  # Time steps
    dt = 0.01  # Step size
    t = np.arange(T) * dt
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Impulse response magnitude over time
    for alpha in alphas:
        # Get eigenvalues (1, 2, ..., N for diagonal)
        eigenvalues = np.arange(1, N + 1, dtype=np.float64)
        B = compute_B_vector(N, alpha)
        
        # C chosen as uniform for visualization
        C = np.ones(N) / np.sqrt(N)
        
        # Impulse response: h(t) = sum_n C_n * exp(-λ_n * t) * B_n
        response = np.zeros(T)
        for step in range(T):
            exp_vals = np.exp(-eigenvalues * t[step])
            response[step] = np.abs(np.sum(C * exp_vals * B))
        
        # Normalize
        response = response / (response[0] + 1e-10)
        axes[0].plot(t, response, label=f'α={alpha}', linewidth=2)
    
    axes[0].set_xlabel('Time', fontsize=12)
    axes[0].set_ylabel('Normalized impulse response |h(t)|', fontsize=12)
    axes[0].set_title('Impulse Response Decay', fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')
    
    # Plot 2: Memory measure M(T) = ∫₀ᵀ |h(t)|² dt / ∫₀^∞ |h(t)|² dt  
    # This shows what fraction of total memory is captured up to time T
    T_long = 1000
    t_long = np.arange(T_long) * dt
    
    for alpha in alphas:
        eigenvalues = np.arange(1, N + 1, dtype=np.float64)
        B = compute_B_vector(N, alpha)
        C = np.ones(N) / np.sqrt(N)
        
        # Compute |h(t)|² for each t
        h_squared = np.zeros(T_long)
        for step in range(T_long):
            exp_vals = np.exp(-eigenvalues * t_long[step])
            h_squared[step] = np.sum(C * exp_vals * B)**2
        
        # Cumulative measure (normalized)
        cumsum = np.cumsum(h_squared) * dt
        total = cumsum[-1]
        memory_measure = cumsum / (total + 1e-10)
        
        axes[1].plot(t_long, memory_measure, label=f'α={alpha}', linewidth=2)
    
    axes[1].set_xlabel('Time horizon T', fontsize=12)
    axes[1].set_ylabel('Memory measure M(T)', fontsize=12)
    axes[1].set_title('Cumulative Memory Measure', fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('FRACTAL: Memory characteristics for different fractional orders α', fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig('results/figure1_memory_measure.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Generated results/figure1_memory_measure.png")


def generate_eigenvalue_comparison():
    """
    Show A(α) eigenvalue structure for different α.
    Paper claims eigenvalues are always {1, 2, ..., N} regardless of α.
    """
    N = 32
    alphas = [0.0, 0.3, 0.5, 0.9]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for alpha in alphas:
        A = compute_A_matrix(N, alpha)
        eigenvalues = np.linalg.eigvals(A)
        eigenvalues = np.sort(eigenvalues.real)
        
        axes[0].plot(range(N), eigenvalues, 'o-', label=f'α={alpha}', markersize=4, linewidth=1)
        
        # Error from expected {1, 2, ..., N}
        expected = np.arange(1, N + 1, dtype=np.float64)
        error = np.abs(eigenvalues - expected)
        axes[1].plot(range(N), error, 'o-', label=f'α={alpha}', markersize=4, linewidth=1)
    
    axes[0].plot(range(N), np.arange(1, N+1), 'k--', label='Expected (n+1)', alpha=0.5)
    axes[0].set_xlabel('Index', fontsize=12)
    axes[0].set_ylabel('Eigenvalue', fontsize=12)
    axes[0].set_title('A(α) Eigenvalues', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Index', fontsize=12)
    axes[1].set_ylabel('|λ_n - (n+1)|', fontsize=12)
    axes[1].set_title('Eigenvalue Error (deviation from n+1)', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')
    
    plt.suptitle('FRACTAL: Eigenvalue structure verification', fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig('results/eigenvalue_verification.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Generated results/eigenvalue_verification.png")


def generate_alpha_diversity_figure():
    """
    Show how different α values create diverse filter characteristics.
    This is the key insight of FRACTAL: multi-α → diverse spectral filters.
    """
    N = 64
    alpha_list = [0, 0, 0.3, 0.3, 0.5, 0.5, 0.9, 0.9]
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    for i, alpha in enumerate(alpha_list):
        row, col = i // 4, i % 4
        
        eigenvalues = np.arange(1, N + 1, dtype=np.float64)
        B = compute_B_vector(N, alpha)
        
        # Show frequency response: |H(ω)| = |C(iωI - Λ)^{-1}B|
        # For visualization, use C = ones / sqrt(N)
        C = np.ones(N) / np.sqrt(N)
        
        freqs = np.logspace(-2, 2, 500)
        H = np.zeros(len(freqs))
        for fi, f in enumerate(freqs):
            omega = 2 * np.pi * f
            # H(jω) = sum_n C_n * B_n / (jω + λ_n)
            transfer = np.sum(C * B / (1j * omega + eigenvalues))
            H[fi] = np.abs(transfer)
        
        axes[row, col].plot(freqs, H, linewidth=2, color=plt.cm.viridis(alpha))
        axes[row, col].set_xscale('log')
        axes[row, col].set_yscale('log')
        axes[row, col].set_title(f'Channel {i}: α={alpha}', fontsize=12)
        axes[row, col].set_xlabel('Frequency')
        axes[row, col].set_ylabel('|H(jω)|')
        axes[row, col].grid(True, alpha=0.3)
    
    plt.suptitle('FRACTAL: Multi-α filter bank frequency responses', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('results/alpha_diversity_filters.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Generated results/alpha_diversity_filters.png")


if __name__ == '__main__':
    print("Generating FRACTAL paper figures...\n")
    
    generate_figure2_heatmap()
    generate_B_vector_plot()
    generate_figure1_memory()
    generate_eigenvalue_comparison()
    generate_alpha_diversity_figure()
    
    print("\nAll figures generated in results/")
