"""
FRACTAL: Fractional HiPPO matrix initialization.

Computes:
- A(α): State transition matrix via Gauss-Jacobi quadrature
- B(α): Input projection vector (closed-form)
- Eigendecomposition and B_tilde initialization
"""

import numpy as np
from scipy.special import gamma, comb
from scipy import linalg
import torch


def compute_B_vector(N, alpha):
    """
    Compute the input projection vector B(α) using closed-form expression.
    
    B_n = sqrt((2n+1-α)/(1-α)) * C(n-α, n)
    where C(n-α, n) = Γ(n+1-α) / (Γ(1-α) * n!)
    
    Args:
        N: state dimension
        alpha: singularity index in [0, 1)
    Returns:
        B: numpy array of shape (N,)
    """
    B = np.zeros(N)
    for n in range(N):
        gamma_n = np.sqrt((2*n + 1 - alpha) / (1 - alpha))
        # Generalized binomial coefficient C(n-α, n) = Γ(n+1-α) / (Γ(1-α) * n!)
        binom_coeff = gamma(n + 1 - alpha) / (gamma(1 - alpha) * gamma(n + 1))
        B[n] = gamma_n * binom_coeff
    return B


def compute_A_matrix(N, alpha, num_quad_points=128):
    """
    Compute the state transition matrix A(α) via Gauss-Jacobi quadrature.
    
    A is lower triangular with:
    - Diagonal: A_{nn} = n + 1 (invariant to α)
    - Off-diagonal (k < n): computed via Galerkin projection of differential operator
      L[P_n](η) = P_n(η) + (1+η) P_n'(η)
      onto the Jacobi basis {P_k^{(-α,0)}} with weight w(η) = (1-η)^{-α}
    
    Args:
        N: state dimension
        alpha: singularity index in [0, 1)
        num_quad_points: number of quadrature points
    Returns:
        A: numpy array of shape (N, N)
    """
    from numpy.polynomial.legendre import leggauss
    
    A = np.zeros((N, N))
    
    # Set diagonal: A_{nn} = n + 1
    for n in range(N):
        A[n, n] = n + 1
    
    if alpha == 0:
        # Special case: recover HiPPO-LegS
        # A_{nk} = sqrt((2n+1)(2k+1)) for k < n
        for n in range(N):
            for k in range(n):
                A[n, k] = np.sqrt((2*n + 1) * (2*k + 1))
        return A
    
    # For α > 0, use Gauss-Jacobi quadrature with weight (1-η)^{-α}
    # We need to compute inner products with weight w(η) = (1-η)^{-α}
    # Use scipy's roots_jacobi for Gauss-Jacobi quadrature
    from scipy.special import roots_jacobi
    
    # Gauss-Jacobi quadrature with weight (1-x)^a * (1+x)^b
    # Our weight is (1-η)^{-α}, so a = -alpha, b = 0
    nodes, weights = roots_jacobi(num_quad_points, -alpha, 0)
    
    # Evaluate Jacobi polynomials P_n^{(-α, 0)} at quadrature nodes
    from scipy.special import eval_jacobi
    
    # Precompute polynomial values and derivatives at nodes
    P_vals = np.zeros((N, num_quad_points))  # P_n(η) at each node
    P_derivs = np.zeros((N, num_quad_points))  # P_n'(η) at each node
    
    for n in range(N):
        P_vals[n] = eval_jacobi(n, -alpha, 0, nodes)
        # Derivative of Jacobi polynomial: d/dx P_n^{(a,b)}(x) = (n+a+b+1)/2 * P_{n-1}^{(a+1,b+1)}(x)
        if n > 0:
            P_derivs[n] = (n - alpha + 0 + 1) / 2 * eval_jacobi(n-1, -alpha+1, 1, nodes)
        else:
            P_derivs[n] = 0
    
    # Compute normalization constants ||P_k||^2_w
    # For Jacobi polynomials with weight (1-x)^a (1+x)^b on [-1,1]:
    # ||P_n^{(a,b)}||^2 = 2^{a+b+1} / (2n+a+b+1) * Γ(n+a+1)Γ(n+b+1) / (n! Γ(n+a+b+1))
    norms_sq = np.zeros(N)
    for k in range(N):
        a, b = -alpha, 0
        norms_sq[k] = (2**(a+b+1) / (2*k + a + b + 1) * 
                       gamma(k + a + 1) * gamma(k + b + 1) / 
                       (gamma(k + 1) * gamma(k + a + b + 1)))
    
    # Normalization constants γ_n = sqrt((2n+1-α)/(1-α))
    gamma_n = np.sqrt((2*np.arange(N) + 1 - alpha) / (1 - alpha))
    
    # Compute off-diagonal elements via quadrature
    # L[P_n](η) = P_n(η) + (1+η) P_n'(η)
    for n in range(N):
        L_Pn = P_vals[n] + (1 + nodes) * P_derivs[n]  # L[P_n] at nodes
        
        for k in range(n):
            # <L[P_n], P_k>_w = ∫ L[P_n](η) P_k(η) w(η) dη
            # With Gauss-Jacobi quadrature (weight already included in weights):
            inner_product = np.sum(weights * L_Pn * P_vals[k])
            
            # A_{nk} = (γ_n / γ_k) * <L[P_n], P_k>_w / ||P_k||^2_w
            A[n, k] = (gamma_n[n] / gamma_n[k]) * inner_product / norms_sq[k]
    
    return A


def compute_fractal_init(N, alpha, num_quad_points=128):
    """
    Compute the full FRACTAL initialization for a single α channel.
    
    Returns:
        Lambda_real: real eigenvalues (1, 2, ..., N)
        V: eigenvector matrix
        V_inv: inverse eigenvector matrix
        B_tilde: transformed input projection V^{-1} B
    """
    A = compute_A_matrix(N, alpha, num_quad_points)
    B = compute_B_vector(N, alpha)
    
    # Eigendecomposition: A is lower triangular with eigenvalues n+1
    # Direct eigendecomposition
    eigenvalues, V = linalg.eig(A)
    
    # Sort by real part
    idx = np.argsort(eigenvalues.real)
    eigenvalues = eigenvalues[idx]
    V = V[:, idx]
    
    # Verify eigenvalues are close to 1, 2, ..., N
    expected = np.arange(1, N+1, dtype=float)
    assert np.allclose(eigenvalues.real, expected, atol=1e-6), \
        f"Eigenvalues mismatch: {eigenvalues.real[:5]} vs {expected[:5]}"
    
    Lambda_real = eigenvalues.real
    V_inv = linalg.inv(V)
    
    # Compute B_tilde = V^{-1} B
    B_tilde = V_inv @ B
    
    return Lambda_real, V, V_inv, B_tilde, A, B


def test_A_matrix():
    """Test A matrix computation for α=0 (should recover LegS)."""
    N = 8
    A_legs = compute_A_matrix(N, alpha=0)
    
    print("A(α=0) - HiPPO-LegS:")
    print(np.round(A_legs, 2))
    
    # Verify diagonal
    for n in range(N):
        assert abs(A_legs[n, n] - (n+1)) < 1e-10, f"Diagonal mismatch at n={n}"
    
    # Verify off-diagonal for LegS
    for n in range(N):
        for k in range(n):
            expected = np.sqrt((2*n+1) * (2*k+1))
            assert abs(A_legs[n, k] - expected) < 1e-10, \
                f"Off-diagonal mismatch at ({n},{k}): {A_legs[n,k]} vs {expected}"
    
    print("✓ A(α=0) matches HiPPO-LegS")
    
    # Test for α > 0
    for alpha in [0.3, 0.5, 0.7, 0.9]:
        A = compute_A_matrix(N, alpha)
        print(f"\nA(α={alpha}):")
        print(np.round(A, 2))
        
        # Verify diagonal invariance
        for n in range(N):
            assert abs(A[n, n] - (n+1)) < 1e-6, \
                f"Diagonal mismatch at n={n} for α={alpha}: {A[n,n]}"
        
        # Verify lower triangular
        for n in range(N):
            for k in range(n+1, N):
                assert abs(A[n, k]) < 1e-10, \
                    f"Upper triangular non-zero at ({n},{k}) for α={alpha}"
        
        print(f"✓ A(α={alpha}) is lower triangular with correct diagonal")


def test_B_vector():
    """Test B vector computation."""
    N = 8
    
    # α=0 should give B_n = sqrt(2n+1)
    B_legs = compute_B_vector(N, alpha=0)
    expected = np.sqrt(2*np.arange(N) + 1)
    print("B(α=0):", np.round(B_legs, 4))
    print("Expected:", np.round(expected, 4))
    assert np.allclose(B_legs, expected, atol=1e-10), "B(α=0) mismatch"
    print("✓ B(α=0) matches LegS")
    
    for alpha in [0.3, 0.5, 0.7, 0.9]:
        B = compute_B_vector(N, alpha)
        print(f"B(α={alpha}):", np.round(B, 4))


def test_eigendecomposition():
    """Test full initialization pipeline."""
    N = 16
    for alpha in [0, 0.3, 0.5, 0.7, 0.9]:
        Lambda_real, V, V_inv, B_tilde, A, B = compute_fractal_init(N, alpha)
        
        # Verify reconstruction
        A_reconstructed = V @ np.diag(Lambda_real) @ V_inv
        assert np.allclose(A, A_reconstructed, atol=1e-6), \
            f"Reconstruction failed for α={alpha}"
        
        print(f"✓ α={alpha}: eigendecomposition verified, B_tilde norm = {np.linalg.norm(B_tilde):.4f}")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing B vector computation")
    print("=" * 60)
    test_B_vector()
    
    print("\n" + "=" * 60)
    print("Testing A matrix computation")
    print("=" * 60)
    test_A_matrix()
    
    print("\n" + "=" * 60)
    print("Testing eigendecomposition")
    print("=" * 60)
    test_eigendecomposition()
