"""
KAN (Kolmogorov-Arnold Network) Layer Implementation
=====================================================
Drop-in replacement for standard MLP/FFN in transformers.

KANs place learnable activation functions (B-splines) on edges
instead of fixed activations on nodes. Based on:
  - KAN paper: https://arxiv.org/abs/2404.19756
  - Efficient-KAN: https://github.com/Blealtan/efficient-kan

Architecture per KANLinear:
  output = base_activation(x) @ W_base + B_spline(x) @ W_spline

The base path (SiLU + linear) provides a stable residual,
while the spline path learns arbitrary univariate functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class KANLinear(nn.Module):
    """Single KAN linear layer with B-spline activations on edges.
    
    For each (input, output) pair, learns a univariate function
    parameterized as a B-spline. Total spline params per layer:
    out_features × in_features × (grid_size + spline_order).
    
    Args:
        in_features: Input dimension
        out_features: Output dimension  
        grid_size: Number of B-spline grid intervals (more = more expressive)
        spline_order: B-spline polynomial order (3 = cubic, smooth)
        grid_range: Range of the spline grid knots
    """
    
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3,
                 grid_range=(-1.0, 1.0)):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        # Base linear path (residual): SiLU(x) @ W_base
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        
        # B-spline grid: uniform knots extended by spline_order on each side
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = torch.linspace(
            grid_range[0] - h * spline_order,
            grid_range[1] + h * spline_order,
            grid_size + 2 * spline_order + 1
        )
        self.register_buffer('grid', grid)
        
        # Spline coefficients
        n_basis = grid_size + spline_order
        self.spline_weight = nn.Parameter(
            torch.randn(out_features, in_features, n_basis) * 
            (1.0 / (in_features * n_basis))
        )
    
    def _b_spline_basis(self, x):
        """Compute B-spline basis values using Cox-de Boor recursion.
        
        Args:
            x: (..., in_features) input tensor
        Returns:
            (..., in_features, n_basis) basis function values
        """
        x = x.unsqueeze(-1)  # (..., in_features, 1)
        grid = self.grid      # (n_knots,)
        
        # Order 0: piecewise constant
        bases = ((x >= grid[:-1]) & (x < grid[1:])).float()
        
        # Recursion up to spline_order
        for k in range(1, self.spline_order + 1):
            left_num = x - grid[:-(k + 1)]
            left_den = grid[k:-1] - grid[:-(k + 1)]
            right_num = grid[k + 1:] - x
            right_den = grid[k + 1:] - grid[1:-k]
            
            left = left_num / (left_den + 1e-8) * bases[..., :-1]
            right = right_num / (right_den + 1e-8) * bases[..., 1:]
            bases = left + right
        
        return bases  # (..., in_features, n_basis)
    
    def forward(self, x):
        """
        Args:
            x: (..., in_features)
        Returns:
            (..., out_features)
        """
        # Base path: silu activation + linear
        base_out = F.linear(F.silu(x), self.base_weight)
        
        # Spline path: learned activation functions
        splines = self._b_spline_basis(x)  # (..., in, n_basis)
        spline_out = torch.einsum('oin,...in->...o', self.spline_weight, splines)
        
        return base_out + spline_out


class KANFFN(nn.Module):
    """KAN-based Feed-Forward Network — drop-in replacement for MLP FFN.
    
    Standard FFN:  Linear(d, 4d) → GELU → Linear(4d, d)
    KAN FFN:       KANLinear(d, 4d) → KANLinear(4d, d)
    
    The KAN layers already contain their own nonlinearities (B-splines),
    so no explicit activation function is needed between them.
    """
    
    def __init__(self, embedding_dim, ffn_dim, dropout=0.1,
                 grid_size=5, spline_order=3):
        super().__init__()
        self.kan1 = KANLinear(embedding_dim, ffn_dim,
                              grid_size=grid_size, spline_order=spline_order)
        self.dropout1 = nn.Dropout(dropout)
        self.kan2 = KANLinear(ffn_dim, embedding_dim,
                              grid_size=grid_size, spline_order=spline_order)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.kan1(x)
        x = self.dropout1(x)
        x = self.kan2(x)
        x = self.dropout2(x)
        return x


if __name__ == '__main__':
    # Quick test
    print("Testing KAN layer...")
    layer = KANLinear(128, 512, grid_size=5, spline_order=3)
    x = torch.randn(2, 64, 128)
    y = layer(x)
    print(f"  Input:  {x.shape}")
    print(f"  Output: {y.shape}")
    print(f"  Params: {sum(p.numel() for p in layer.parameters()):,}")
    
    print("\nTesting KAN FFN...")
    ffn = KANFFN(128, 512, dropout=0.1)
    y = ffn(x)
    print(f"  Input:  {x.shape}")
    print(f"  Output: {y.shape}")
    print(f"  Params: {sum(p.numel() for p in ffn.parameters()):,}")
    print("PASS ✓")
