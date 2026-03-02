"""
Small GPT-Style Transformer with Configurable FFN (MLP or KAN)
===============================================================
A minimal causal language model for comparing MLP vs KAN as the
feed-forward network inside transformer layers.

Default config (~1-2M params):
  - 4 layers, 128 dim, 4 heads, ffn_dim=512, seq_len=128
  - Pre-norm (LayerNorm before attention & FFN)
  - Weight-tied embeddings (input embedding = output projection)
  - Sinusoidal positional encoding

Usage:
    model_mlp = SmallTransformer(vocab_size=8000, ffn_type='mlp')
    model_kan = SmallTransformer(vocab_size=8000, ffn_type='kan')
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from kan_layer import KANFFN


class SinusoidalPE(nn.Module):
    """Fixed sinusoidal positional encoding."""
    
    def __init__(self, dim, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)
    
    def forward(self, seq_len):
        return self.pe[:seq_len]


class MLPFFN(nn.Module):
    """Standard 2-layer MLP FFN with GELU activation."""
    
    def __init__(self, embedding_dim, ffn_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embedding_dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """Single transformer block: attention + FFN with pre-norm residuals."""
    
    def __init__(self, embedding_dim, num_heads, ffn_dim, dropout=0.1,
                 ffn_type='mlp', kan_grid_size=5, kan_spline_order=3):
        super().__init__()
        
        # Multi-head self-attention
        self.attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # FFN: MLP or KAN
        if ffn_type == 'mlp':
            self.ffn = MLPFFN(embedding_dim, ffn_dim, dropout=dropout)
        elif ffn_type == 'kan':
            self.ffn = KANFFN(embedding_dim, ffn_dim, dropout=dropout,
                              grid_size=kan_grid_size, spline_order=kan_spline_order)
        else:
            raise ValueError(f"Unknown ffn_type: {ffn_type}. Use 'mlp' or 'kan'.")
        
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, attn_mask=None):
        # Pre-norm attention + residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm,
                                attn_mask=attn_mask, is_causal=True)
        x = x + self.dropout(attn_out)
        
        # Pre-norm FFN + residual
        x = x + self.ffn(self.norm2(x))
        
        return x


class SmallTransformer(nn.Module):
    """Small causal transformer for language modeling.
    
    Args:
        vocab_size: Vocabulary size
        embedding_dim: Model dimension (default: 128)
        num_layers: Number of transformer blocks (default: 4)
        num_heads: Number of attention heads (default: 4)
        ffn_dim: FFN hidden dimension (default: 512)
        max_seq_len: Maximum sequence length (default: 128)
        dropout: Dropout rate (default: 0.1)
        ffn_type: 'mlp' for standard GELU FFN, 'kan' for KAN FFN
        kan_grid_size: B-spline grid intervals (KAN only, default: 5)
        kan_spline_order: B-spline order (KAN only, default: 3)
    """
    
    def __init__(self, vocab_size, embedding_dim=128, num_layers=4,
                 num_heads=4, ffn_dim=512, max_seq_len=128, dropout=0.1,
                 ffn_type='mlp', kan_grid_size=5, kan_spline_order=3):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.ffn_type = ffn_type
        
        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, embedding_dim)
        self.pos_enc = SinusoidalPE(embedding_dim, max_len=max_seq_len)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
                ffn_type=ffn_type,
                kan_grid_size=kan_grid_size,
                kan_spline_order=kan_spline_order,
            )
            for _ in range(num_layers)
        ])
        
        # Output
        self.norm = nn.LayerNorm(embedding_dim)
        self.head = nn.Linear(embedding_dim, vocab_size, bias=False)
        
        # Weight tying
        self.head.weight = self.token_emb.weight
        
        # Init
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def count_parameters(self):
        """Return total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, input_ids, labels=None):
        """
        Args:
            input_ids: (B, N) token indices
            labels: (B, N) target token indices (optional, for loss)
        Returns:
            logits: (B, N, vocab_size)
            loss: scalar (if labels provided)
        """
        B, N = input_ids.shape
        
        # Embeddings
        x = self.token_emb(input_ids) + self.pos_enc(N).unsqueeze(0)
        x = self.dropout(x)
        
        # Causal mask
        causal_mask = torch.triu(
            torch.full((N, N), float('-inf'), device=input_ids.device),
            diagonal=1
        )
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, attn_mask=causal_mask)
        
        # Output
        logits = self.head(self.norm(x))
        
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )
        
        return logits, loss


if __name__ == '__main__':
    print("=" * 60)
    print("SmallTransformer — Model Smoke Test")
    print("=" * 60)
    
    vocab = 256
    x = torch.randint(0, vocab, (2, 64))
    
    for ftype in ['mlp', 'kan']:
        m = SmallTransformer(vocab_size=vocab, ffn_type=ftype)
        logits, loss = m(x, labels=x)
        print(f"\n  [{ftype.upper()}]")
        print(f"    Params:  {m.count_parameters():,}")
        print(f"    Logits:  {logits.shape}")
        print(f"    Loss:    {loss.item():.3f}")
    
    print("\n✓ All models passed!")
