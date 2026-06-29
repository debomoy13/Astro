import torch
import torch.nn as nn
import math

class TemporalAttention(nn.Module):
    """
    Temporal Multi-Head Self-Attention Module.
    
    This layer learns which timestamps in the preprocessed light curve are most
    important (e.g. transit ingress, bottom, egress) and suppresses flat regions.
    
    Mathematical Formulation:
      Given input sequence H of shape (B, T, D) where B is batch size, T is
      temporal sequence length, and D is feature embedding dimension:
      
      1. Project input H to Query (Q), Key (K), and Value (V) spaces:
         Q = H * W_q + b_q   (W_q in R^{D x D_k})
         K = H * W_k + b_k   (W_k in R^{D x D_k})
         V = H * W_v + b_v   (W_v in R^{D x D})
         
      2. Compute raw attention scores (scaled dot-product):
         S = (Q * K^T) / sqrt(D_k)
         
      3. Compute normalized attention weight distribution A via Softmax along columns:
         A_{i, j} = exp(S_{i, j}) / sum_k(exp(S_{i, k}))
         
      4. Apply attention weights to compute weighted context representations:
         O = A * V
         
      5. Output is passed through a Dropout layer, a residual connection, and LayerNorm:
         Output = LayerNorm(H + Dropout(O))
    """
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Multi-Head Attention layer in PyTorch
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, T, D)
        Returns:
            output: Attention contextualized tensor of shape (B, T, D)
            attn_weights: Attention weight matrix of shape (B, T, T)
        """
        # x shape: (B, T, D)
        attn_output, attn_weights = self.mha(x, x, x, need_weights=True)
        
        # Residual connection + Layer Normalization
        output = self.layer_norm(x + attn_output)
        
        return output, attn_weights
