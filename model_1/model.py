import torch
import torch.nn as nn
import torch.nn.functional as F

from model_1.config import Config
from model_1.attention import TemporalAttention

class ConvBlock1D(nn.Module):
    """
    Modular 1D Convolutional block:
      Conv1D -> BatchNorm1d -> ReLU -> [Residual Connection] -> MaxPool1d
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, pool_size=2, use_residual=True):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.use_residual = use_residual
        
        # 1x1 Convolution projection to match channels for residual addition if they differ
        if use_residual and in_channels != out_channels:
            self.residual_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_proj = None
            
        if pool_size > 1:
            self.pool = nn.MaxPool1d(pool_size)
        else:
            self.pool = None

    def forward(self, x):
        # Input shape: (B, C_in, L)
        identity = x
        
        out = self.conv(x)
        out = self.bn(out)
        out = F.relu(out)
        
        # Apply residual addition before pooling
        if self.use_residual:
            if self.residual_proj is not None:
                identity = self.residual_proj(identity)
            out = out + identity
            
        if self.pool is not None:
            out = self.pool(out)
            
        # Output shape: (B, C_out, L_pooled)
        return out


class ExoplanetDeepModel(nn.Module):
    """
    Modern PyTorch Deep Learning Model for Exoplanet Transit Detection.
    Satisfies ISRO hackathon specifications:
      Stage 1: 1D CNN Feature Extractor
      Stage 2: Temporal Attention Module
      Stage 3: Bidirectional LSTM
      Stage 4: Shared Feature Embedding
      Multi-Task Heads: Classification, Regression, and Confidence
    """
    def __init__(self, config=Config):
        super().__init__()
        self.config = config
        
        # ------------------
        # Stage 1: Conv1D Extractor
        # ------------------
        cnn_layers = []
        in_c = config.INPUT_DIM
        
        # Build configurable CNN blocks
        for i, out_c in enumerate(config.CNN_CHANNELS):
            k_size = config.CNN_KERNEL_SIZES[i]
            # Pool only at the end of the CNN sequence
            pool = config.CNN_POOL_SIZE if i == len(config.CNN_CHANNELS) - 1 else 1
            
            cnn_layers.append(
                ConvBlock1D(
                    in_channels=in_c, 
                    out_channels=out_c, 
                    kernel_size=k_size, 
                    pool_size=pool,
                    use_residual=config.CNN_USE_RESIDUAL
                )
            )
            in_c = out_c
            
        self.cnn_extractor = nn.Sequential(*cnn_layers)
        
        # ------------------
        # Stage 2: Temporal Attention
        # ------------------
        # input dim for attention must match the final CNN channel size
        self.temporal_attention = TemporalAttention(
            embed_dim=config.CNN_CHANNELS[-1],
            num_heads=config.ATTENTION_HEADS,
            dropout=config.DROPOUT
        )
        
        # ------------------
        # Stage 3: Bi-directional LSTM
        # ------------------
        self.lstm = nn.LSTM(
            input_size=config.CNN_CHANNELS[-1],
            hidden_size=config.LSTM_HIDDEN_DIM,
            num_layers=config.LSTM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=config.DROPOUT if config.LSTM_LAYERS > 1 else 0.0
        )
        
        # ------------------
        # Stage 4: Shared Feature Embedding
        # ------------------
        # The BiLSTM output dim is 2 * hidden_dim
        lstm_out_dim = 2 * config.LSTM_HIDDEN_DIM
        self.shared_fc = nn.Linear(lstm_out_dim, config.SHARED_EMBEDDING_DIM)
        self.shared_bn = nn.BatchNorm1d(config.SHARED_EMBEDDING_DIM)
        self.shared_dropout = nn.Dropout(config.DROPOUT)
        
        # ------------------
        # Multi-Task Heads
        # ------------------
        # A. Classification Head (Planet, Binary, Variability, Noise, Blend)
        self.class_head = nn.Linear(config.SHARED_EMBEDDING_DIM, config.NUM_CLASSES)
        
        # B. Regression Head (depth, duration, period, midpoint)
        self.reg_head = nn.Linear(config.SHARED_EMBEDDING_DIM, 4)
        
        # C. Confidence Head
        self.conf_head = nn.Linear(config.SHARED_EMBEDDING_DIM, 1)

    def forward(self, x):
        """
        Forward Pass of the Network:
        Args:
            x: Input tensor of shape (B, INPUT_DIM, L)
               - Option A (Full): (B, 1, 2000) or (B, 2, 2000)
               - Option B (BLS Window): (B, 1, 200) or (B, 2, 200)
        Returns:
            class_logits: Shape (B, NUM_CLASSES)
            reg_outputs: Shape (B, 4) -> [scaled_depth, scaled_duration, scaled_period, scaled_midpoint]
            confidence: Shape (B, 1) -> transit confidence probability
            attn_weights: Shape (B, L_pooled, L_pooled) -> for mapping attention maps
        """
        # --- Stage 1: CNN Feature Extraction ---
        # Input tensor shape: (B, C_in, L)
        features = self.cnn_extractor(x)
        # Features shape: (B, CNN_CHANNELS[-1], L_pooled)
        # e.g., features shape: (B, 128, 500) if pooled once by 4
        
        # --- Stage 2: Temporal Attention ---
        # Permute to shape: (B, L_pooled, CNN_CHANNELS[-1]) for attention along time
        features_perm = features.permute(0, 2, 1)
        # features_perm shape: (B, 500, 128)
        
        attn_out, attn_weights = self.temporal_attention(features_perm)
        # attn_out shape: (B, 500, 128)
        # attn_weights shape: (B, 500, 500)
        
        # --- Stage 3: BiLSTM ---
        lstm_out, (hn, cn) = self.lstm(attn_out)
        # lstm_out shape: (B, L_pooled, 2 * LSTM_HIDDEN_DIM) -> (B, 500, 256)
        
        # Perform Global Average Pooling over the temporal dimension
        # This aggregates global information while preserving temporal alignment
        pooled = torch.mean(lstm_out, dim=1)
        # pooled shape: (B, 256)
        
        # --- Stage 4: Shared Feature Embedding ---
        shared = self.shared_fc(pooled)
        shared = self.shared_bn(shared)
        shared = F.relu(shared)
        shared = self.shared_dropout(shared)
        # shared shape: (B, 128)
        
        # --- Multi-Task Heads ---
        # A. Classification Logits
        class_logits = self.class_head(shared)  # (B, NUM_CLASSES)
        
        # B. Regression Parameters
        reg_outputs = self.reg_head(shared)     # (B, 4)
        
        # C. Confidence Prediction
        confidence_logits = self.conf_head(shared)
        confidence = torch.sigmoid(confidence_logits)  # (B, 1)
        
        return class_logits, reg_outputs, confidence, attn_weights
