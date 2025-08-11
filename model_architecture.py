# CCSC_MTST/model_architecture.py
import torch
import torch.nn as nn

class Patching(nn.Module):
    def __init__(self, patch_length):
        super().__init__()
        self.patch_length = patch_length

    def forward(self, x):
        num_patches = x.shape[1] // self.patch_length
        return x.view(x.shape[0], num_patches, self.patch_length, x.shape[2])

class MTST(nn.Module):
    def __init__(self, num_features, time_steps, patch_length, d_model, num_heads, num_encoder_layers, dim_feedforward, dropout=0.1):
        super().__init__()
        assert time_steps % patch_length == 0, "Time steps must be divisible by patch length"
        self.patching = Patching(patch_length)
        num_patches = time_steps // patch_length
        
        # Correctly calculate input dimension for the linear projection
        patch_input_dim = patch_length * num_features
        self.linear_projection = nn.Linear(patch_input_dim, d_model)
        
        self.positional_encoding = nn.Parameter(torch.zeros(1, num_patches, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        self.output_aggregation_pool = nn.AdaptiveAvgPool1d(1)
        self.classification_output_layer = nn.Linear(d_model, 1)

    def forward(self, x):
        # x shape: (B, T, F) -> e.g., (batch_size, 24, 16)
        x = self.patching(x)
        # x shape: (B, N, P, F) -> e.g., (batch_size, 12, 2, 16)
        
        B, N, P, F = x.shape
        x = x.reshape(B, N, P * F)
        # x shape: (B, N, P * F) -> e.g., (batch_size, 12, 32)
        
        x = self.linear_projection(x)
        x = x + self.positional_encoding
        
        x = self.transformer_encoder(x)
        # x shape: (B, N, D) -> e.g., (batch_size, 12, 128)
        
        # Permute for pooling: (B, D, N)
        x = x.permute(0, 2, 1)
        x = self.output_aggregation_pool(x).squeeze(-1)
        # x shape: (B, D) -> e.g., (batch_size, 128)
        
        output_logits = self.classification_output_layer(x)
        return output_logits