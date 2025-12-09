from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class MorphologicalRefiner(nn.Module):
    """
    CNN-based model to refine ECG morphological features.
    Takes the autoencoder-reconstructed signal and refines it to better match
    the ground truth morphological features (peaks, valleys, slopes).
    """
    
    def __init__(
        self,
        input_length: int = 640,  # 5 seconds at 128Hz
        num_filters: int = 64,
        kernel_sizes: Tuple[int, ...] = (3, 5, 7, 9),
        dropout: float = 0.2,
    ):
        super(MorphologicalRefiner, self).__init__()
        self.input_length = input_length
        
        # Multi-scale convolutional layers to capture different morphological features
        self.conv_layers = nn.ModuleList()
        for kernel_size in kernel_sizes:
            conv_block = nn.Sequential(
                nn.Conv1d(1, num_filters, kernel_size=kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Conv1d(num_filters, num_filters, kernel_size=kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(inplace=True),
            )
            self.conv_layers.append(conv_block)
        
        # Combine multi-scale features
        total_filters = num_filters * len(kernel_sizes)
        self.fusion = nn.Sequential(
            nn.Conv1d(total_filters, num_filters * 2, kernel_size=1),
            nn.BatchNorm1d(num_filters * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        
        # Refinement layers
        self.refine_layers = nn.Sequential(
            nn.Conv1d(num_filters * 2, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(num_filters, num_filters // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_filters // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_filters // 2, 1, kernel_size=1),
        )
        
        # Residual connection
        self.residual_scale = nn.Parameter(torch.ones(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input signal (batch_size, 1, length) or (batch_size, length)
        
        Returns:
            Refined signal (batch_size, 1, length) or (batch_size, length)
        """
        # Ensure input is (batch_size, 1, length)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        batch_size, channels, length = x.shape
        
        # Multi-scale feature extraction
        multi_scale_features = []
        for conv_block in self.conv_layers:
            features = conv_block(x)
            multi_scale_features.append(features)
        
        # Concatenate multi-scale features
        fused = torch.cat(multi_scale_features, dim=1)
        
        # Fusion and refinement
        fused = self.fusion(fused)
        refined = self.refine_layers(fused)
        
        # Residual connection with learnable scaling
        output = x + self.residual_scale * refined
        
        # Return in same format as input
        if output.shape[1] == 1:
            output = output.squeeze(1)
        
        return output

