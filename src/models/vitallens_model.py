import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from typing import Dict, Tuple, Optional

class VitalLensModel(nn.Module):
    def __init__(self, num_frames=150, num_classes=1):
        """
        VitalLens model architecture
        
        Args:
            num_frames: Number of input frames (temporal dimension)
            num_classes: Output dimension (1 for BPM regression)
        """
        super(VitalLensModel, self).__init__()
        
        # EfficientNetV2-S backbone (pre-trained on ImageNet)
        self.backbone = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        
        # Remove the final classifier
        self.feature_extractor = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Get feature dimension from EfficientNetV2-S
        self.feature_dim = 1280
        
        # Temporal processing layers
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(self.feature_dim, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        
        # Global temporal pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Final regression head
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch, frames, channels, height, width)
        
        Returns:
            BPM predictions of shape (batch, 1)
        """
        batch_size, num_frames, channels, height, width = x.shape
        
        # Reshape for processing individual frames
        x = x.view(batch_size * num_frames, channels, height, width)
        
        # Extract features from each frame
        with torch.set_grad_enabled(self.training):
            features = self.feature_extractor(x)  # (batch*frames, feature_dim, 1, 1)
            features = features.squeeze(-1).squeeze(-1)  # (batch*frames, feature_dim)
        
        # Reshape back to temporal sequence
        features = features.view(batch_size, num_frames, self.feature_dim)
        features = features.transpose(1, 2)  # (batch, feature_dim, frames)
        
        # Temporal processing
        temporal_features = self.temporal_conv(features)  # (batch, 128, frames)
        
        # Global pooling across time
        pooled_features = self.global_pool(temporal_features).squeeze(-1)  # (batch, 128)
        
        # Final prediction
        bpm_pred = self.classifier(pooled_features)  # (batch, 1)
        
        return bpm_pred


class RPPGLoss(nn.Module):
    """Custom loss function for rPPG combining regression and signal quality"""
    
    def __init__(self, alpha=1.0, beta=0.1):
        super(RPPGLoss, self).__init__()
        self.alpha = alpha  # Weight for BPM regression loss
        self.beta = beta    # Weight for physiological constraint
        
    def forward(self, pred_bpm, target_bpm):
        # Primary regression loss (L1 for robustness)
        regression_loss = F.l1_loss(pred_bpm.squeeze(), target_bpm)
        
        # Physiological constraint: penalize unrealistic BPM values
        min_bpm, max_bpm = 40, 200
        constraint_loss = torch.mean(
            torch.clamp(min_bpm - pred_bpm.squeeze(), min=0) +
            torch.clamp(pred_bpm.squeeze() - max_bpm, min=0)
        )
        
        total_loss = self.alpha * regression_loss + self.beta * constraint_loss
        
        return total_loss, regression_loss, constraint_loss
