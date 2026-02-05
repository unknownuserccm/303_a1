"""
VGG16 Pretrained Model with Transfer Learning
Uses PyTorch's pretrained VGG16 and fine-tunes it for fashion classification
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.metrics import average_precision_score
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt


class VGG16Pretrained(nn.Module):
    """
    VGG16 with pretrained weights from ImageNet
    
    Args:
        num_classes: Number of output classes
        freeze_features: If True, freeze convolutional layers
        dropout_rate: Dropout probability for classifier
        fine_tune_from: Layer index from which to start fine-tuning (if not freezing all)
    """
    
    def __init__(self, num_classes: int = 8,
                 freeze_features: bool = True,
                 dropout_rate: float = 0.5,
                 fine_tune_from: int = None):
        super(VGG16Pretrained, self).__init__()
        
        # Load pretrained VGG16
        self.vgg16 = models.vgg16(pretrained=True)
        
        # Freeze feature extraction layers if specified
        if freeze_features:
            for param in self.vgg16.features.parameters():
                param.requires_grad = False
        elif fine_tune_from is not None:
            # Freeze layers before fine_tune_from index
            for idx, child in enumerate(self.vgg16.features.children()):
                if idx < fine_tune_from:
                    for param in child.parameters():
                        param.requires_grad = False
        
        # Replace the classifier
        # Original: 25088 -> 4096 -> 4096 -> 1000
        # Modified: 25088 -> 4096 -> 4096 -> num_classes
        self.vgg16.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(4096, num_classes)
        )
        
        # Initialize new classifier weights
        self._initialize_classifier()
    
    def _initialize_classifier(self):
        """Initialize the new classifier weights"""
        for m in self.vgg16.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass through VGG16"""
        return self.vgg16(x)
    
    def get_num_params(self):
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def get_trainable_params(self):
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def unfreeze_all(self):
        """Unfreeze all layers for full fine-tuning"""
        for param in self.parameters():
            param.requires_grad = True
    
    def freeze_all_except_classifier(self):
        """Freeze all layers except the classifier"""
        for param in self.vgg16.features.parameters():
            param.requires_grad = False
        for param in self.vgg16.classifier.parameters():
            param.requires_grad = True
    
    def unfreeze_last_n_blocks(self, n: int):
        """
        Unfreeze the last n convolutional blocks
        
        Args:
            n: Number of blocks to unfreeze (1-5)
        """
        # VGG16 has 5 blocks, each ending with a MaxPool2d
        # We'll count backwards from the end
        block_indices = []
        pool_count = 0
        
        for idx, layer in enumerate(self.vgg16.features):
            if isinstance(layer, nn.MaxPool2d):
                pool_count += 1
                if pool_count > (5 - n):
                    block_indices.append(idx)
        
        # Unfreeze layers in the last n blocks
        for idx, child in enumerate(self.vgg16.features.children()):
            if idx >= min(block_indices):
                for param in child.parameters():
                    param.requires_grad = True


class VGG16PretrainedSmallClassifier(nn.Module):
    """
    VGG16 Pretrained with a smaller classifier head
    Reduces overfitting on smaller datasets
    """
    
    def __init__(self, num_classes: int = 8,
                 freeze_features: bool = True,
                 dropout_rate: float = 0.5):
        super(VGG16PretrainedSmallClassifier, self).__init__()
        
        # Load pretrained VGG16
        self.vgg16 = models.vgg16(pretrained=True)
        
        # Freeze features
        if freeze_features:
            for param in self.vgg16.features.parameters():
                param.requires_grad = False
        
        # Replace with smaller classifier
        self.vgg16.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, num_classes)
        )
        
        self._initialize_classifier()
    
    def _initialize_classifier(self):
        for m in self.vgg16.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.vgg16(x)


class VGG16PretrainedWithGlobalPool(nn.Module):
    """
    VGG16 Pretrained with Global Average Pooling instead of FC layers
    More parameter efficient
    """
    
    def __init__(self, num_classes: int = 8,
                 freeze_features: bool = True,
                 dropout_rate: float = 0.3):
        super(VGG16PretrainedWithGlobalPool, self).__init__()
        
        # Load pretrained VGG16
        vgg16 = models.vgg16(pretrained=True)
        
        # Use only features (conv layers)
        self.features = vgg16.features
        
        # Freeze features
        if freeze_features:
            for param in self.features.parameters():
                param.requires_grad = False
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Simple classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        
        self._initialize_classifier()
    
    def _initialize_classifier(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def create_vgg16_pretrained(model_type: str = 'standard',
                           num_classes: int = 8,
                           freeze_features: bool = True,
                           **kwargs):
    """
    Factory function to create different VGG16 pretrained variants
    
    Args:
        model_type: 'standard', 'small_classifier', or 'global_pool'
        num_classes: Number of output classes
        freeze_features: Whether to freeze feature extraction layers
        **kwargs: Additional arguments
    
    Returns:
        VGG16 pretrained model
    """
    
    if model_type == 'small_classifier':
        return VGG16PretrainedSmallClassifier(
            num_classes=num_classes,
            freeze_features=freeze_features,
            **kwargs
        )
    elif model_type == 'global_pool':
        return VGG16PretrainedWithGlobalPool(
            num_classes=num_classes,
            freeze_features=freeze_features,
            **kwargs
        )
    else:  # standard
        return VGG16Pretrained(
            num_classes=num_classes,
            freeze_features=freeze_features,
            **kwargs
        )


if __name__ == "__main__":
    # Test the models
    print("Testing VGG16 Pretrained Models...")
    
    # Test standard model
    print("\n1. Standard VGG16 Pretrained:")
    model = VGG16Pretrained(num_classes=8, freeze_features=True)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test small classifier variant
    print("\n2. VGG16 with Small Classifier:")
    model_small = VGG16PretrainedSmallClassifier(num_classes=8, freeze_features=True)
    trainable_params_small = sum(p.numel() for p in model_small.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params_small:,}")
    
    # Test global pooling variant
    print("\n3. VGG16 with Global Average Pooling:")
    model_gap = VGG16PretrainedWithGlobalPool(num_classes=8, freeze_features=True)
    trainable_params_gap = sum(p.numel() for p in model_gap.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params_gap:,}")
    
    # Test unfreezing
    print("\n4. Testing layer unfreezing:")
    model.unfreeze_last_n_blocks(2)
    trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters after unfreezing last 2 blocks: {trainable_after:,}")
