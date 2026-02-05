"""
VGG16 Model Implementation from Scratch
Custom implementation of the VGG16 architecture for fashion classification
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


class VGG16Scratch(nn.Module):
    """
    VGG16 Architecture implemented from scratch
    
    Architecture:
    - Block 1: 2 x Conv(64) + MaxPool
    - Block 2: 2 x Conv(128) + MaxPool
    - Block 3: 3 x Conv(256) + MaxPool
    - Block 4: 3 x Conv(512) + MaxPool
    - Block 5: 3 x Conv(512) + MaxPool
    - FC Layers: 4096 -> 4096 -> num_classes
    
    Args:
        num_classes: Number of output classes
        dropout_rate: Dropout probability for FC layers
        use_batch_norm: Whether to use batch normalization
    """
    
    def __init__(self, num_classes: int = 8, 
                 dropout_rate: float = 0.5,
                 use_batch_norm: bool = False,
                 init_weights: bool = True):
        super(VGG16Scratch, self).__init__()
        
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        # Block 1: 2 conv layers with 64 filters
        self.block1 = self._make_conv_block(3, 64, num_layers=2)
        
        # Block 2: 2 conv layers with 128 filters
        self.block2 = self._make_conv_block(64, 128, num_layers=2)
        
        # Block 3: 3 conv layers with 256 filters
        self.block3 = self._make_conv_block(128, 256, num_layers=3)
        
        # Block 4: 3 conv layers with 512 filters
        self.block4 = self._make_conv_block(256, 512, num_layers=3)
        
        # Block 5: 3 conv layers with 512 filters
        self.block5 = self._make_conv_block(512, 512, num_layers=3)
        
        # Max pooling layer (used after each block)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Adaptive average pooling to handle different input sizes
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(4096, num_classes)
        )
        
        if init_weights:
            self._initialize_weights()
    
    def _make_conv_block(self, in_channels: int, out_channels: int, 
                         num_layers: int):
        """
        Create a convolutional block with multiple conv layers
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            num_layers: Number of conv layers in the block
        
        Returns:
            Sequential block of conv layers
        """
        layers = []
        
        for i in range(num_layers):
            # First layer uses in_channels, rest use out_channels
            conv_in = in_channels if i == 0 else out_channels
            
            # Convolutional layer
            layers.append(
                nn.Conv2d(conv_in, out_channels, 
                         kernel_size=3, padding=1)
            )
            
            # Batch normalization (optional)
            if self.use_batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            
            # ReLU activation
            layers.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', 
                                       nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
        
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Convolutional blocks with pooling
        x = self.block1(x)
        x = self.maxpool(x)
        
        x = self.block2(x)
        x = self.maxpool(x)
        
        x = self.block3(x)
        x = self.maxpool(x)
        
        x = self.block4(x)
        x = self.maxpool(x)
        
        x = self.block5(x)
        x = self.maxpool(x)
        
        # Adaptive pooling
        x = self.avgpool(x)
        
        # Flatten
        x = torch.flatten(x, 1)
        
        # Fully connected layers
        x = self.classifier(x)
        
        return x
    
    def get_num_params(self):
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def get_trainable_params(self):
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_feature_maps(self, x):
        """
        Get feature maps from each block (for visualization)
        
        Args:
            x: Input tensor
        
        Returns:
            Dictionary of feature maps from each block
        """
        features = {}
        
        x = self.block1(x)
        features['block1'] = x
        x = self.maxpool(x)
        
        x = self.block2(x)
        features['block2'] = x
        x = self.maxpool(x)
        
        x = self.block3(x)
        features['block3'] = x
        x = self.maxpool(x)
        
        x = self.block4(x)
        features['block4'] = x
        x = self.maxpool(x)
        
        x = self.block5(x)
        features['block5'] = x
        
        return features


class VGG16ScratchSmall(nn.Module):
    """
    Smaller variant of VGG16 with reduced FC layer sizes
    Useful for faster training and less overfitting
    """
    
    def __init__(self, num_classes: int = 8, dropout_rate: float = 0.5):
        super(VGG16ScratchSmall, self).__init__()
        
        # Same convolutional blocks as VGG16
        self.block1 = self._make_conv_block(3, 64, 2)
        self.block2 = self._make_conv_block(64, 128, 2)
        self.block3 = self._make_conv_block(128, 256, 3)
        self.block4 = self._make_conv_block(256, 512, 3)
        self.block5 = self._make_conv_block(512, 512, 3)
        
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Smaller FC layers
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(1024, num_classes)
        )
        
        self._initialize_weights()
    
    def _make_conv_block(self, in_channels, out_channels, num_layers):
        layers = []
        for i in range(num_layers):
            conv_in = in_channels if i == 0 else out_channels
            layers.extend([
                nn.Conv2d(conv_in, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ])
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.maxpool(self.block1(x))
        x = self.maxpool(self.block2(x))
        x = self.maxpool(self.block3(x))
        x = self.maxpool(self.block4(x))
        x = self.maxpool(self.block5(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    # Test the model
    print("Testing VGG16 Scratch Implementation...")
    
    # Create model
    model = VGG16Scratch(num_classes=8, dropout_rate=0.5)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test feature extraction
    features = model.get_feature_maps(dummy_input)
    print("\nFeature map shapes:")
    for block_name, feature_map in features.items():
        print(f"{block_name}: {feature_map.shape}")
    
    # Test small variant
    print("\n\nTesting VGG16 Small Variant...")
    model_small = VGG16ScratchSmall(num_classes=8)
    total_params_small = sum(p.numel() for p in model_small.parameters())
    print(f"Total parameters (small): {total_params_small:,}")
    output_small = model_small(dummy_input)
    print(f"Output shape (small): {output_small.shape}")
