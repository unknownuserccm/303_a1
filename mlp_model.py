"""
Multilayer Perceptron (MLP) Model
A fully connected neural network for fashion classification
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
import torch.nn.functional as F


class MLPModel(nn.Module):
    """
    Multilayer Perceptron for Fashion Classification
    
    Args:
        input_size: Size of flattened input (e.g., 224*224*3 for RGB images)
        hidden_sizes: List of hidden layer sizes
        num_classes: Number of output classes
        dropout_rate: Dropout probability for regularization
        use_batch_norm: Whether to use batch normalization
    """
    
    def __init__(self, input_size: int = 224*224*3, 
                 hidden_sizes: list = [2048, 1024, 512, 256],
                 num_classes: int = 8,
                 dropout_rate: float = 0.5,
                 use_batch_norm: bool = True):
        super(MLPModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        # Build the network
        layers = []
        in_features = input_size
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            # Linear layer
            layers.append(nn.Linear(in_features, hidden_size))
            
            # Batch normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            
            # Activation
            layers.append(nn.ReLU(inplace=True))
            
            # Dropout
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            in_features = hidden_size
        
        # Output layer
        layers.append(nn.Linear(in_features, num_classes))
        
        # Combine all layers
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
        
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Flatten the input
        x = x.view(x.size(0), -1)
        
        # Pass through network
        x = self.network(x)
        
        return x
    
    def get_num_params(self):
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def get_trainable_params(self):
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MLPModelSmall(nn.Module):
    """
    Smaller MLP variant for faster experimentation
    """
    
    def __init__(self, input_size: int = 224*224*3,
                 hidden_sizes: list = [1024, 512, 256],
                 num_classes: int = 8,
                 dropout_rate: float = 0.3):
        super(MLPModelSmall, self).__init__()
        
        self.flatten = nn.Flatten()
        
        layers = []
        in_features = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(in_features, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            ])
            in_features = hidden_size
        
        layers.append(nn.Linear(in_features, num_classes))
        
        self.network = nn.Sequential(*layers)
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)


class MLPModelLarge(nn.Module):
    """
    Larger MLP variant with more capacity
    """
    
    def __init__(self, input_size: int = 224*224*3,
                 hidden_sizes: list = [4096, 2048, 1024, 512, 256],
                 num_classes: int = 8,
                 dropout_rate: float = 0.5,
                 use_batch_norm: bool = True):
        super(MLPModelLarge, self).__init__()
        
        self.flatten = nn.Flatten()
        
        layers = []
        in_features = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_features, hidden_size))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            
            layers.extend([
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            ])
            
            in_features = hidden_size
        
        layers.append(nn.Linear(in_features, num_classes))
        
        self.network = nn.Sequential(*layers)
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)


def create_mlp_model(model_size: str = 'medium', input_size: int = 224*224*3,
                    num_classes: int = 8, **kwargs):
    """
    Factory function to create MLP models of different sizes
    
    Args:
        model_size: 'small', 'medium', or 'large'
        input_size: Flattened input size
        num_classes: Number of output classes
        **kwargs: Additional arguments for model configuration
    
    Returns:
        MLP model instance
    """
    
    if model_size == 'small':
        return MLPModelSmall(input_size=input_size, num_classes=num_classes, **kwargs)
    elif model_size == 'large':
        return MLPModelLarge(input_size=input_size, num_classes=num_classes, **kwargs)
    else:  # medium (default)
        return MLPModel(input_size=input_size, num_classes=num_classes, **kwargs)


if __name__ == "__main__":
    # Test the model
    print("Testing MLP Models...")
    
    # Test different model sizes
    for size in ['small', 'medium', 'large']:
        print(f"\n{size.upper()} Model:")
        model = create_mlp_model(model_size=size, input_size=224*224*3, num_classes=8)
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
        
        # Test forward pass
        dummy_input = torch.randn(4, 3, 224, 224)
        output = model(dummy_input)
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
        
        # Print model architecture
        if size == 'medium':
            print(f"\nModel Architecture:\n{model}")
