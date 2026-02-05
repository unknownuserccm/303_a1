"""
Fashion Dataset Class - UPDATED FOR YOUR FOLDER STRUCTURE
Handles loading, preprocessing, and augmentation of fashion images

Your structure: ict303_a1/data/data/train, ict303_a1/data/data/valid, ict303_a1/data/data/test/unknown
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
from PIL import Image
from torch.utils.data import Dataset
from typing import Tuple, Optional


class FashionDataset(Dataset):
    """
    Custom Dataset for Fashion Classification
    
    Args:
        root_dir: Root directory (should be the ict303_a1 folder)
        split: 'train', 'valid', or 'test'
        transform: Optional transform to be applied on images
        img_size: Image size for resizing (default: 224 for VGG16)
    """
    
    def __init__(self, root_dir: str, split: str = 'train', 
                 transform: Optional[transforms.Compose] = None,
                 img_size: int = 224):
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        
        # Define class names based on your dataset structure
        self.classes = ['accessories', 'jackets', 'jeans', 'knitwear', 
                       'shirts', 'shoes', 'shorts', 'tees']
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.idx_to_class = {i: cls_name for cls_name, i in self.class_to_idx.items()}
        
        # Set up paths for YOUR structure: ict303_a1/data/data/train (or valid or test)
        if split == 'test':
            # For test: ict303_a1/data/data/test/unknown
            self.data_dir = os.path.join(root_dir, 'data', 'data', 'test', 'unknown')
        else:
            # For train/valid: ict303_a1/data/data/train or ict303_a1/data/data/valid
            self.data_dir = os.path.join(root_dir, 'data', 'data', split)
        
        print(f"Looking for {split} data in: {self.data_dir}")
        
        # Load image paths and labels
        self.samples = self._load_samples()
        
        # Set up transforms
        if transform is None:
            self.transform = self._get_default_transforms()
        else:
            self.transform = transform
    
    def _load_samples(self):
        """Load all image paths and their corresponding labels"""
        samples = []
        
        if self.split == 'test':
            # Test set has no labels (unknown folder)
            if os.path.exists(self.data_dir):
                for img_name in os.listdir(self.data_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(self.data_dir, img_name)
                        samples.append((img_path, -1))  # -1 for unknown label
            else:
                print(f"WARNING: Test directory not found: {self.data_dir}")
        else:
            # Train and valid sets have class folders directly in train/valid folder
            if os.path.exists(self.data_dir):
                print(f"Found {self.split} directory. Looking for class folders...")
                
                # List what's in the directory
                contents = os.listdir(self.data_dir)
                print(f"Contents: {contents}")
                
                for class_name in self.classes:
                    class_dir = os.path.join(self.data_dir, class_name)
                    if os.path.exists(class_dir):
                        img_count = 0
                        for img_name in os.listdir(class_dir):
                            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                                img_path = os.path.join(class_dir, img_name)
                                label = self.class_to_idx[class_name]
                                samples.append((img_path, label))
                                img_count += 1
                        print(f"  - {class_name}: {img_count} images")
                    else:
                        print(f"  - {class_name}: NOT FOUND")
            else:
                print(f"WARNING: {self.split} directory not found: {self.data_dir}")
        
        if len(samples) == 0:
            print(f"\n⚠️  WARNING: No samples found for {self.split} split!")
            print(f"Expected path: {self.data_dir}")
            print(f"\nPlease verify your folder structure:")
            print(f"  {self.root_dir}/data/data/{self.split}/")
            if self.split != 'test':
                print(f"    ├── accessories/")
                print(f"    ├── jackets/")
                print(f"    ├── jeans/")
                print(f"    └── ... (other class folders)")
        
        return samples
    
    def _get_default_transforms(self):
        """Get default transforms based on split"""
        if self.split == 'train':
            # Data augmentation for training
            return transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, 
                                     saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            # No augmentation for validation/test
            return transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image if loading fails
            image = Image.new('RGB', (self.img_size, self.img_size), color='black')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_dataloaders(root_dir: str, batch_size: int = 32, 
                   img_size: int = 224, num_workers: int = 4,
                   train_transform: Optional[transforms.Compose] = None,
                   val_transform: Optional[transforms.Compose] = None):
    """
    Create dataloaders for training, validation, and test sets
    
    Args:
        root_dir: Root directory (the ict303_a1 folder)
        batch_size: Batch size for dataloaders
        img_size: Image size for resizing
        num_workers: Number of workers for data loading
        train_transform: Optional custom transform for training
        val_transform: Optional custom transform for validation/test
    
    Returns:
        train_loader, val_loader, test_loader, dataset_info
    """
    
    print("="*70)
    print("LOADING DATASET")
    print("="*70)
    print(f"Root directory: {root_dir}")
    print()
    
    # Create datasets
    print("Loading TRAIN dataset...")
    train_dataset = FashionDataset(root_dir, split='train', 
                                  transform=train_transform, img_size=img_size)
    print()
    
    print("Loading VALID dataset...")
    val_dataset = FashionDataset(root_dir, split='valid', 
                                transform=val_transform, img_size=img_size)
    print()
    
    print("Loading TEST dataset...")
    test_dataset = FashionDataset(root_dir, split='test', 
                                 transform=val_transform, img_size=img_size)
    print("="*70)
    print()
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=num_workers, 
                             pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=num_workers, 
                           pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                            shuffle=False, num_workers=num_workers, 
                            pin_memory=True)
    
    # Dataset information
    dataset_info = {
        'num_classes': len(train_dataset.classes),
        'classes': train_dataset.classes,
        'class_to_idx': train_dataset.class_to_idx,
        'idx_to_class': train_dataset.idx_to_class,
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'test_size': len(test_dataset)
    }
    
    return train_loader, val_loader, test_loader, dataset_info


if __name__ == "__main__":
    # Test the dataset
    print("Testing Fashion Dataset...")
    
    # Update this path to your dataset location
    root_dir = r"C:\Github\303_a1\ict303_a1"  # ✅ Correct - raw string
    
    train_loader, val_loader, test_loader, info = get_dataloaders(
        root_dir, batch_size=8, img_size=224
    )
    
    print(f"\nDataset Information:")
    print(f"Number of classes: {info['num_classes']}")
    print(f"Classes: {info['classes']}")
    print(f"Training samples: {info['train_size']}")
    print(f"Validation samples: {info['val_size']}")
    print(f"Test samples: {info['test_size']}")
    
    # Test batch loading
    if info['train_size'] > 0:
        print("\nTesting batch loading...")
        images, labels = next(iter(train_loader))
        print(f"Batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
    else:
        print("\n⚠️  No training samples found - cannot test batch loading")