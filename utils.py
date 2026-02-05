"""
Utility Functions
Helper functions for visualization, metrics computation, and analysis
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
from sklearn.metrics import classification_report
from typing import List, Tuple
import seaborn as sns


def denormalize_image(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Denormalize image tensor for visualization
    
    Args:
        img_tensor: Normalized image tensor
        mean: Mean used for normalization
        std: Std used for normalization
    
    Returns:
        Denormalized image tensor
    """
    img = img_tensor.clone()
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(img, 0, 1)


def visualize_predictions(model, data_loader, device, class_names, 
                         num_images=16, save_path=None):
    """
    Visualize model predictions on a batch of images
    
    Args:
        model: Trained model
        data_loader: Data loader
        device: Device to run inference on
        class_names: List of class names
        num_images: Number of images to visualize
        save_path: Path to save the visualization
    """
    model.eval()
    
    # Get a batch of images
    images, labels = next(iter(data_loader))
    images = images[:num_images]
    labels = labels[:num_images]
    
    # Get predictions
    with torch.no_grad():
        outputs = model(images.to(device))
        _, predicted = outputs.max(1)
        probs = torch.softmax(outputs, dim=1)
    
    # Plot
    n_cols = 4
    n_rows = (num_images + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten() if num_images > 1 else [axes]
    
    for idx in range(num_images):
        # Denormalize image
        img = denormalize_image(images[idx])
        img = img.permute(1, 2, 0).cpu().numpy()
        
        # Get prediction info
        true_label = class_names[labels[idx]]
        pred_label = class_names[predicted[idx]]
        confidence = probs[idx][predicted[idx]].item() * 100
        
        # Plot
        axes[idx].imshow(img)
        axes[idx].axis('off')
        
        # Color code: green if correct, red if wrong
        color = 'green' if labels[idx] == predicted[idx] else 'red'
        title = f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.1f}%'
        axes[idx].set_title(title, color=color, fontsize=10)
    
    # Hide extra subplots
    for idx in range(num_images, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_class_distribution(data_loader, class_names, save_path=None):
    """
    Plot the distribution of classes in the dataset
    
    Args:
        data_loader: Data loader
        class_names: List of class names
        save_path: Path to save the plot
    """
    all_labels = []
    
    for _, labels in data_loader:
        all_labels.extend(labels.numpy())
    
    all_labels = np.array(all_labels)
    
    # Count classes
    unique, counts = np.unique(all_labels, return_counts=True)
    
    # Plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(unique)), counts)
    plt.xticks(range(len(unique)), [class_names[i] for i in unique], rotation=45, ha='right')
    plt.ylabel('Number of Samples')
    plt.title('Class Distribution')
    plt.grid(axis='y', alpha=0.3)
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    plt.close()


def compute_per_class_accuracy(predictions, labels, class_names):
    """
    Compute accuracy for each class
    
    Args:
        predictions: Model predictions
        labels: True labels
        class_names: List of class names
    
    Returns:
        Dictionary of per-class accuracies
    """
    per_class_acc = {}
    
    for idx, class_name in enumerate(class_names):
        class_mask = labels == idx
        class_preds = predictions[class_mask]
        class_labels = labels[class_mask]
        
        if len(class_labels) > 0:
            accuracy = (class_preds == class_labels).sum() / len(class_labels) * 100
            per_class_acc[class_name] = accuracy
        else:
            per_class_acc[class_name] = 0.0
    
    return per_class_acc


def plot_per_class_accuracy(per_class_acc, save_path=None):
    """
    Plot per-class accuracy
    
    Args:
        per_class_acc: Dictionary of per-class accuracies
        save_path: Path to save the plot
    """
    classes = list(per_class_acc.keys())
    accuracies = list(per_class_acc.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(classes)), accuracies)
    plt.xticks(range(len(classes)), classes, rotation=45, ha='right')
    plt.ylabel('Accuracy (%)')
    plt.title('Per-Class Accuracy')
    plt.ylim(0, 105)
    plt.grid(axis='y', alpha=0.3)
    
    # Add accuracy labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1f}%',
                ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    plt.close()


def compare_models_performance(results_dict, metric='accuracy', save_path=None):
    """
    Compare performance of multiple models
    
    Args:
        results_dict: Dictionary with model names as keys and metrics as values
        metric: Metric to compare ('accuracy', 'f1_score', 'precision', 'recall')
        save_path: Path to save the plot
    """
    models = list(results_dict.keys())
    values = [results_dict[model][metric] for model in models]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(models)), values)
    plt.xticks(range(len(models)), models, rotation=45, ha='right')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.title(f'Model Comparison - {metric.replace("_", " ").title()}')
    plt.ylim(0, 105 if metric == 'accuracy' else 1.1)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        if metric == 'accuracy':
            label = f'{val:.2f}%'
        else:
            label = f'{val:.3f}'
        plt.text(bar.get_x() + bar.get_width()/2., height,
                label, ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    plt.close()


def visualize_feature_maps(model, image, layer_name, device, save_path=None):
    """
    Visualize feature maps from a specific layer
    
    Args:
        model: Model with feature extraction capability
        image: Input image tensor
        layer_name: Name of layer to visualize
        device: Device to run inference on
        save_path: Path to save visualization
    """
    model.eval()
    
    # Get feature maps
    if hasattr(model, 'get_feature_maps'):
        with torch.no_grad():
            features = model.get_feature_maps(image.unsqueeze(0).to(device))
        
        if layer_name in features:
            feature_map = features[layer_name][0]  # Get first image in batch
            
            # Plot first 16 channels
            n_features = min(16, feature_map.size(0))
            fig, axes = plt.subplots(4, 4, figsize=(12, 12))
            axes = axes.flatten()
            
            for idx in range(n_features):
                feat = feature_map[idx].cpu().numpy()
                axes[idx].imshow(feat, cmap='viridis')
                axes[idx].axis('off')
                axes[idx].set_title(f'Channel {idx}')
            
            # Hide extra subplots
            for idx in range(n_features, 16):
                axes[idx].axis('off')
            
            plt.suptitle(f'Feature Maps - {layer_name}')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
            plt.close()


def print_classification_report(predictions, labels, class_names):
    """
    Print detailed classification report
    
    Args:
        predictions: Model predictions
        labels: True labels
        class_names: List of class names
    """
    report = classification_report(labels, predictions, 
                                   target_names=class_names,
                                   digits=3)
    print("\nClassification Report:")
    print("=" * 70)
    print(report)


def save_predictions_to_file(predictions, image_paths, class_names, save_path):
    """
    Save predictions to a CSV file
    
    Args:
        predictions: Model predictions
        image_paths: List of image file paths
        class_names: List of class names
        save_path: Path to save CSV file
    """
    import csv
    
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Image', 'Predicted_Class', 'Class_Index'])
        
        for img_path, pred in zip(image_paths, predictions):
            writer.writerow([img_path, class_names[pred], pred])
    
    print(f"Predictions saved to {save_path}")


if __name__ == "__main__":
    print("Utility functions module loaded successfully")
