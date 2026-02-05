"""
Trainer Class
Handles training, validation, and evaluation of models with TensorBoard logging
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
import time
from tqdm import tqdm
from typing import Dict, Tuple, Optional
from sklearn.metrics import classification_report, precision_recall_fscore_support
import seaborn as sns


class Trainer:
    """
    Trainer class for model training and evaluation
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on (cuda/cpu)
        scheduler: Learning rate scheduler (optional)
        log_dir: Directory for TensorBoard logs
        class_names: List of class names for visualization
    """
    
    def __init__(self, model: nn.Module,
                 train_loader,
                 val_loader,
                 criterion: nn.Module,
                 optimizer: optim.Optimizer,
                 device: torch.device,
                 scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
                 log_dir: str = './runs',
                 class_names: list = None):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.class_names = class_names
        
        # Create TensorBoard writer
        self.writer = SummaryWriter(log_dir)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_model_path = None
    
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """
        Train for one epoch
        
        Returns:
            Average loss and accuracy for the epoch
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, epoch: int) -> Tuple[float, float]:
        """
        Validate for one epoch
        
        Returns:
            Average loss and accuracy for the epoch
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]')
            
            for batch_idx, (images, labels) in enumerate(pbar):
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': running_loss / (batch_idx + 1),
                    'acc': 100. * correct / total
                })
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, num_epochs: int, save_dir: str = './checkpoints',
              early_stopping_patience: int = None):
        """
        Train the model for multiple epochs
        
        Args:
            num_epochs: Number of epochs to train
            save_dir: Directory to save model checkpoints
            early_stopping_patience: Number of epochs to wait before early stopping
        """
        os.makedirs(save_dir, exist_ok=True)
        
        best_val_acc = 0.0
        patience_counter = 0
        
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(1, num_epochs + 1):
            start_time = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.validate_epoch(epoch)
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
            else:
                current_lr = self.optimizer.param_groups[0]['lr']
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(current_lr)
            
            # Log to TensorBoard
            self.writer.add_scalars('Loss', {
                'train': train_loss,
                'val': val_loss
            }, epoch)
            
            self.writer.add_scalars('Accuracy', {
                'train': train_acc,
                'val': val_acc
            }, epoch)
            
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            # Print epoch summary
            epoch_time = time.time() - start_time
            print(f'\nEpoch {epoch}/{num_epochs} - {epoch_time:.2f}s')
            print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
            print(f'Learning Rate: {current_lr:.6f}')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.best_val_acc = val_acc
                checkpoint_path = os.path.join(save_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss
                }, checkpoint_path)
                self.best_model_path = checkpoint_path
                print(f'✓ Best model saved with val_acc: {val_acc:.2f}%')
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if early_stopping_patience and patience_counter >= early_stopping_patience:
                print(f'\nEarly stopping triggered after {epoch} epochs')
                break
            
            # Save checkpoint every 10 epochs
            if epoch % 10 == 0:
                checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, checkpoint_path)
        
        print(f'\nTraining completed!')
        print(f'Best validation accuracy: {best_val_acc:.2f}%')
        
        self.writer.close()
    
    def evaluate(self, data_loader, return_predictions: bool = False):
        """
        Evaluate the model and compute metrics
        
        Args:
            data_loader: Data loader for evaluation
            return_predictions: Whether to return predictions and labels
        
        Returns:
            Dictionary of metrics (and optionally predictions and labels)
        """
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        running_loss = 0.0
        
        with torch.no_grad():
            for images, labels in tqdm(data_loader, desc='Evaluating'):
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                
                # Get probabilities for MAP calculation
                probs = torch.softmax(outputs, dim=1)
                
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Compute metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        accuracy = 100. * (all_preds == all_labels).sum() / len(all_labels)
        avg_loss = running_loss / len(data_loader)
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # Classification report
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted'
        )
        
        # Calculate Mean Average Precision (MAP)
        try:
            # Binarize the labels for multi-class AP calculation
            num_classes = len(self.class_names) if self.class_names else all_probs.shape[1]
            y_true_bin = label_binarize(all_labels, classes=range(num_classes))
            
            # Calculate average precision for each class
            map_score = average_precision_score(y_true_bin, all_probs, average='weighted')
        except Exception as e:
            print(f"Warning: Could not calculate MAP: {e}")
            map_score = 0.0
        
        metrics = {
            'accuracy': accuracy,
            'loss': avg_loss,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'map': map_score,  # Mean Average Precision
            'confusion_matrix': cm
        }
        
        if return_predictions:
            metrics['predictions'] = all_preds
            metrics['labels'] = all_labels
            metrics['probabilities'] = all_probs
        
        return metrics
    
    def plot_confusion_matrix(self, cm, save_path: str = None):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_training_curves(self, save_path: str = None):
        """Plot training and validation curves"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curves
        axes[0].plot(self.history['train_loss'], label='Train Loss')
        axes[0].plot(self.history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy curves
        axes[1].plot(self.history['train_acc'], label='Train Acc')
        axes[1].plot(self.history['val_acc'], label='Val Acc')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {checkpoint_path}")
        if 'val_acc' in checkpoint:
            print(f"Checkpoint validation accuracy: {checkpoint['val_acc']:.2f}%")
