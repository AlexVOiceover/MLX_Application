#!/usr/bin/env python3
# train.py

"""
Training script for MNIST Digit Classifier.

This module provides functionality for loading the MNIST dataset,
defining a CNN model architecture, and training the model.
"""

# Import necessary libraries
import logging
import os
from pathlib import Path
from typing import Tuple, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_mnist_data(
    batch_size: int = 64, data_dir: str = "./data"
) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    """Load MNIST training and test datasets.
    
    This function downloads the MNIST dataset if not already available,
    applies appropriate transformations, and creates DataLoader objects
    for both training and test sets.
    
    Args:
        batch_size: The batch size for the DataLoader. Defaults to 64.
        data_dir: Directory where the MNIST data is stored. Defaults to "./data".
        
    Returns:
        A tuple containing:
            - training DataLoader
            - test DataLoader
            - dataset_info dictionary with details about the datasets
            
    Raises:
        RuntimeError: If there's an issue downloading or loading the MNIST data.
    """
    try:
        # Create data directory if it doesn't exist
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        
        # Define transformations
        # The MNIST dataset pixel values are 0-255, we normalize to 0-1
        # Mean and std are calculated from the MNIST training set: mean=0.1307, std=0.3081
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Download and load training data
        logger.info("Loading MNIST training dataset...")
        train_dataset = datasets.MNIST(
            root=data_dir,
            train=True,
            download=True,
            transform=transform
        )
        
        # Download and load test data
        logger.info("Loading MNIST test dataset...")
        test_dataset = datasets.MNIST(
            root=data_dir,
            train=False,
            download=True,
            transform=transform
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=os.cpu_count() or 1,
            pin_memory=torch.cuda.is_available()
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=os.cpu_count() or 1,
            pin_memory=torch.cuda.is_available()
        )
        
        # Create a dictionary with dataset information
        dataset_info = {
            "train_size": len(train_dataset),
            "test_size": len(test_dataset),
            "image_shape": train_dataset[0][0].shape,
            "classes": len(train_dataset.classes),
            "class_labels": train_dataset.classes if hasattr(train_dataset, "classes") else list(range(10)),
            "batch_size": batch_size,
            "num_batches_train": len(train_loader),
            "num_batches_test": len(test_loader)
        }
        
        logger.info(f"MNIST dataset loaded with {dataset_info['train_size']} training and {dataset_info['test_size']} test samples")
        
        return train_loader, test_loader, dataset_info
        
    except Exception as e:
        logger.error(f"Error loading MNIST dataset: {str(e)}")
        raise RuntimeError(f"Failed to load MNIST dataset: {str(e)}")


class MNISTClassifier(nn.Module):
    """CNN model for MNIST digit classification.
    
    Architecture:
        - Input: 1x28x28 grayscale images
        - Conv Layer 1: 32 filters of size 3x3, ReLU activation
        - Max Pooling: 2x2 with stride 2
        - Conv Layer 2: 64 filters of size 3x3, ReLU activation
        - Max Pooling: 2x2 with stride 2
        - Fully Connected Layer: 10 units (one per digit class)
    """

    def __init__(self) -> None:
        """Initialize the model architecture with all layers."""
        super(MNISTClassifier, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        
        # Max pooling layer (used for both conv layers)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected output layer
        # Input size calculation:
        # Input image: 28x28
        # After conv1 (3x3 kernel): 26x26
        # After pool: 13x13
        # After conv2 (3x3 kernel): 11x11
        # After pool: 5x5
        # With 64 channels: 64 * 5 * 5 = 1600
        self.fc = nn.Linear(64 * 5 * 5, 10)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)
            
        Returns:
            torch.Tensor: Raw logits of shape (batch_size, 10)
        """
        # First conv layer + ReLU + pooling
        x = self.pool(F.relu(self.conv1(x)))
        
        # Second conv layer + ReLU + pooling
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten the tensor for the fully connected layer
        x = x.view(-1, 64 * 5 * 5)
        
        # Output layer (logits)
        x = self.fc(x)
        
        return x


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 10,
    learning_rate: float = 0.001,
    model_save_path: str = "models/model.pth"
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """Train the model on MNIST dataset.
    
    Args:
        model: The neural network model to train
        train_loader: DataLoader for the training dataset
        test_loader: DataLoader for the test dataset
        epochs: Number of training epochs
        learning_rate: Learning rate for the optimizer
        model_save_path: Path to save the best model
        
    Returns:
        Tuple containing:
            - The trained model (best version based on validation accuracy)
            - Dictionary with training history (loss and accuracy metrics)
    """
    logger.info("Starting model training...")
    
    # Create directory for saving the model if it doesn't exist
    Path(os.path.dirname(model_save_path)).mkdir(parents=True, exist_ok=True)
    
    # Initialize device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Move model to device
    model = model.to(device)
    
    # Initialize loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Initialize tracking variables
    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
    best_accuracy = 0.0
    
    # Training loop
    for epoch in range(epochs):
        # Set model to training mode
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Training phase
        for batch_idx, (images, labels) in enumerate(train_loader):
            # Move data to device
            images, labels = images.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Print batch progress
            if (batch_idx + 1) % 100 == 0:
                logger.info(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}")
        
        # Calculate average training loss and accuracy for this epoch
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        
        # Evaluation phase
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
                
        # Calculate average test loss and accuracy
        avg_test_loss = test_loss / len(test_loader)
        test_accuracy = 100 * test_correct / test_total
        
        # Update history
        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(train_accuracy)
        history["test_loss"].append(avg_test_loss)
        history["test_acc"].append(test_accuracy)
        
        # Print epoch results
        logger.info(f"Epoch [{epoch+1}/{epochs}], "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
              f"Test Loss: {avg_test_loss:.4f}, Test Acc: {test_accuracy:.2f}%")
        
        # Save the best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            try:
                # Save model state dict
                torch.save(model.state_dict(), model_save_path)
                logger.info(f"Saved improved model with accuracy: {test_accuracy:.2f}%")
            except Exception as e:
                logger.error(f"Error saving model: {str(e)}")
    
    # Load the best model
    try:
        model.load_state_dict(torch.load(model_save_path))
        logger.info(f"Loaded best model from {model_save_path}")
    except Exception as e:
        logger.error(f"Error loading best model: {str(e)}")
    
    logger.info("Training completed!")
    return model, history


def print_model_summary(model: nn.Module) -> None:
    """Print a summary of the model architecture.
    
    Args:
        model: PyTorch model to summarize
    """
    print("\nModel Architecture Summary:")
    print("-" * 40)
    
    # Print model class name
    print(f"Model Type: {model.__class__.__name__}")
    
    # Count and print the total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"Total Parameters: {total_params:,}")
    
    # Print individual layers
    print("\nLayers:")
    for name, module in model.named_children():
        print(f"  {name}: {module}")
    
    print("-" * 40)
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


def plot_training_history(history: Dict[str, List[float]]) -> None:
    """Plot the training and testing metrics.
    
    Args:
        history: Dictionary containing training history metrics
    """
    try:
        import matplotlib.pyplot as plt
        
        # Create figure with two subplots
        plt.figure(figsize=(12, 5))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['test_loss'], label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Test Loss')
        plt.grid(True)
        
        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Train Accuracy')
        plt.plot(history['test_acc'], label='Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.title('Training and Test Accuracy')
        plt.grid(True)
        
        # Save the plot
        plt.tight_layout()
        Path('./models').mkdir(parents=True, exist_ok=True)
        plt.savefig('./models/training_history.png')
        logger.info("Training history plot saved to ./models/training_history.png")
        
    except ImportError:
        logger.warning("Matplotlib not installed. Skipping plot generation.")
    except Exception as e:
        logger.error(f"Error creating training history plot: {str(e)}")


if __name__ == "__main__":
    try:
        # Load MNIST dataset
        logger.info("Loading MNIST dataset...")
        train_loader, test_loader, dataset_info = load_mnist_data()
        
        # Display dataset information
        logger.info("\nMNIST Dataset Information:")
        logger.info(f"Training samples: {dataset_info['train_size']}")
        logger.info(f"Test samples: {dataset_info['test_size']}")
        logger.info(f"Image shape: {dataset_info['image_shape']}")
        logger.info(f"Number of classes: {dataset_info['classes']}")
        
        # Create model
        logger.info("Initializing model...")
        model = MNISTClassifier()
        print_model_summary(model)
        
        # Set training parameters
        num_epochs = 5
        learning_rate = 0.001
        model_save_path = "models/model.pth"
        
        # Train the model
        logger.info(f"Starting training for {num_epochs} epochs...")
        model, history = train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=num_epochs,
            learning_rate=learning_rate,
            model_save_path=model_save_path
        )
        
        # Plot training history
        plot_training_history(history)
        
        # Final evaluation
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        final_accuracy = 100 * correct / total
        logger.info(f"\nFinal Model Accuracy: {final_accuracy:.2f}%")
        logger.info(f"Model saved to: {model_save_path}")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        print(f"An error occurred: {str(e)}")
