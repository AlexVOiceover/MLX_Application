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


def train_model(model, train_loader, test_loader, epochs=10):
    """Train the model on MNIST dataset."""
    # TODO: Implement training loop
    pass


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


if __name__ == "__main__":
    try:
        # Demonstrate loading the MNIST dataset
        train_loader, test_loader, dataset_info = load_mnist_data()
        
        # Display dataset information
        print("\nMNIST Dataset Information:")
        print(f"Training samples: {dataset_info['train_size']}")
        print(f"Test samples: {dataset_info['test_size']}")
        print(f"Image shape: {dataset_info['image_shape']}")
        print(f"Number of classes: {dataset_info['classes']}")
        
        # Get a batch of data to show sample dimensions
        images, labels = next(iter(train_loader))
        print(f"Sample batch shape: {images.shape}")
        
        # Create model and display its architecture
        model = MNISTClassifier()
        print_model_summary(model)
        
        # Test a forward pass with sample data
        with torch.no_grad():
            output = model(images)
            print(f"\nForward pass test:")
            print(f"Input shape: {images.shape}")
            print(f"Output shape: {output.shape}")
        
        print("\nModel initialized successfully!")
        
    except Exception as e:
        logger.error(f"Error in demonstration: {str(e)}")
        print(f"An error occurred: {str(e)}")
