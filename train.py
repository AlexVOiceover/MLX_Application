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
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        # Download and load training data
        logger.info("Loading MNIST training dataset...")
        train_dataset = datasets.MNIST(
            root=data_dir, train=True, download=True, transform=transform
        )

        # Download and load test data
        logger.info("Loading MNIST test dataset...")
        test_dataset = datasets.MNIST(
            root=data_dir, train=False, download=True, transform=transform
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=os.cpu_count() or 1,
            pin_memory=torch.cuda.is_available(),
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=os.cpu_count() or 1,
            pin_memory=torch.cuda.is_available(),
        )

        # Create a dictionary with dataset information
        dataset_info = {
            "train_size": len(train_dataset),
            "test_size": len(test_dataset),
            "image_shape": train_dataset[0][0].shape,
            "classes": len(train_dataset.classes),
            "class_labels": (
                train_dataset.classes
                if hasattr(train_dataset, "classes")
                else list(range(10))
            ),
            "batch_size": batch_size,
            "num_batches_train": len(train_loader),
            "num_batches_test": len(test_loader),
        }

        logger.info(
            f"MNIST dataset loaded with {dataset_info['train_size']} training and {dataset_info['test_size']} test samples"
        )

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
        - Fully Connected Layer 1: 128 units, ReLU activation
        - Dropout: 0.5 probability (during training only)
        - Fully Connected Layer 2 (Output): 10 units (one per digit class)

    This architecture strikes a balance between simplicity and effectiveness
    for the MNIST digit classification task. It uses a standard pattern of
    convolutional layers followed by max pooling to extract features, and
    fully connected layers for classification.
    """

    def __init__(self) -> None:
        """Initialize the model architecture with all layers."""
        super(MNISTClassifier, self).__init__()

        # First convolutional layer
        # Input: 1x28x28, Output: 32x26x26
        self.conv1 = nn.Conv2d(
            in_channels=1,  # MNIST images are grayscale (1 channel)
            out_channels=32,  # 32 different filters/feature maps
            kernel_size=3,  # 3x3 filter size
            stride=1,  # Standard stride
            padding=0,  # No padding
        )

        # Second convolutional layer
        # Input: 32x13x13 (after pooling), Output: 64x11x11
        self.conv2 = nn.Conv2d(
            in_channels=32,  # Input from first conv layer
            out_channels=64,  # 64 different filters/feature maps
            kernel_size=3,  # 3x3 filter size
            stride=1,  # Standard stride
            padding=0,  # No padding
        )

        # Max pooling layer (used twice in forward pass)
        # First use: 32x26x26 -> 32x13x13
        # Second use: 64x11x11 -> 64x5x5
        self.pool = nn.MaxPool2d(
            kernel_size=2, stride=2  # 2x2 pooling window  # Non-overlapping windows
        )

        # Determine the size of the flattened features after conv + pooling layers
        # After conv1 + pool: 32x13x13
        # After conv2 + pool: 64x5x5
        # Flattened: 64*5*5 = 1600
        self.fc1 = nn.Linear(64 * 5 * 5, 128)  # First fully connected layer
        self.dropout = nn.Dropout(0.5)  # Dropout layer for regularization
        self.fc2 = nn.Linear(128, 10)  # Output layer (10 classes for digits 0-9)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)

        Returns:
            torch.Tensor: Raw logits of shape (batch_size, 10)
        """
        # First conv layer + ReLU + pooling
        # Input: (batch_size, 1, 28, 28)
        # After conv1: (batch_size, 32, 26, 26)
        # After pooling: (batch_size, 32, 13, 13)
        x = self.pool(F.relu(self.conv1(x)))

        # Second conv layer + ReLU + pooling
        # Input: (batch_size, 32, 13, 13)
        # After conv2: (batch_size, 64, 11, 11)
        # After pooling: (batch_size, 64, 5, 5)
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten the tensor for the fully connected layers
        # Input: (batch_size, 64, 5, 5)
        # After flattening: (batch_size, 64*5*5=1600)
        x = x.view(-1, 64 * 5 * 5)

        # First fully connected layer + ReLU + dropout
        # Input: (batch_size, 1600)
        # After fc1 + ReLU: (batch_size, 128)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        # Final output layer (logits)
        # Input: (batch_size, 128)
        # Output: (batch_size, 10)
        x = self.fc2(x)

        return x

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make a prediction for input tensor x.

        This method:
        1. Sets the model to evaluation mode
        2. Performs a forward pass
        3. Applies softmax to get probabilities
        4. Returns the class with the highest probability

        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)

        Returns:
            torch.Tensor: Predicted class indices of shape (batch_size,)
        """
        self.eval()  # Set model to evaluation mode
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
            predicted_classes = torch.argmax(probabilities, dim=1)
        return predicted_classes


def train_model(model, train_loader, test_loader, epochs=10):
    """Train the model on MNIST dataset."""
    # TODO: Implement training loop
    pass


if __name__ == "__main__":
    # Demonstrate loading the MNIST dataset
    try:
        train_loader, test_loader, dataset_info = load_mnist_data()

        # Display dataset information
        print("\nMNIST Dataset Information:")
        print(f"Training samples: {dataset_info['train_size']}")
        print(f"Test samples: {dataset_info['test_size']}")
        print(f"Image shape: {dataset_info['image_shape']}")
        print(f"Number of classes: {dataset_info['classes']}")
        print(f"Batch size: {dataset_info['batch_size']}")
        print(f"Training batches: {dataset_info['num_batches_train']}")
        print(f"Test batches: {dataset_info['num_batches_test']}")

        # Get a batch of data to show sample dimensions
        images, labels = next(iter(train_loader))
        print(f"\nSample batch shape: {images.shape}")
        print(f"Sample labels shape: {labels.shape}")
        print(f"Sample labels: {labels[:10]}")

        print("\nDataset loaded successfully!")

    except Exception as e:
        logging.error(f"Error in dataset demonstration: {str(e)}")
        print(f"An error occurred: {str(e)}")
