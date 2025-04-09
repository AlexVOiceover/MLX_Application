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
from typing import Tuple, Dict, Any

import torch
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


class MNISTClassifier:
    """CNN model for MNIST digit classification."""

    # TODO: Implement model architecture
    pass


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
