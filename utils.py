#!/usr/bin/env python3
# utils.py

"""
Utility functions for image preprocessing and model operations.

This module provides functionality for preprocessing drawn digits,
converting them to tensors, and making predictions using the PyTorch model.
"""

# Import necessary libraries
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_environment_variables() -> Dict[str, Any]:
    """Load environment variables from .env file.

    This function loads environment variables from a .env file in the project
    root directory. It returns a dictionary containing the environment variables
    needed for the application.

    Returns:
        Dict[str, Any]: Dictionary containing environment variables with the following keys:
            - db_host: PostgreSQL server hostname
            - db_port: PostgreSQL server port
            - db_name: PostgreSQL database name
            - db_user: PostgreSQL username
            - db_password: PostgreSQL password
            - model_path: Path to the trained model file
            - debug: Boolean indicating if debug mode is enabled

    Example:
        >>> env_vars = load_environment_variables()
        >>> db_host = env_vars['db_host']
    """
    # Load environment variables from .env file
    load_dotenv()

    # Get database connection settings
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "5432")
    db_name = os.getenv("DB_NAME", "mnist_predictions")
    db_user = os.getenv("DB_USER", "postgres")
    db_password = os.getenv("DB_PASSWORD", "")

    # Get application settings
    model_path = os.getenv("MODEL_PATH", "models/model.pth")
    debug = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")

    return {
        "db_host": db_host,
        "db_port": int(db_port),
        "db_name": db_name,
        "db_user": db_user,
        "db_password": db_password,
        "model_path": model_path,
        "debug": debug,
    }


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


def preprocess_image(image_data):
    """Preprocess image data from Streamlit canvas."""
    # TODO: Implement image preprocessing
    pass


def load_model(model_path: str = "models/model.pth") -> Optional[MNISTClassifier]:
    """Load the trained PyTorch model.
    
    This function loads a trained MNISTClassifier model from a saved state dict file.
    It handles cases where the model might be called from different working directories
    by using absolute paths.
    
    Args:
        model_path: Path to the saved model file. Defaults to "models/model.pth".
        
    Returns:
        The loaded MNISTClassifier model in evaluation mode, or None if loading fails.
        
    Raises:
        FileNotFoundError: If the model file doesn't exist and can't be found.
    """
    try:
        # Convert to absolute path if it's a relative path
        if not os.path.isabs(model_path):
            # First try relative to current working directory
            abs_path = os.path.join(os.getcwd(), model_path)
            
            # If that doesn't exist, try relative to this file's directory
            if not os.path.exists(abs_path):
                module_dir = os.path.dirname(os.path.abspath(__file__))
                abs_path = os.path.join(module_dir, model_path)
                
                # If that still doesn't work, try one directory up (project root)
                if not os.path.exists(abs_path):
                    project_root = os.path.dirname(module_dir)
                    abs_path = os.path.join(project_root, model_path)
        else:
            abs_path = model_path
            
        # Check if the model file exists
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Model file not found at: {abs_path}")
            
        logger.info(f"Loading model from: {abs_path}")
        
        # Create a new instance of the model
        model = MNISTClassifier()
        
        # Load the state dictionary
        state_dict = torch.load(abs_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        
        # Set the model to evaluation mode
        model.eval()
        
        logger.info("Model loaded successfully")
        return model
        
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None
