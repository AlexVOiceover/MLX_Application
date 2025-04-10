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
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dotenv import load_dotenv
from PIL import Image

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


def preprocess_image(image_data: np.ndarray) -> np.ndarray:
    """Preprocess image data from Streamlit canvas for MNIST model.
    
    This function takes raw image data from a Streamlit drawable canvas,
    which is typically an RGBA numpy array, and processes it to match
    the format expected by the MNIST model:
    - Resize to 28x28 pixels (MNIST standard)
    - Convert to grayscale
    - Normalize to [0, 1] range
    - Invert colors if needed (MNIST uses white digits on black background)
    
    Args:
        image_data: A numpy array containing image data from Streamlit canvas
            Usually in RGBA format with shape (height, width, 4)
    
    Returns:
        A preprocessed numpy array of shape (28, 28) with values in [0, 1] range,
        where the digit is white (1.0) on a black (0.0) background
        
    Raises:
        ValueError: If the input data is not a valid numpy array or is empty
        RuntimeError: If image processing operations fail
    """
    try:
        # Input validation
        if not isinstance(image_data, np.ndarray):
            raise ValueError("Input must be a numpy array")
        
        if image_data.size == 0:
            raise ValueError("Input array is empty")
            
        logger.info(f"Input image shape: {image_data.shape}, dtype: {image_data.dtype}")
        
        # Convert RGBA to grayscale if needed
        if len(image_data.shape) == 3 and image_data.shape[2] in [3, 4]:
            # For RGBA/RGB images from canvas, the drawing is typically white on transparent
            # We'll extract just the alpha channel if it exists (for transparency)
            if image_data.shape[2] == 4:  # RGBA
                # The alpha channel indicates where the user has drawn
                gray_image = image_data[:, :, 3]
            else:  # RGB
                # Convert RGB to grayscale using standard luminance formula
                gray_image = np.dot(image_data[..., :3], [0.2989, 0.5870, 0.1140])
        else:
            # Already grayscale
            gray_image = image_data
            
        # Resize to 28x28 (MNIST standard size)
        if gray_image.shape[0] != 28 or gray_image.shape[1] != 28:
            # Use PIL for high-quality resizing
            from PIL import Image
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(np.uint8(gray_image))
            # Resize to 28x28 with antialiasing
            pil_image = pil_image.resize((28, 28), Image.LANCZOS)
            # Convert back to numpy array
            gray_image = np.array(pil_image)
            
        # Normalize to [0, 1] range
        if gray_image.max() > 1.0:
            # Assuming values are in [0, 255] range
            gray_image = gray_image / 255.0
            
        # The MNIST dataset has white digits (1.0) on black background (0.0)
        # If our image is inverted (black digits on white), then invert it
        # We'll check the average color of the background vs foreground
        # If background is darker than foreground, we're already in MNIST format
        # Otherwise, invert the colors
        
        # A simple threshold to determine foreground pixels
        threshold = 0.3
        foreground_mask = gray_image > threshold
        
        # If less than 20% of pixels are foreground (typical for digits),
        # we check if we need to invert
        if np.mean(foreground_mask) < 0.2:
            # Calculate average of foreground and background
            foreground_avg = np.mean(gray_image[foreground_mask]) if np.any(foreground_mask) else 0
            background_avg = np.mean(gray_image[~foreground_mask]) if np.any(~foreground_mask) else 1
            
            # If background is lighter than foreground, invert the image
            if background_avg > foreground_avg:
                logger.info("Inverting image colors to match MNIST format")
                gray_image = 1.0 - gray_image
        
        logger.info(f"Preprocessed image shape: {gray_image.shape}")
        return gray_image
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise RuntimeError(f"Image preprocessing failed: {str(e)}")


def visualize_preprocessed_image(image_array: np.ndarray) -> np.ndarray:
    """Create a visualization of the preprocessed image for display.
    
    This helper function converts the preprocessed image into a format
    suitable for display using Streamlit's image display functions.
    
    Args:
        image_array: A preprocessed numpy array of shape (28, 28) with values in [0, 1]
        
    Returns:
        A numpy array of shape (28, 28, 3) in RGB format with values in [0, 255] range,
        suitable for display with st.image()
    """
    try:
        # Input validation
        if not isinstance(image_array, np.ndarray):
            raise ValueError("Input must be a numpy array")
            
        if image_array.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {image_array.shape}")
        
        # Create a larger image for better visibility
        from PIL import Image
        
        # Convert to 0-255 range
        display_image = (image_array * 255).astype(np.uint8)
        
        # Convert to PIL for resizing
        pil_image = Image.fromarray(display_image)
        
        # Create a visualization that shows the digit clearly
        # Return as numpy array in RGB format for st.image
        return np.array(pil_image.convert('RGB'))
        
    except Exception as e:
        logger.error(f"Error creating visualization: {str(e)}")
        # Return a simple gray image if visualization fails
        return np.ones((28, 28, 3), dtype=np.uint8) * 128


def convert_to_tensor(image_array: np.ndarray) -> torch.Tensor:
    """Convert preprocessed numpy array to PyTorch tensor for model inference.
    
    This function takes a preprocessed grayscale image as a numpy array and
    converts it to a PyTorch tensor with the correct dimensions and normalization
    for the MNIST model.
    
    Args:
        image_array: A preprocessed numpy array of shape (28, 28) with values in [0, 1]
        
    Returns:
        A PyTorch tensor of shape (1, 1, 28, 28) with MNIST normalization applied,
        ready for model inference. The dimensions are:
        - 1: Batch size (single image)
        - 1: Channels (grayscale)
        - 28, 28: Image height and width
        
    Raises:
        ValueError: If the input array doesn't have the expected shape or range
    """
    try:
        # Input validation
        if not isinstance(image_array, np.ndarray):
            raise ValueError("Input must be a numpy array")
            
        if image_array.ndim != 2 or image_array.shape[0] != 28 or image_array.shape[1] != 28:
            raise ValueError(f"Expected numpy array of shape (28, 28), got {image_array.shape}")
            
        if image_array.min() < 0 or image_array.max() > 1:
            raise ValueError("Input array values must be in range [0, 1]")
        
        # Convert to tensor
        tensor = torch.from_numpy(image_array.astype(np.float32))
        
        # Add batch and channel dimensions: (28, 28) -> (1, 1, 28, 28)
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        
        # Apply MNIST normalization (same as in training)
        mean = 0.1307
        std = 0.3081
        tensor = (tensor - mean) / std
        
        logger.info(f"Converted tensor shape: {tensor.shape}")
        return tensor
        
    except Exception as e:
        logger.error(f"Error converting to tensor: {str(e)}")
        raise RuntimeError(f"Tensor conversion failed: {str(e)}")


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


def get_confidence_scores(output_tensor: torch.Tensor) -> list[float]:
    """Convert model output logits to confidence scores using softmax.
    
    Args:
        output_tensor: The raw output tensor from the model's forward pass,
            typically of shape (batch_size, num_classes)
            
    Returns:
        A list of confidence scores (probabilities) for each digit class (0-9),
        with values in the range [0.0, 1.0] that sum to 1.0
        
    Raises:
        ValueError: If the input tensor is not of the expected shape
    """
    try:
        if not isinstance(output_tensor, torch.Tensor):
            raise ValueError("Input must be a PyTorch tensor")
            
        # Handle both batched and single example outputs
        if output_tensor.dim() == 2:
            # For batched output, we use the first example
            if output_tensor.shape[0] > 1:
                logger.warning(f"Received batch of {output_tensor.shape[0]} examples, using only the first one")
            if output_tensor.shape[1] != 10:
                raise ValueError(f"Expected 10 output classes, got {output_tensor.shape[1]}")
            # Apply softmax along the class dimension
            probabilities = F.softmax(output_tensor[0], dim=0)
        elif output_tensor.dim() == 1:
            # For single example with just class scores
            if output_tensor.shape[0] != 10:
                raise ValueError(f"Expected 10 output classes, got {output_tensor.shape[0]}")
            # Apply softmax to the class scores
            probabilities = F.softmax(output_tensor, dim=0)
        else:
            raise ValueError(f"Expected tensor of dim 1 or 2, got {output_tensor.dim()}")
        
        # Convert to list of float values
        return probabilities.tolist()
        
    except Exception as e:
        logger.error(f"Error calculating confidence scores: {str(e)}")
        # Return a list of zeros with 1.0 at position 0 as a fallback
        return [1.0 if i == 0 else 0.0 for i in range(10)]


def predict_digit(model: MNISTClassifier, image_tensor: torch.Tensor) -> Tuple[int, float]:
    """Predict a digit using the trained model.
    
    This function takes a preprocessed tensor representation of a digit image
    and passes it through the model to predict which digit it represents.
    
    Args:
        model: A trained MNISTClassifier model in evaluation mode
        image_tensor: A preprocessed image tensor of shape (1, 1, 28, 28)
        
    Returns:
        Tuple containing:
            - predicted digit (0-9)
            - confidence score (0.0-1.0) for the predicted digit
            
    Raises:
        ValueError: If inputs don't match expected formats
        RuntimeError: If prediction fails
    """
    try:
        if not isinstance(model, MNISTClassifier):
            raise ValueError("Model must be an instance of MNISTClassifier")
            
        if not isinstance(image_tensor, torch.Tensor):
            raise ValueError("Image must be a PyTorch tensor")
            
        if image_tensor.dim() != 4 or image_tensor.shape[0] != 1 or image_tensor.shape[1] != 1:
            raise ValueError(f"Expected tensor of shape (1, 1, height, width), got {image_tensor.shape}")
        
        # Ensure model is in evaluation mode
        model.eval()
        
        # Make prediction
        with torch.no_grad():  # Disable gradient computation for inference
            # Forward pass
            outputs = model(image_tensor)
            
            # Get confidence scores
            confidence_scores = get_confidence_scores(outputs)
            
            # Find the class with the highest confidence
            predicted_class = confidence_scores.index(max(confidence_scores))
            confidence = confidence_scores[predicted_class]
            
            return predicted_class, confidence
            
    except Exception as e:
        logger.error(f"Error predicting digit: {str(e)}")
        raise RuntimeError(f"Prediction failed: {str(e)}")


def predict_with_all_scores(model: MNISTClassifier, image_tensor: torch.Tensor) -> Tuple[int, float, list[float]]:
    """Predict a digit and return all confidence scores.
    
    This is an enhanced version of predict_digit that also returns
    confidence scores for all possible classes.
    
    Args:
        model: A trained MNISTClassifier model
        image_tensor: A preprocessed image tensor of shape (1, 1, 28, 28)
        
    Returns:
        Tuple containing:
            - predicted digit (0-9)
            - confidence score (0.0-1.0) for the predicted digit
            - list of confidence scores for all digits (0-9)
    """
    try:
        # Ensure model is in evaluation mode
        model.eval()
        
        with torch.no_grad():
            # Forward pass
            outputs = model(image_tensor)
            
            # Get confidence scores for all classes
            all_confidences = get_confidence_scores(outputs)
            
            # Find the class with the highest confidence
            predicted_class = all_confidences.index(max(all_confidences))
            confidence = all_confidences[predicted_class]
            
            return predicted_class, confidence, all_confidences
            
    except Exception as e:
        logger.error(f"Error in prediction with scores: {str(e)}")
        raise RuntimeError(f"Prediction with scores failed: {str(e)}")


def test_image_processing():
    """Test the image preprocessing and tensor conversion functions.
    
    This function creates a simple test image with a digit, processes it
    using the preprocessing functions, and displays the results.
    
    Returns:
        Tuple of (preprocessed_image, tensor) for further testing
    """
    try:
        # Create a simple test image (a white digit 5 on black background)
        # This is just a simplified representation for testing
        test_image = np.zeros((100, 100), dtype=np.uint8)
        
        # Draw a simple digit (e.g., number 5)
        # Top horizontal line
        test_image[20:30, 30:70] = 255
        # Left vertical line (top)
        test_image[30:50, 30:40] = 255
        # Middle horizontal line
        test_image[50:60, 30:70] = 255
        # Right vertical line (bottom)
        test_image[60:80, 60:70] = 255
        # Bottom horizontal line
        test_image[80:90, 30:60] = 255
        
        print("1. Created test image")
        
        # Preprocess the image
        preprocessed = preprocess_image(test_image)
        print(f"2. Preprocessed image shape: {preprocessed.shape}, min: {preprocessed.min()}, max: {preprocessed.max()}")
        
        # Convert to tensor
        tensor = convert_to_tensor(preprocessed)
        print(f"3. Tensor shape: {tensor.shape}, min: {tensor.min().item():.4f}, max: {tensor.max().item():.4f}")
        
        # Try to load the model
        try:
            model = load_model()
            if model:
                # Make predictions with both functions
                digit, confidence = predict_digit(model, tensor)
                print(f"4. Basic prediction: digit={digit}, confidence={confidence:.4f}")
                
                # Use the function that returns all scores
                digit, confidence, all_confidences = predict_with_all_scores(model, tensor)
                print(f"5. Detailed prediction: digit={digit}, confidence={confidence:.4f}")
                print(f"   All confidence scores: {[f'{i}: {conf:.4f}' for i, conf in enumerate(all_confidences)]}")
            else:
                print("4. Model not found, skipping prediction test")
        except Exception as e:
            print(f"4. Model loading or prediction failed: {e}")
        
        return preprocessed, tensor
        
    except Exception as e:
        print(f"Error in test_image_processing: {e}")
        return None, None


if __name__ == "__main__":
    # Test the image processing utilities
    print("\n=== Testing Image Processing Utilities ===\n")
    preprocessed, tensor = test_image_processing()
    
    if preprocessed is not None and tensor is not None:
        print("\n=== Test Completed Successfully ===")
    else:
        print("\n=== Test Failed ===")
        
    print("\nTo use these utilities in your Streamlit app:")
    print("""
    # Example usage in Streamlit app
    from utils import (
        preprocess_image, convert_to_tensor, 
        load_model, predict_digit, get_confidence_scores,
        visualize_preprocessed_image
    )
    
    # Get image data from Streamlit canvas
    image_data = st_canvas.image_data
    
    # Preprocess the image
    preprocessed = preprocess_image(image_data)
    
    # Visualize preprocessed image (optional)
    display_image = visualize_preprocessed_image(preprocessed)
    st.image(display_image, caption="Preprocessed Image", width=150)
    
    # Convert to tensor
    tensor = convert_to_tensor(preprocessed)
    
    # Load model
    model = load_model()
    
    # Make prediction (just the digit and its confidence)
    digit, confidence = predict_digit(model, tensor)
    
    # Display result
    st.write(f"Predicted digit: {digit} (Confidence: {confidence:.2%})")
    
    # For advanced display, you can get all confidence scores
    # _, _, all_scores = predict_with_all_scores(model, tensor)
    # Create a bar chart of all confidence scores
    # st.bar_chart({f"Digit {i}": score for i, score in enumerate(all_scores)})
    """)
