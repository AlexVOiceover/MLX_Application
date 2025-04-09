# MNIST Digit Classifier

A web application for recognizing handwritten digits using a PyTorch model trained on the MNIST dataset. Users can draw digits directly on the interface, receive predictions with confidence scores, provide feedback on the accuracy, and view prediction history.

## Features

- Draw digits on a canvas interface
- Real-time digit recognition using a CNN model
- Prediction confidence visualization
- User feedback collection
- Prediction history logging to PostgreSQL

## Development Setup

### Prerequisites

- Python 3.10+
- PostgreSQL 15

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd mnist-digit-classifier
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
   - On Windows:
     ```bash
     .\venv\Scripts\activate
     ```

4. Install dependencies:
   ```bash
   make setup        # Install production dependencies
   make setup-dev    # Install development dependencies
   ```

5. Configure the environment:
   ```bash
   cp .env.example .env
   # Edit .env with your PostgreSQL credentials
   ```

### Development Tools

This project uses several development tools to ensure code quality:

- **Black**: Automatic code formatting 
  ```bash
  make format    # Format code with black and isort
  ```

- **Flake8**: Code linting
  ```bash
  make lint      # Run linting checks 
  ```

- **isort**: Import sorting
  ```bash
  # Included in the format command
  ```

- **mypy**: Type checking 
  ```bash
  make type-check  # Run type checking
  ```

- **pytest**: Testing framework
  ```bash
  make test      # Run tests
  ```

### Running the Application

1. Train the model (if not already trained):
   ```bash
   make train
   ```

2. Start the application:
   ```bash
   make run
   ```

3. Open your browser and navigate to http://localhost:8501

## Project Structure

- `app.py`: Main Streamlit application
- `utils.py`: Image preprocessing and model utilities
- `db.py`: Database interactions
- `train.py`: Model training script
- `models/`: Directory for storing trained models

## License

[Insert license information here]