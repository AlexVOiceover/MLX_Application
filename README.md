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

- [Miniconda](https://docs.anaconda.com/free/miniconda/index.html) or [Anaconda](https://www.anaconda.com/download)
- PostgreSQL 15 (automatically installed in the conda environment)

### Dependencies

This project uses the following main dependencies:
- **PyTorch (2.6.x)** and **torchvision (0.21.x)**: Deep learning framework for model training and inference
- **Streamlit (1.32.x)**: Web application framework for the user interface
- **streamlit-drawable-canvas**: Component for drawing digits on the web interface
- **psycopg2 (2.9.x)**: PostgreSQL adapter for database interactions
- **Pillow (10.x)** and **NumPy (1.26.x)**: Image processing libraries
- **python-dotenv (1.0.x)**: Environment variable management

All dependencies are managed through Conda using the `environment.yml` file.

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd MLX_Application
   ```

2. Create and activate the conda environment:
   ```bash
   make setup
   conda activate mlx-app
   ```

3. Select the correct interpreter in your IDE/editor:
   - **VS Code**:
     - Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
     - Type "Python: Select Interpreter"
     - Select the interpreter from your conda environment (should show `Python 3.10.x ('mlx-app': conda)`)
   - **PyCharm**:
     - Go to `File > Settings > Project > Python Interpreter`
     - Click the gear icon and select "Add..."
     - Choose "Conda Environment" and select your existing environment
   - **Command Line**:
     - Verify the correct interpreter with `which python`
     - It should point to a path in your conda environment

   > **⚠️ Important**: Using the wrong interpreter will cause package import errors or 404 errors!

4. Configure the environment:
   ```bash
   cp .env.example .env
   # Edit .env with your PostgreSQL credentials and application settings
   ```

### Environment Variables

The application uses the following environment variables that should be defined in the `.env` file:

| Variable      | Description                      | Example Value            |
|---------------|----------------------------------|--------------------------|
| DB_HOST       | PostgreSQL server hostname       | localhost                |
| DB_PORT       | PostgreSQL server port           | 5432                     |
| DB_NAME       | PostgreSQL database name         | mnist_predictions        |
| DB_USER       | PostgreSQL username              | postgres                 |
| DB_PASSWORD   | PostgreSQL password              | your_password_here       |
| MODEL_PATH    | Path to the trained model file   | models/model.pth         |
| DEBUG         | Enable/disable debug mode        | False                    |

Make sure to configure these variables before running the application.

#### Using Environment Variables in the Application

The application uses the `load_environment_variables()` function from `utils.py` to load these variables:

```python
from utils import load_environment_variables

# Load environment variables
env_vars = load_environment_variables()

# Access variables
db_host = env_vars["db_host"]
db_port = env_vars["db_port"]
model_path = env_vars["model_path"]
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
- `environment.yml`: Conda environment configuration

## Troubleshooting

### Common Issues

#### Package Import Errors or 404 Errors

If you see errors like `ModuleNotFoundError` or HTTP 404 errors when installing packages:

1. **Check your active environment**:
   ```bash
   # Should show (mlx-app)
   conda info --envs
   ```

2. **Verify the interpreter**:
   ```bash
   # Should point to a path in your conda environment
   which python
   python --version
   ```

3. **Ensure you've selected the correct interpreter in your IDE/editor**:
   - See the installation steps above for selecting the correct interpreter

4. **Update the conda environment**:
   ```bash
   make update
   ```

#### Database Connection Issues

If you encounter database connection errors:

1. **Check your .env file**:
   - Ensure DB_HOST, DB_PORT, DB_NAME, DB_USER, and DB_PASSWORD are set correctly

2. **Verify PostgreSQL is running**:
   ```bash
   # On Linux
   sudo systemctl status postgresql
   # On macOS with Homebrew
   brew services list
   ```

3. **Test the connection directly**:
   ```bash
   psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME
   ```

## License

[Insert license information here]