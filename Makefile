.PHONY: setup setup-dev format lint type-check test run clean

# Variables
PYTHON = python
CONDA_ENV = mlx-app
STREAMLIT = streamlit

# Setup commands
setup:
	@echo "Creating conda environment and installing dependencies..."
	conda env create -f environment.yml

update:
	@echo "Updating conda environment..."
	conda env update -f environment.yml

# Code quality commands
format:
	conda run -n $(CONDA_ENV) black .
	conda run -n $(CONDA_ENV) isort .

lint:
	conda run -n $(CONDA_ENV) flake8 .

type-check:
	conda run -n $(CONDA_ENV) mypy app.py utils.py db.py train.py

# Testing command
test:
	conda run -n $(CONDA_ENV) pytest

# Application commands
run:
	conda run -n $(CONDA_ENV) $(STREAMLIT) run app.py

# Train model
train:
	conda run -n $(CONDA_ENV) $(PYTHON) train.py

# Clean up command
clean:
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Help command
help:
	@echo "Available commands:"
	@echo "  make setup     - Create conda environment and install dependencies"
	@echo "  make update    - Update conda environment with newest dependencies"
	@echo "  make format    - Format code using black and isort"
	@echo "  make lint      - Run linting checks with flake8"
	@echo "  make type-check - Run type checking with mypy"
	@echo "  make test      - Run tests with pytest"
	@echo "  make run       - Run the Streamlit application"
	@echo "  make train     - Train the MNIST model"
	@echo "  make clean     - Clean up cache files"