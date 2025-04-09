.PHONY: setup setup-dev format lint type-check test run clean

# Variables
PYTHON = python
PIP = pip
VENV = venv
BIN = $(VENV)/bin
STREAMLIT = $(BIN)/streamlit

# Setup commands
setup:
	$(PIP) install -r requirements.txt

setup-dev: setup
	$(PIP) install -r requirements-dev.txt

# Code quality commands
format:
	$(BIN)/black .
	$(BIN)/isort .

lint:
	$(BIN)/flake8 .

type-check:
	$(BIN)/mypy app.py utils.py db.py train.py

# Testing command
test:
	$(BIN)/pytest

# Application commands
run:
	$(STREAMLIT) run app.py

# Train model
train:
	$(PYTHON) train.py

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
	@echo "  make setup      - Install production dependencies"
	@echo "  make setup-dev  - Install development dependencies"
	@echo "  make format     - Format code using black and isort"
	@echo "  make lint       - Run linting checks with flake8"
	@echo "  make type-check - Run type checking with mypy"
	@echo "  make test       - Run tests with pytest"
	@echo "  make run        - Run the Streamlit application"
	@echo "  make train      - Train the MNIST model"
	@echo "  make clean      - Clean up cache files"