Implementation Plan for MNIST Digit Classifier
For MNIST Digit Classifier
Version 1.0
Prepared by Technical Project Planner
April 9, 2025
Revision History
NameDateReason For ChangesVersionTechnical Project PlannerApril 9, 2025Initial Draft1.0
Implementation Tracking
Tracking Mechanism

Each task has a unique identifier (e.g., 1.1, 1.2)
Status: Not Started → In Progress → Completed
Dependencies clearly identified

Completion Criteria

 Project structure established
 Development tools configured
 Model training implemented
 Database functionality working
 Streamlit UI implemented
 All components integrated
 Containerization complete
 Documentation finalized

Continuous Improvement

Regular code reviews
Periodic architecture reassessment
Performance monitoring

Implementation Plan
1. Project Initialization
Objective: Create the foundational project structure and environment
1.1 Setup Project Structure

 Task - Prompt:

Create the initial project structure for an MNIST Digit Classifier application. This should include:

1. Create a directory structure following these specifications:
   - Root directory: `mnist-digit-classifier`
   - Subdirectories: `models/` (for storing the trained model)

2. Create the following empty Python files in the root directory:
   - `app.py` (main Streamlit application)
   - `utils.py` (image preprocessing utilities)
   - `db.py` (database interaction)
   - `train.py` (model training script)

3. Create configuration files:
   - `.gitignore` (include standard Python patterns, .env, and any environment-specific files)
   - `requirements.txt` (leave empty for now, we'll populate it in the next step)
   - `.env.example` (template for environment variables, without actual values)

Return the commands needed to create this structure and the content of each file (can be placeholder comments for now).
1.2 Development Tools Setup

 Task - Prompt:

Set up code quality and development tools for the MNIST Digit Classifier project. Specifically:

1. Create a comprehensive requirements-dev.txt file that includes:
   - flake8 (for linting)
   - black (for code formatting)
   - isort (for import sorting)
   - mypy (for type checking)
   - pytest (for testing)

2. Create configuration files for these tools:
   - .flake8 (flake8 configuration)
   - pyproject.toml (black and isort configuration)
   - mypy.ini (mypy configuration)

3. Create a Makefile with commands for:
   - Installing dependencies (both production and development)
   - Running linting and formatting checks
   - Running type checking
   - Running tests
   - Starting the application

4. Update the README.md to include information about the development tools and how to use them

Return the content for all these files with proper configurations that match the project standards.
1.3 Setup Dependencies

 Task - Prompt:

Create a comprehensive requirements.txt file for the MNIST Digit Classifier project based on these specifications:

1. The project uses:
   - Python 3.10+
   - PyTorch 2.2.x and torchvision 0.17.x for the model
   - Streamlit 1.32.x for the UI
   - psycopg2 2.9.x for PostgreSQL database connection
   - Pillow 10.x and NumPy 1.26.x for image processing
   - python-dotenv 1.0.x for environment configuration
   - streamlit-drawable-canvas for drawing interface

2. Pin specific versions to ensure reproducibility

3. Expand the README.md with:
   - Brief project description
   - Setup instructions for development environment
   - Required environment variables
   - How to install dependencies

Return the content for both files.
1.4 Environment Configuration

 Task - Prompt:

Create environment configuration files for the MNIST Digit Classifier. Specifically:

1. Create a `.env.example` file with placeholders for:
   - PostgreSQL connection details (host, port, database, user, password)
   - Any application-specific settings (e.g., model path)
   - Debug mode flag

2. Create a simple Python function in `utils.py` to load environment variables using python-dotenv.

3. Update the README.md to explain how to use the .env file.

Remember to follow the project standards, using snake_case for functions and including docstrings.
2. Model Development
Objective: Create and train a PyTorch model on the MNIST dataset
2.1 Create Data Loading

 Task - Prompt:

Implement the MNIST dataset loading functionality in the `train.py` file. Specifically:

1. Import necessary libraries (PyTorch, torchvision, etc.)

2. Create a function `load_mnist_data()` that:
   - Downloads the MNIST training and test datasets if not already available
   - Creates DataLoader objects for both with appropriate batch sizes
   - Applies necessary transformations (ToTensor, Normalize)
   - Returns the training and test dataloaders

3. Add appropriate error handling and logging

4. If the file is run directly, it should demonstrate loading the data and display basic dataset information (size, dimensions, etc.)

Follow PEP8 conventions and add type hints.
2.2 Create Model Architecture

 Task - Prompt:

Implement a CNN model architecture for MNIST digit classification in the `train.py` file. Specifically:

1. Create a PyTorch nn.Module subclass called `MNISTClassifier` that:
   - Implements a simple but effective CNN architecture with:
     - 2 convolutional layers with ReLU activations and max pooling
     - Fully connected output layer with 10 classes (digits 0-9)
   - Includes a forward method that processes a batch of images

2. Add docstrings explaining the architecture

3. If the file is run directly, it should instantiate the model and print its architecture summary

Follow PyTorch best practices, PEP8 conventions, and add appropriate type hints.
2.3 Implement Training Loop

 Task - Prompt:

Implement the training loop for the MNIST model in the `train.py` file. Building on the previous code:

1. Create a function `train_model(model, train_loader, test_loader, epochs=10)` that:
   - Takes the model, training data, test data, and number of epochs
   - Uses cross-entropy loss and Adam optimizer
   - Trains the model for the specified number of epochs
   - Evaluates accuracy on the test set after each epoch
   - Returns the trained model and a history of metrics

2. Add functionality to save the best model based on validation accuracy to the 'models/model.pth' file

3. If the file is run directly, it should:
   - Load the data (using your previous function)
   - Create the model (using your previous class)
   - Train the model
   - Save the best model
   - Print final accuracy metrics

Include progress reporting during training and proper error handling.
2.4 Add Model Loading Utility

 Task - Prompt:

Create a model loading utility in the `utils.py` file to allow the Streamlit app to load the trained model. Specifically:

1. Import necessary libraries (PyTorch, etc.)

2. Implement a function `load_model(model_path='models/model.pth')` that:
   - Takes a path to the saved model file
   - Creates an instance of the MNISTClassifier class (copy the class definition from train.py)
   - Loads the saved weights into the model
   - Sets the model to evaluation mode
   - Returns the loaded model

3. Add appropriate error handling if the model file is not found

4. Ensure this function works even if called from different working directories

Follow PEP8 conventions and add type hints.
3. Database Framework
Objective: Create database functionality for logging predictions
3.1 Database Connection

 Task - Prompt:

Implement the database connection functionality in the `db.py` file. Specifically:

1. Import necessary libraries (psycopg2, dotenv, etc.)

2. Create a function `get_db_connection()` that:
   - Loads database credentials from environment variables
   - Establishes a connection to PostgreSQL
   - Returns the connection object

3. Create a function `close_connection(conn)` to properly close the connection

4. Add appropriate error handling and connection pooling best practices

5. Add a simple test function that verifies the connection works when run directly

Follow PEP8 conventions and add type hints. Ensure the code handles connection errors gracefully.
3.2 Create Database Schema

 Task - Prompt:

Implement the database schema creation functionality in the `db.py` file. Building on the previous code:

1. Create a function `create_tables()` that:
   - Uses the database connection from your previous function
   - Creates a 'predictions' table (if it doesn't exist) with columns for:
     - id (primary key)
     - timestamp
     - predicted_digit (integer)
     - confidence (float)
     - true_digit (integer, can be NULL)

2. Add appropriate error handling

3. If the file is run directly, it should:
   - Call the create_tables() function
   - Print confirmation that the tables were created successfully

Follow PostgreSQL best practices for table design and indexing.
3.3 Implement Prediction Logging

 Task - Prompt:

Implement the prediction logging functionality in the `db.py` file. Building on the previous code:

1. Create a function `log_prediction(predicted_digit, confidence, true_digit=None)` that:
   - Takes the predicted digit (int), confidence score (float), and optional true digit (int)
   - Gets a database connection
   - Inserts a new record into the predictions table with the current timestamp
   - Commits the transaction
   - Closes the connection
   - Returns the ID of the new record

2. Add appropriate error handling and transaction management

3. If the file is run directly, it should include a test that logs a sample prediction

Follow database best practices for parameterized queries to prevent SQL injection.
3.4 Implement History Retrieval

 Task - Prompt:

Implement the prediction history retrieval functionality in the `db.py` file. Building on the previous code:

1. Create a function `get_prediction_history(limit=100)` that:
   - Takes an optional limit parameter (default 100)
   - Gets a database connection
   - Queries the predictions table for the most recent entries (ordered by timestamp)
   - Returns the results as a list of dictionaries

2. Add appropriate error handling

3. If the file is run directly, it should:
   - Call the function and print a sample of the results
   - Handle the case of an empty database gracefully

Ensure the function returns results in a format that's easily consumable by Streamlit (e.g., can be passed directly to st.dataframe()).
4. Image Processing Utilities
Objective: Create utility functions for processing drawn digits
4.1 Implement Image Preprocessing

 Task - Prompt:

Implement image preprocessing utilities in the `utils.py` file. Specifically:

1. Import necessary libraries (PIL, numpy, torch, etc.)

2. Create a function `preprocess_image(image_data)` that:
   - Takes image data from a Streamlit canvas (numpy array)
   - Resizes it to 28x28 pixels (MNIST standard size)
   - Converts it to grayscale if it's not already
   - Normalizes pixel values to the range [0, 1]
   - Inverts the colors if needed (MNIST has white digits on black background)
   - Returns the preprocessed image as a numpy array

3. Create a function `convert_to_tensor(image_array)` that:
   - Takes a preprocessed numpy array
   - Converts it to a PyTorch tensor
   - Adds a batch dimension and channel dimension if needed
   - Applies the same normalization as used during training (mean=0.1307, std=0.3081)
   - Returns a tensor ready for model inference

4. Add appropriate error handling and input validation

Follow PyTorch best practices for tensor manipulation and add type hints.
4.2 Implement Prediction Function

 Task - Prompt:

Implement a prediction function in the `utils.py` file that uses the model to predict digits. Building on the previous code:

1. Create a function `predict_digit(model, preprocessed_tensor)` that:
   - Takes a trained model and a preprocessed tensor
   - Runs the model inference with the tensor as input
   - Gets the predicted class (0-9) and the confidence score
   - Returns a tuple of (predicted_digit, confidence_score)

2. Add a function `get_confidence_scores(output_tensor)` that:
   - Takes the raw output tensor from the model
   - Applies softmax to get probabilities
   - Returns a list of confidence scores for each possible digit (0-9)

3. Add appropriate error handling and ensure prediction works in evaluation mode

Follow PyTorch best practices for inference and add type hints.
5. Streamlit UI Development
Objective: Create the user interface using Streamlit
5.1 Basic App Structure

 Task - Prompt:

Create the basic structure for the Streamlit app in `app.py`. Specifically:

1. Import necessary libraries (streamlit, PIL, numpy, torch, etc.)

2. Import your custom modules (utils, db)

3. Set up the app structure with:
   - A title and brief description
   - Section headers for drawing, prediction, and history
   - Basic page configuration (title, favicon, layout)

4. Add model loading code that:
   - Uses your utils.load_model() function to load the trained model when the app starts
   - Caches the model to prevent reloading (using @st.cache_resource)
   - Handles errors if model loading fails

5. If the file is run directly, it should start the Streamlit app

Follow Streamlit best practices and ensure the UI is user-friendly.
5.2 Drawing Canvas Implementation

 Task - Prompt:

Implement the drawing canvas in the Streamlit app. Building on the previous code in `app.py`:

1. Import the streamlit_drawable_canvas package

2. Add the package to requirements.txt if not already included

3. Create a drawing section with:
   - A streamlit_drawable_canvas component sized appropriately (e.g., 280x280 pixels)
   - A clear button to reset the canvas
   - Instructions for the user

4. Add functionality to retrieve the image data from the canvas as a numpy array

5. Add a "Predict" button that will be connected to prediction functionality later

Ensure the drawing experience is smooth and the canvas is properly configured for digit drawing (black background, white stroke).
5.3 Prediction Display

 Task - Prompt:

Implement the prediction display in the Streamlit app. Building on the previous code in `app.py`:

1. Create a function `make_prediction()` that:
   - Gets the image data from the canvas
   - Uses your utils functions to preprocess the image and convert to tensor
   - Uses your prediction function to get the predicted digit and confidence
   - Returns the prediction result

2. Connect this function to the "Predict" button

3. Create a display section that shows:
   - The predicted digit (large and visible)
   - The confidence score (as a percentage)
   - A visual indication of confidence (e.g., color coding or progress bar)

4. Add a form for the user to input the correct digit (for feedback) with:
   - A number input limited to 0-9
   - A "Submit" button

Follow Streamlit best practices for forms and buttons.
5.4 History Display Implementation

 Task - Prompt:

Implement the prediction history display in the Streamlit app. Building on the previous code in `app.py`:

1. Create a function `show_history()` that:
   - Uses your db.get_prediction_history() function to retrieve recent predictions
   - Formats the data appropriately (e.g., formats timestamp, converts confidence to percentage)
   - Displays the data in a Streamlit dataframe or table

2. Add this function to an appropriate section of the UI

3. Implement auto-refresh functionality so the history updates after each new prediction

4. Format the table for readability (column widths, column names, etc.)

Follow Streamlit best practices for data display and ensure the table is sortable if possible.
5.5 Feedback Submission

 Task - Prompt:

Implement the feedback submission functionality in the Streamlit app. Building on the previous code in `app.py`:

1. Connect the "Submit" button from the feedback form to a function that:
   - Gets the predicted digit and confidence from the current prediction
   - Gets the user-provided true digit from the form
   - Uses your db.log_prediction() function to store the data
   - Shows a success message to the user
   - Triggers a refresh of the history display

2. Add appropriate validation to ensure:
   - A prediction has been made before feedback can be submitted
   - The true digit input is valid (0-9)

3. Add error handling for database connection issues

Follow Streamlit best practices for form submission and user feedback.
6. Integration and Error Handling
Objective: Connect all components and add robust error handling
6.1 Error Handling

 Task - Prompt:

Enhance error handling throughout the application. Specifically:

1. In `utils.py`, improve error handling for:
   - Model loading failures
   - Image preprocessing issues
   - Tensor conversion problems

2. In `db.py`, improve error handling for:
   - Database connection failures
   - Query execution errors
   - Transaction management

3. In `app.py`, add user-friendly error messages for:
   - Model loading failures
   - Prediction errors
   - Database connection issues
   - Invalid user input

4. Create a utility function `display_error(error_message)` in `app.py` that:
   - Shows an error message to the user in a consistent format
   - Logs the detailed error internally

Make sure all errors are handled gracefully without crashing the application and that users receive helpful feedback.
6.2 Performance Optimization

 Task - Prompt:

Optimize the performance of the application. Specifically:

1. In `utils.py`:
   - Add caching for expensive operations where appropriate
   - Optimize image preprocessing for speed

2. In `db.py`:
   - Implement connection pooling for better database performance
   - Add indexing recommendations for the predictions table

3. In `app.py`:
   - Use Streamlit's caching mechanisms (st.cache_data, st.cache_resource) appropriately
   - Minimize re-runs of expensive operations

4. Look for opportunities to make the application more responsive overall

Document your optimizations with comments explaining the reasoning behind each change.
7. Containerization
Objective: Containerize the application using Docker
7.1 Create Dockerfile

 Task - Prompt:

Create a Dockerfile for the MNIST Digit Classifier application. Specifically:

1. Start from a Python 3.10 slim base image

2. Install only the necessary dependencies from requirements.txt

3. Copy the application files to appropriate locations in the container

4. Set up a non-root user for running the application

5. Configure the entrypoint to start the Streamlit app

6. Expose the appropriate port (default Streamlit port is 8501)

Follow Docker best practices including multi-stage builds if appropriate and keeping the image size minimal.
7.2 Create Docker Compose File

 Task - Prompt:

Create a docker-compose.yml file to orchestrate the application and database. Specifically:

1. Define two services:
   - `app`: The Streamlit application using the Dockerfile
   - `db`: PostgreSQL database (use official PostgreSQL image)

2. Configure environment variables for both services:
   - Use variables from .env file for sensitive information
   - Set appropriate defaults where possible

3. Set up volume mounting for the database to persist data

4. Configure networking between the services

5. Add healthchecks for both services

6. Include comments explaining key configuration choices

Follow Docker Compose best practices and ensure all necessary variables are properly set for both development and production use.
8. Testing and Documentation
Objective: Add tests and complete project documentation
8.1 Create Tests

 Task - Prompt:

Create basic tests for the application components. Specifically:

1. Create a `tests` directory with:
   - `test_utils.py` to test image preprocessing and model functions
   - `test_db.py` to test database functions

2. For utils tests, include:
   - Test for image preprocessing function with sample input
   - Test for tensor conversion function
   - Test for model loading with a mock model file

3. For database tests, include:
   - Test for connection function (with a mock or test database)
   - Test for prediction logging function
   - Test for history retrieval function

4. Add a simple way to run all tests (e.g., using pytest)

5. Update the Makefile to include commands for running tests

Follow testing best practices, including using fixtures and mocking external dependencies where appropriate.
8.2 Complete Documentation

 Task - Prompt:

Complete the project documentation. Specifically:

1. Update the README.md with:
   - Comprehensive project description
   - Screenshots of the application
   - Detailed setup instructions for both development and production
   - Usage instructions
   - Troubleshooting section

2. Add docstrings to all functions in:
   - utils.py
   - db.py
   - app.py
   - train.py

3. Create a DEPLOYMENT.md file with:
   - Step-by-step instructions for deploying on Hetzner
   - Environment setup requirements
   - Monitoring and maintenance suggestions

4. Create a CONTRIBUTING.md file with:
   - Code style guidelines
   - Pull request process
   - Development environment setup

Follow Markdown best practices and ensure all documentation is clear, accurate, and helpful.
9. Final Integration and Deployment
Objective: Ensure all components work together seamlessly and prepare for deployment
9.1 Integration Testing

 Task - Prompt:

Create an end-to-end test script for the application. Specifically:

1. Create a script `test_integration.py` that:
   - Tests the full flow from drawing to prediction to database logging
   - Verifies that all components interact correctly
   - Checks that error cases are handled properly

2. Include tests for:
   - Model loading and prediction
   - Database connections and queries
   - UI flow (can be manual instructions if automated testing is difficult)

3. Add instructions for running the integration test

4. Update the Makefile to include a command for running the integration test

Focus on testing integration points between components rather than detailed unit tests.
9.2 Deployment Instructions

 Task - Prompt:

Create detailed deployment instructions for the Hetzner server. Specifically:

1. Update DEPLOYMENT.md with:
   - Server requirements (CPU, RAM, disk space)
   - Steps to install Docker and Docker Compose
   - Instructions for setting up environment variables
   - Commands to build and start the application
   - Troubleshooting common deployment issues

2. Add a section on security considerations, including:
   - Proper firewall configuration
   - Using HTTPS with SSL/TLS
   - Securing the PostgreSQL database

3. Include monitoring and maintenance instructions:
   - How to check logs
   - How to update the application
   - How to back up the database

Make the instructions detailed enough for someone with basic server administration knowledge to follow.
9.3 Final Quality Check

 Task - Prompt:

Perform a final quality check of the entire codebase. Specifically:

1. Run all code quality tools:
   - black for formatting
   - flake8 for linting
   - mypy for type checking
   - isort for import sorting

2. Fix any remaining issues

3. Run all tests (unit, integration) to ensure they pass

4. Check for any remaining TODOs or placeholder code

5. Verify that all documentation is up-to-date

Document your findings and any final changes made to ensure code quality.
Output Expectations

Fully functional MNIST Digit Classifier web application
Comprehensive test suite
Deployment-ready Docker containers
Detailed documentation
Clean, maintainable codebase following best practices