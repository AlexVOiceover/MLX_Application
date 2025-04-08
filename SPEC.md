# Functional Specification

## For MNIST Digit Classifier

Version 0.1  
Prepared by Alexander  
Date: April 8, 2025

## Revision History

| Name       | Date       | Reason for Changes | Version |
|------------|------------|--------------------|---------|
| Alexander | 08/04/2025 | Initial Draft      | 0.1     |

## 1. Introduction

### 1.1 Document Purpose

This document provides a comprehensive overview of the functional requirements and core objectives for the MNIST Digit Classifier project. It serves as a guide for developers and stakeholders to understand the system's intended functionality and design considerations.

### 1.2 Product Scope

The MNIST Digit Classifier is a web-based application designed to recognize and classify handwritten digits using a trained PyTorch model. Users can draw digits directly on the interface, receive predictions with confidence scores, provide feedback on the accuracy, and have this data logged for further analysis.

### 1.3 Definitions, Acronyms, and Abbreviations

- **MNIST**: Modified National Institute of Standards and Technology
- **PyTorch**: An open-source machine learning library
- **Streamlit**: An open-source app framework for Machine Learning and Data Science projects
- **Docker**: A platform for developing, shipping, and running applications in containers
- **PostgreSQL**: An open-source relational database management system
- **CNN**: Convolutional Neural Network - a class of deep neural networks commonly used for image analysis

### 1.4 References

- MNIST Dataset: [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)
- PyTorch Documentation: [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)
- Streamlit Documentation: [https://docs.streamlit.io/](https://docs.streamlit.io/)
- Docker Documentation: [https://docs.docker.com/](https://docs.docker.com/)
- PostgreSQL Documentation: [https://www.postgresql.org/docs/](https://www.postgresql.org/docs/)

### 1.5 Document Overview

This document is structured to provide an in-depth understanding of the project's functional requirements, architecture, data handling, error handling strategies, and testing plans. It is intended to guide the development process and ensure all stakeholders have a clear understanding of the system's design and functionality.

## 2. Product Overview

### 2.1 Product Perspective

This is a standalone application developed for educational purposes, focusing on the end-to-end process of building, training, deploying, and interacting with a machine learning model.

### 2.2 Product Functions

- **Model Training**: Train a PyTorch model on the MNIST dataset to classify handwritten digits.
- **User Interface**: Provide a Streamlit-based web interface for users to draw digits and receive predictions.
- **Prediction Display**: Show the predicted digit along with the model's confidence score.
- **Feedback Collection**: Allow users to input the correct digit for logging purposes.
- **Data Logging**: Store each prediction, confidence score, user feedback, and timestamp in a PostgreSQL database.
- **Containerization**: Use Docker to containerize the application components.
- **Deployment**: Deploy the containerized application on a Hetzner server.

### 2.3 Product Constraints

- **Deployment Environment**: The application will be deployed on a Hetzner server.
- **Resource Limitations**: The application should be optimized to run efficiently within the resource constraints of the chosen deployment environment.

### 2.4 User Classes and Characteristics

- **End Users**: Individuals interacting with the web interface to draw digits and receive predictions.
- **Developers**: Individuals responsible for building, deploying, and maintaining the application.

### 2.5 Operating Environment

- **Server**: Hetzner cloud server with Docker installed.
- **Client**: Modern web browsers supporting HTML5 and JavaScript.

### 2.6 Design and Implementation Constraints

- **Technologies**: The application must be developed using PyTorch for the model, Streamlit for the web interface, Docker for containerization, and PostgreSQL for data storage.
- **Containerization**: All components should be containerized using Docker and orchestrated with a single `docker-compose.yml` file.

### 2.7 User Documentation

Comprehensive documentation will be provided, including setup instructions, usage guidelines, and troubleshooting tips.

### 2.8 Assumptions and Dependencies

- **Assumptions**:
  - Users have access to a modern web browser.
  - The deployment server has Docker installed and is accessible via a public IP or domain.
- **Dependencies**:
  - PyTorch
  - Streamlit
  - Docker
  - PostgreSQL

## 3. Functional Requirements

### 3.1 Model Training

- **Description**: Develop a standalone Python script (`train.py`) to train a PyTorch model on the MNIST dataset.
- **Inputs**: MNIST dataset.
- **Processing**: Train the model and save the trained weights to a file (`model.pth`).
- **Outputs**: `model.pth` file stored in the `/models/` directory of the repository.

### 3.2 User Interface

- **Description**: Create a Streamlit web application that allows users to draw digits on a canvas.
- **Features**:
  - Drawing area using a canvas input.
  - "Predict" button to trigger the prediction process.
  - Display of predicted digit and model confidence.
  - Required input field for user to provide the true digit.
  - Submit button to log the results to the database.
  - Scrollable history table showing all past predictions.

### 3.3 Logging and Database

- **Description**: Log each interaction in a PostgreSQL database.
- **Data Logged**:
  - Timestamp of submission
  - Predicted digit
  - Confidence score
  - User-provided true label
- **Implementation**: A setup script (`setup_db.sql`) should be included to initialize the database schema.

### 3.4 Docker and Deployment

- **Containerization**:
  - One Docker Compose file (`docker-compose.yml`) to coordinate:
    - PyTorch model service (can be integrated in the Streamlit app)
    - Streamlit app container
    - PostgreSQL database container
- **Deployment**:
  - Targeted deployment on Hetzner cloud instances.
  - Instructions for deployment included in `README.md`.

## 4. Non-Functional Requirements

- **Performance**: Fast response time for predictions (<1 second ideally).
- **Reliability**: Graceful error handling and stability under user interaction.
- **Security**: Application will be deployed in a secure Docker environment; PostgreSQL credentials stored securely.
- **Scalability**: Not required for MVP; single-user demo environment is sufficient.

## 5. Error Handling

- Display clear error messages if:
  - Model is not loaded.
  - Database connection fails.
  - User submits form without providing the true label.

<!-- ## 6. Testing Plan

- **Unit Testing**:
  - Model training pipeline
  - Prediction function
- **Integration Testing**:
  - End-to-end test of prediction and logging
- **Manual UI Testing**:
  - Drawing input
  - Display of predictions
  - Validation of required fields
- **Database Testing**:
  - Validate that records are correctly inserted with each submission -->
