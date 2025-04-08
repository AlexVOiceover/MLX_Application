# Architecture Decision Record: MNIST Digit Classifier

## Status
Accepted

## Context
This project involves building, containerizing, and deploying a web application for handwritten digit recognition using a PyTorch model trained on the MNIST dataset. The system includes a Streamlit web app, a PostgreSQL database for logging predictions, and containerized deployment using Docker and Docker Compose. The goal is to keep the architecture simple, maintainable, and easy to deploy.

## Decision
The system architecture will follow the principles of modularity and simplicity (KISS), structured as follows:

### Model Loading
- The trained PyTorch model (`model.pth`) will be loaded **once** at the global scope when the Streamlit app starts.
- This minimizes load time and system resources, avoiding repeated disk access.

### Preprocessing Logic
- Image preprocessing (resizing, grayscale conversion, normalization) will be placed in a dedicated `utils.py` module.
- This separation improves readability and testability, and avoids cluttering the main app logic.

### Database Logic
- All interactions with the PostgreSQL database (e.g., insert, fetch) will be managed in a `db.py` module.
- This keeps the app logic clean and supports future extensions like retries or error logging.

### History Display
- The prediction history will be displayed using Streamlit’s built-in scrollable table (`st.dataframe`).
- All predictions will be shown to the user, without pagination, as performance concerns are negligible at this scale.

### Config and Secrets
- Database credentials and environment configuration will be handled via a `.env` file.
- These values will be loaded into the app using standard environment variable access (e.g., `os.getenv`).
- The `.env` values will be referenced inside `docker-compose.yml` to avoid hardcoding sensitive information.

### Schema Setup
- On first run, `db.py` will execute a SQL statement to create the predictions table **if it does not exist**.
- This removes the need for a separate SQL setup script, keeping deployment frictionless.

### Containerization
- A single `docker-compose.yml` file will define the services:
  - **streamlit_app**: hosts the Streamlit interface and loads the model.
  - **db**: runs the PostgreSQL instance with mounted volume for persistence.
- This structure allows full deployment with one command: `docker-compose up`.

## Consequences
- Simplicity: The architecture is minimal and focused, aligned with the learning and demo goals of the project.
- Modularity: Each concern (model, preprocessing, database, UI) is cleanly separated into modules.
- Maintainability: Developers can easily adjust, replace, or test individual components.
- Scalability: While not built for horizontal scaling, the modular design allows for future extension (e.g., separating model inference into a microservice).
- Portability: Using Docker and `.env` files ensures the app can be deployed anywhere Docker runs, with clear, versioned configuration.

## Related Decisions
- The application is intended for deployment on a Hetzner server.
- Streamlit is chosen as the UI framework due to ease of setup and integration with Python.
- PyTorch is the selected framework for training and inference on the MNIST dataset.
