# Technology Stack: MNIST Digit Classifier

## Programming Language
- **Python**: 3.10+

## Machine Learning Framework
- **PyTorch**: 2.2.x
- **Torchvision**: 0.17.x

## User Interface
- **Streamlit**: 1.32.x

## Dependency Management
- **pip**: 23.x
- **requirements.txt**: used to list all dependencies
- **venv**: Python's built-in virtual environment tool is used to isolate dependencies during development

## Database
- **PostgreSQL**: 15
- **psycopg2** (Python PostgreSQL adapter): 2.9.x

## Image Processing
- **Pillow**: 10.x
- **NumPy**: 1.26.x

## Environment & Configuration
- **python-dotenv**: 1.0.x (for loading `.env` config files)

## Containerization & Deployment
- **Docker**: 24.x
- **Docker Compose**: 2.24.x
- Target deployment server: **Hetzner Cloud VPS** (Ubuntu 22.04 LTS)

<!-- ## Testing
- **pytest**: 8.x (optional, for unit and integration testing) -->

## Additional Tools
- **Git**: version control
- **Make**: optional for task automation (e.g. `make up`, `make train`, etc.)

---

## Notes
- Versions listed are tested and known to work reliably with this setup.
- Developers must use a Python virtual environment to isolate dependencies:
  ```bash
  python -m venv venv
  source venv/bin/activate
  ```
- A `requirements.txt` file will be provided to mirror this stack.

---
This stack provides a lightweight, reproducible environment suitable for both local development and deployment on a Hetzner server.

