# Technology Stack: MNIST Digit Classifier

## Overview
This document outlines the full technology stack used in the MNIST Digit Classifier project, including tools, libraries, and specific version information. It ensures consistency across development, testing, and deployment environments.

---

## 🔧 Programming Language
- **Python**: 3.10+

## 🧠 Machine Learning Framework
- **PyTorch**: 2.2.x
- **Torchvision**: 0.17.x

## 🖼️ User Interface
- **Streamlit**: 1.32.x

## 📦 Dependency Management
- **pip**: 23.x
- **requirements.txt**: used to list all dependencies

## 🐘 Database
- **PostgreSQL**: 15
- **psycopg2** (Python PostgreSQL adapter): 2.9.x

## 🔄 Image Processing
- **Pillow**: 10.x
- **NumPy**: 1.26.x

## 📂 Environment & Configuration
- **python-dotenv**: 1.0.x (for loading `.env` config files)

## 🐳 Containerization & Deployment
- **Docker**: 24.x
- **Docker Compose**: 2.24.x
- Target deployment server: **Hetzner Cloud VPS** (Ubuntu 22.04 LTS)

## 🧪 Testing
- **pytest**: 8.x (optional, for unit and integration testing)

## 🧰 Additional Tools
- **Git**: version control
- **Make**: optional for task automation (e.g. `make up`, `make train`, etc.)

---

## Notes
- Versions listed are tested and known to work reliably with this setup.
- Developers should use a Python virtual environment to isolate dependencies:
  ```bash
  python -m venv venv
  source venv/bin/activate
  ```
- A `requirements.txt` file will be provided to mirror this stack.

---
This stack provides a lightweight, reproducible environment suitable for both local development and deployment on a Hetzner server.

