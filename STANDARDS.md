# Standards and Best Practices

This document outlines the development and operational standards for the MNIST Digit Classifier project. It ensures a shared understanding of expectations and practices to maintain code quality, clarity, and operational consistency.

---

## 1. Code Quality

- **Formatting**: Use `black` (Python code formatter) and `isort` for import sorting.
- **Linting**: Run `flake8` or `ruff` to catch syntax and style issues.
- **Naming**: Follow PEP8 naming conventions:
  - Functions & variables: `snake_case`
  - Classes: `PascalCase`
- **Typing**: Use type hints wherever practical. Validate with `mypy`.
- **Comments**:
  - Prefer clarity in code over excessive comments.
  - Use docstrings for all public functions and classes.

---

## 2. File Structure

Maintain the following modular project layout:

```
project-root/
├── app.py            # Streamlit UI logic
├── db.py             # Database interactions
├── utils.py          # Image preprocessing
├── train.py          # Model training script
├── models/
│   └── model.pth     # Trained model file
├── .env              # Environment variables
├── requirements.txt  # Python dependencies
├── docker-compose.yml
└── Dockerfile
```

---

## 3. Git and Version Control

- **Commits**:
  - Use clear, concise messages: `feat:`, `fix:`, `chore:`, `docs:`.
  - Group related changes in a single commit.
- **Branches**:
  - Use feature branches named after their purpose, e.g., `feature/ui-canvas`.

---

## 4. Dependencies

- Use `requirements.txt` to declare all project dependencies with versions pinned.
- Avoid using `latest` as a version specifier.

---

## 5. Environment Variables

- Store sensitive configuration (e.g., DB credentials) in a `.env` file.
- Access using `os.getenv()` or `dotenv`.
- Never commit `.env` files to source control.

---

## 6. Docker and Deployment

- Use Docker and `docker-compose.yml` for development and production parity.
- Each service must be defined clearly with relevant ports and environment variables.
- Images should be lightweight (start from `python:3.10-slim` base).

---

## 7. Logging & Error Handling

- Use `try/except` blocks for database operations and predictions.
- Log errors clearly using Python’s `logging` module.
- Avoid exposing internal errors in the UI—show user-friendly messages.

---

## 8. Testing

- Where applicable, include basic tests for:
  - Image preprocessing (`utils.py`)
  - Database insertions (`db.py`)
  - Model prediction interface (`model.forward`)
- Use `pytest` and store tests in a `/tests` directory.
