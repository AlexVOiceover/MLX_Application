#!/usr/bin/env python3
# db.py

"""
Database interaction module for MNIST Digit Classifier.

This module provides functionality for connecting to a PostgreSQL database,
logging predictions, and retrieving prediction history.
"""

# Import necessary libraries
import os
from typing import Dict, List, Optional, Tuple, Union
import psycopg
from psycopg.rows import dict_row
from utils import load_environment_variables


def get_db_connection():
    """Establish a connection to the PostgreSQL database."""
    env_vars = load_environment_variables()
    
    connection_string = (
        f"host={env_vars['db_host']} "
        f"port={env_vars['db_port']} "
        f"dbname={env_vars['db_name']} "
        f"user={env_vars['db_user']} "
        f"password={env_vars['db_password']}"
    )
    
    conn = psycopg.connect(connection_string, row_factory=dict_row)
    return conn


def close_connection(conn):
    """Close the database connection."""
    if conn is not None:
        conn.close()


def create_tables():
    """Create necessary tables if they don't exist."""
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id SERIAL PRIMARY KEY,
                    predicted_digit INTEGER NOT NULL,
                    confidence FLOAT NOT NULL,
                    true_digit INTEGER,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
        conn.commit()
    except Exception as e:
        print(f"Error creating tables: {e}")
    finally:
        close_connection(conn)


def log_prediction(predicted_digit: int, confidence: float, true_digit: Optional[int] = None):
    """Log a prediction to the database.
    
    Args:
        predicted_digit: The digit predicted by the model
        confidence: The confidence score of the prediction
        true_digit: The actual digit (if provided by user feedback)
    
    Returns:
        bool: True if logged successfully, False otherwise
    """
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO predictions 
                (predicted_digit, confidence, true_digit) 
                VALUES (%s, %s, %s)
                """,
                (predicted_digit, confidence, true_digit)
            )
        conn.commit()
        return True
    except Exception as e:
        print(f"Error logging prediction: {e}")
        return False
    finally:
        close_connection(conn)


def get_prediction_history(limit: int = 100) -> List[Dict]:
    """Retrieve prediction history from the database.
    
    Args:
        limit: Maximum number of records to retrieve
    
    Returns:
        List of dictionaries containing prediction records
    """
    conn = None
    results = []
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT 
                    id, predicted_digit, confidence, true_digit, 
                    timestamp::text as timestamp
                FROM predictions 
                ORDER BY timestamp DESC 
                LIMIT %s
                """,
                (limit,)
            )
            results = cur.fetchall()
        return results
    except Exception as e:
        print(f"Error retrieving prediction history: {e}")
        return []
    finally:
        close_connection(conn)


if __name__ == "__main__":
    # Test database functionality
    create_tables()
    print("Database tables created successfully.")
