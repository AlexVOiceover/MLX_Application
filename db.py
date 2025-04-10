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


def log_prediction(predicted_digit: int, confidence: float, true_digit: Optional[int] = None) -> Optional[int]:
    """Log a prediction to the database.
    
    This function inserts a new record into the predictions table with the
    digit prediction data and current timestamp.
    
    Args:
        predicted_digit: The digit predicted by the model (0-9)
        confidence: The confidence score of the prediction (0.0-1.0)
        true_digit: The actual digit provided by user feedback (optional)
    
    Returns:
        Optional[int]: ID of the inserted record, or None if insertion failed
    
    Raises:
        ValueError: If predicted_digit is not between 0-9 or confidence is not between 0-1
    """
    # Input validation
    if not (0 <= predicted_digit <= 9):
        raise ValueError("predicted_digit must be between 0 and 9")
    
    if not (0.0 <= confidence <= 1.0):
        raise ValueError("confidence must be between 0.0 and 1.0")
    
    if true_digit is not None and not (0 <= true_digit <= 9):
        raise ValueError("true_digit must be between 0 and 9")
    
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            # Insert the prediction data using parameterized query
            cur.execute(
                """
                INSERT INTO predictions 
                (predicted_digit, confidence, true_digit) 
                VALUES (%s, %s, %s)
                RETURNING id
                """,
                (predicted_digit, confidence, true_digit)
            )
            
            # Get the ID of the inserted record
            result = cur.fetchone()
            record_id = result['id'] if result else None
            
            # Commit the transaction
            conn.commit()
            
            print(f"Prediction logged with ID: {record_id}")
            return record_id
            
    except Exception as e:
        print(f"Error logging prediction: {e}")
        if conn:
            conn.rollback()  # Rollback transaction on error
        return None
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
    try:
        # Test database functionality
        print("\n1. Creating tables...")
        create_tables()
        print("✅ Tables created successfully")
        
        # Test logging a prediction
        print("\n2. Logging a test prediction...")
        predicted_digit = 5
        confidence = 0.95
        record_id = log_prediction(predicted_digit, confidence)
        if record_id:
            print(f"✅ Prediction logged successfully with ID: {record_id}")
        else:
            print("❌ Failed to log prediction")
        
        # Test logging a prediction with user feedback
        print("\n3. Logging a test prediction with user feedback...")
        predicted_digit = 7
        confidence = 0.85
        true_digit = 7  # User confirms the prediction was correct
        record_id = log_prediction(predicted_digit, confidence, true_digit)
        if record_id:
            print(f"✅ Prediction with feedback logged successfully with ID: {record_id}")
        else:
            print("❌ Failed to log prediction with feedback")
        
        # Test retrieving prediction history
        print("\n4. Retrieving prediction history...")
        history = get_prediction_history(limit=5)
        print(f"Retrieved {len(history)} records from prediction history")
        if history:
            print("\nMost recent predictions:")
            for record in history:
                feedback = f"(User said: {record['true_digit']})" if record['true_digit'] is not None else "(No feedback)"
                print(f"  ID: {record['id']} | Predicted: {record['predicted_digit']} | "
                      f"Confidence: {record['confidence']:.2f} | {feedback}")
        
    except Exception as e:
        print(f"❌ An error occurred during testing: {e}")
