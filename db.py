#!/usr/bin/env python3
# db.py

"""
Database interaction module for MNIST Digit Classifier.

This module provides functionality for connecting to a PostgreSQL database,
logging predictions, and retrieving prediction history.
"""

# Import necessary libraries
import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, Union

import psycopg2
from psycopg2 import pool
from psycopg2.extras import DictCursor

from utils import load_environment_variables

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create a connection pool (initialized in get_connection_pool())
connection_pool = None


def get_connection_pool(
    min_connections: int = 1, max_connections: int = 10
) -> pool.ThreadedConnectionPool:
    """Get or initialize a database connection pool.
    
    This function initializes a connection pool if it doesn't exist,
    or returns the existing one. Using a connection pool improves
    performance by reusing connections instead of creating new ones
    for each database operation.
    
    Args:
        min_connections: Minimum number of connections in the pool
        max_connections: Maximum number of connections in the pool
        
    Returns:
        A ThreadedConnectionPool instance
        
    Raises:
        Exception: If connection to the database fails
    """
    global connection_pool
    
    if connection_pool is None:
        try:
            # Load environment variables
            env_vars = load_environment_variables()
            
            # Create connection pool
            connection_pool = pool.ThreadedConnectionPool(
                min_connections,
                max_connections,
                host=env_vars["db_host"],
                port=env_vars["db_port"],
                dbname=env_vars["db_name"],
                user=env_vars["db_user"],
                password=env_vars["db_password"],
            )
            
            logger.info(
                f"Connection pool created with {min_connections} to {max_connections} connections"
            )
            
        except Exception as e:
            logger.error(f"Error creating connection pool: {str(e)}")
            raise
    
    return connection_pool


@contextmanager
def get_db_connection():
    """Get a connection from the connection pool.
    
    This function is a context manager that gets a connection from the pool,
    yields it for use, and then returns it to the pool when done. This ensures
    connections are properly returned to the pool even if exceptions occur.
    
    Yields:
        A database connection from the pool
        
    Raises:
        Exception: If getting a connection from the pool fails
    """
    conn = None
    try:
        # Get connection from pool
        conn = get_connection_pool().getconn()
        conn.autocommit = False  # Ensure explicit transaction management
        yield conn
    except Exception as e:
        logger.error(f"Error getting database connection: {str(e)}")
        raise
    finally:
        # Return connection to pool if it exists
        if conn is not None:
            get_connection_pool().putconn(conn)


def close_connection(conn: Any) -> None:
    """Close a database connection explicitly.
    
    This function is deprecated in favor of using the context manager
    approach with get_db_connection(), which automatically returns
    connections to the pool. It's included for backward compatibility.
    
    Args:
        conn: The database connection to close
    """
    try:
        logger.warning(
            "close_connection() is deprecated. "
            "Use the get_db_connection() context manager instead."
        )
        if conn is not None:
            get_connection_pool().putconn(conn)
    except Exception as e:
        logger.error(f"Error closing database connection: {str(e)}")


def close_all_connections() -> None:
    """Close all connections in the pool and destroy the pool.
    
    This function should be called when shutting down the application
    to ensure all database connections are properly closed.
    """
    global connection_pool
    
    if connection_pool is not None:
        try:
            connection_pool.closeall()
            connection_pool = None
            logger.info("All database connections closed")
        except Exception as e:
            logger.error(f"Error closing all database connections: {str(e)}")


def create_tables() -> bool:
    """Create necessary tables if they don't exist.
    
    This function creates the 'predictions' table if it doesn't exist.
    The table stores digit prediction data with the following columns:
    - id: Primary key
    - timestamp: Time when the prediction was made
    - predicted_digit: The digit predicted by the model
    - confidence: Confidence score of the prediction
    - true_digit: The correct digit (if provided by user)
    
    Returns:
        bool: True if tables were created successfully, False otherwise
    """
    try:
        # Use the context manager to get a connection
        with get_db_connection() as conn:
            # Create a cursor
            with conn.cursor() as cur:
                # Create predictions table if it doesn't exist
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS predictions (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        predicted_digit INTEGER NOT NULL,
                        confidence FLOAT NOT NULL,
                        true_digit INTEGER
                    )
                """)
                
                # Create an index on timestamp for faster history queries
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_predictions_timestamp
                    ON predictions (timestamp DESC)
                """)
                
                # Commit the transaction
                conn.commit()
                
                logger.info("Database tables created successfully")
                return True
                
    except Exception as e:
        logger.error(f"Error creating tables: {str(e)}")
        return False


def test_connection() -> bool:
    """Test the database connection.
    
    This function attempts to connect to the database and run a simple query.
    It's useful for verifying that the database connection works.
    
    Returns:
        bool: True if connection works, False otherwise
    """
    try:
        # Use the context manager to get a connection
        with get_db_connection() as conn:
            # Create a cursor
            with conn.cursor() as cur:
                # Run a simple query
                cur.execute("SELECT 1")
                result = cur.fetchone()
                
                if result and result[0] == 1:
                    logger.info("Database connection test successful")
                    return True
                else:
                    logger.error("Database connection test failed")
                    return False
                
    except Exception as e:
        logger.error(f"Error testing database connection: {str(e)}")
        return False


def log_prediction(predicted_digit, confidence, true_digit=None):
    """Log a prediction to the database."""
    # TODO: Implement prediction logging
    pass


def get_prediction_history(limit=100):
    """Retrieve prediction history from the database."""
    # TODO: Implement history retrieval
    pass


if __name__ == "__main__":
    # Test database functionality
    try:
        print("Testing database connection...")
        if test_connection():
            print("✅ Database connection successful")
        else:
            print("❌ Database connection failed")
            
        print("\nCreating tables...")
        if create_tables():
            print("✅ Tables created successfully")
        else:
            print("❌ Failed to create tables")
            
    except Exception as e:
        print(f"❌ An error occurred: {str(e)}")
    finally:
        # Close all connections
        close_all_connections()
