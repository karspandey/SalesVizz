import sqlite3
import os
from pathlib import Path
from datetime import datetime

def init_db():
    """Initialize the database and create all required tables"""
    try:
        # Get the path to the data directory
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
        db_path = os.path.join(data_dir, 'salesvizz.db')
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Connect to database and create tables
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create users table if it doesn't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT NOT NULL
        )
        ''')
        
        # Models table - stores trained ML models
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                user_id INTEGER,
                model_blob BLOB,
                metrics TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Analysis History table - stores analysis results
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                analysis_type TEXT,
                results TEXT,
                dataset_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Predictions table - stores prediction history
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                model_id INTEGER,
                input_data TEXT,
                prediction TEXT,
                probability REAL,
                confidence_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id),
                FOREIGN KEY (model_id) REFERENCES models (id)
            )
        ''')
        
        # Shared Analysis table - stores shared analysis with access keys
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS shared_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_key TEXT UNIQUE NOT NULL,
                user_id INTEGER,
                title TEXT NOT NULL,
                description TEXT,
                dataset_preview TEXT,
                analysis_results TEXT,
                model_metrics TEXT,
                feature_importance TEXT,
                is_public BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                view_count INTEGER DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        conn.commit()
        print("Database initialized successfully!")
        
    except Exception as e:
        print(f"Error initializing database: {str(e)}")
        raise e
        
    finally:
        if 'conn' in locals():
            conn.close() 