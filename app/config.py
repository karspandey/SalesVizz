import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Database
DATABASE_PATH = os.path.join(BASE_DIR, "data", "salesvizz.db")

# Model storage
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Data settings
ALLOWED_EXTENSIONS = ['.csv']
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

# ML settings
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# User roles and permissions
ROLES = {
    'admin': [
        'manage_users',          # Create, update, delete users
        'manage_system',         # System configuration and settings
        'upload_data',          # Upload new datasets
        'delete_data',          # Delete datasets
        'train_models',         # Train new ML models
        'manage_models',        # Modify and delete models
        'run_analysis',         # Run any type of analysis
        'generate_reports',     # Generate all types of reports
        'view_logs',           # View system logs and audit trails
        'all'                   # Full access to all features
    ],
    'data_scientist': [
        'upload_data',          # Upload new datasets
        'preprocess_data',      # Data preprocessing and feature engineering
        'train_models',         # Train and tune ML models
        'run_analysis',         # Run advanced analyses
        'generate_reports',     # Generate detailed reports
        'make_predictions',     # Make predictions using any model
        'view_history'          # View analysis and prediction history
    ],
    'business_analyst': [
        'upload_data',          # Upload new datasets
        'use_models',           # Use existing models
        'basic_analysis',       # Run pre-built analyses
        'view_reports',         # View all reports
        'generate_basic_reports', # Generate basic reports
        'make_predictions'      # Make predictions using approved models
    ],
    'viewer': [
        'view_data',           # View approved datasets
        'view_reports',        # View generated reports
        'view_analysis',       # View analysis results
        'view_dashboards'      # Access basic dashboards
    ]
}
