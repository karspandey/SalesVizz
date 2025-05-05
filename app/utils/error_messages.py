"""Error messages for the application"""

ERRORS = {
    'auth': {
        'invalid_credentials': "Invalid username or password",
        'user_exists': "Username already exists",
        'password_mismatch': "Passwords don't match",
        'unauthorized': "You don't have permission to access this feature"
    },
    'data': {
        'upload_failed': "Failed to upload file",
        'invalid_format': "Invalid file format. Please upload a CSV file",
        'empty_file': "The uploaded file is empty",
        'missing_churn': "Please select the churn column",
        'invalid_churn': "Invalid churn column. Must contain only 0 and 1"
    },
    'model': {
        'training_failed': "Model training failed",
        'no_model': "No trained model available. Please train a model first",
        'prediction_failed': "Failed to make prediction",
        'invalid_input': "Invalid input data for prediction"
    },
    'report': {
        'generation_failed': "Failed to generate report",
        'save_failed': "Failed to save report"
    }
}

SUCCESS_MESSAGES = {
    'auth': {
        'login_success': "Login successful!",
        'register_success': "Registration successful! Please login."
    },
    'data': {
        'upload_success': "Data uploaded successfully!",
        'validation_success': "Data validation successful"
    },
    'model': {
        'training_success': "Model trained successfully!",
        'prediction_success': "Prediction completed successfully"
    },
    'report': {
        'generation_success': "Report generated successfully",
        'save_success': "Report saved successfully"
    }
} 