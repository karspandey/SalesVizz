import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
import logging

class DataValidator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def validate_csv_file(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Validate uploaded CSV file"""
        try:
            # Check if dataframe is empty
            if df.empty:
                return False, "The uploaded file is empty"
            
            # Check minimum number of rows
            if len(df) < 10:
                return False, "Dataset must contain at least 10 rows"
            
            # Check for too many columns
            if len(df.columns) > 100:
                return False, "Dataset contains too many columns (max 100)"
            
            # Check for duplicate column names
            if len(df.columns) != len(set(df.columns)):
                return False, "Dataset contains duplicate column names"
            
            return True, "Validation successful"
            
        except Exception as e:
            self.logger.error(f"Error validating CSV: {str(e)}")
            return False, f"Error validating file: {str(e)}"
    
    def validate_churn_column(self, df: pd.DataFrame, churn_column: str) -> Tuple[bool, str]:
        """Validate the selected churn column"""
        try:
            # Check if column exists
            if churn_column not in df.columns:
                return False, "Selected churn column not found in dataset"
            
            # Check if binary
            unique_values = df[churn_column].unique()
            if not set(unique_values).issubset({0, 1}):
                return False, "Churn column must contain only binary values (0/1)"
            
            # Check for missing values
            if df[churn_column].isnull().any():
                return False, "Churn column contains missing values"
            
            return True, "Churn column validation successful"
            
        except Exception as e:
            self.logger.error(f"Error validating churn column: {str(e)}")
            return False, f"Error validating churn column: {str(e)}"
    
    def validate_input_data(self, input_data: Dict) -> Tuple[bool, str]:
        """Validate manual input data for predictions"""
        try:
            # Check if all fields are filled
            empty_fields = [k for k, v in input_data.items() if not str(v).strip()]
            if empty_fields:
                return False, f"Please fill in all fields: {', '.join(empty_fields)}"
            
            # Check numeric values
            for key, value in input_data.items():
                try:
                    float(value)
                except ValueError:
                    if not isinstance(value, str):
                        return False, f"Invalid value for {key}: must be numeric or text"
            
            return True, "Input validation successful"
            
        except Exception as e:
            self.logger.error(f"Error validating input data: {str(e)}")
            return False, f"Error validating input: {str(e)}" 