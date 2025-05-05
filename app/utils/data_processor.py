import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
from sklearn.preprocessing import StandardScaler, LabelEncoder
import json

class DataProcessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.categorical_columns = []
        self.numerical_columns = []
        
    def identify_column_types(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Identify numerical and categorical columns"""
        numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        return numerical_columns, categorical_columns
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        # For numerical columns, fill with median
        for col in self.numerical_columns:
            df[col] = df[col].fillna(df[col].median())
        
        # For categorical columns, fill with mode
        for col in self.categorical_columns:
            df[col] = df[col].fillna(df[col].mode()[0])
        
        return df
    
    def handle_outliers(self, df: pd.DataFrame, threshold: float = 3) -> pd.DataFrame:
        """Handle outliers using Z-score method"""
        for col in self.numerical_columns:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            df[col] = df[col].mask(z_scores > threshold, df[col].median())
        return df
    
    def encode_categorical_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables using Label Encoding with handling for unseen categories"""
        df_encoded = df.copy()
        
        for col in self.categorical_columns:
            if col not in self.label_encoders:
                # First time encoding - fit and transform
                self.label_encoders[col] = LabelEncoder()
                df_encoded[col] = self.label_encoders[col].fit_transform(df_encoded[col])
            else:
                # Handle unseen categories
                known_categories = set(self.label_encoders[col].classes_)
                # Replace unseen categories with the most frequent category
                most_frequent = self.label_encoders[col].transform([self.label_encoders[col].classes_[0]])[0]
                df_encoded[col] = df_encoded[col].apply(
                    lambda x: x if x in known_categories else self.label_encoders[col].classes_[0]
                )
                df_encoded[col] = self.label_encoders[col].transform(df_encoded[col])
        
        return df_encoded
    
    def scale_numerical_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale numerical variables using StandardScaler"""
        df_scaled = df.copy()
        df_scaled[self.numerical_columns] = self.scaler.fit_transform(df_scaled[self.numerical_columns])
        return df_scaled
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Complete preprocessing pipeline"""
        # Make a copy to avoid modifying the original dataframe
        df_processed = df.copy()
        
        # Identify column types
        self.numerical_columns, self.categorical_columns = self.identify_column_types(df_processed)
        
        # Handle missing values
        df_processed = self.handle_missing_values(df_processed)
        
        # Handle outliers
        df_processed = self.handle_outliers(df_processed)
        
        # Encode categorical variables
        df_processed = self.encode_categorical_variables(df_processed)
        
        # Scale numerical variables
        df_processed = self.scale_numerical_variables(df_processed)
        
        return df_processed
    
    def save_preprocessing_state(self, filepath: str):
        """Save preprocessing state for later use"""
        state = {
            'label_encoders': {
                col: {
                    'classes': self.label_encoders[col].classes_.tolist()
                } for col in self.label_encoders
            },
            'scaler': {
                'mean': self.scaler.mean_.tolist(),
                'scale': self.scaler.scale_.tolist()
            },
            'categorical_columns': self.categorical_columns,
            'numerical_columns': self.numerical_columns
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f)
    
    def load_preprocessing_state(self, filepath: str):
        """Load preprocessing state"""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        # Restore label encoders
        self.label_encoders = {}
        for col, encoder_state in state['label_encoders'].items():
            le = LabelEncoder()
            le.classes_ = np.array(encoder_state['classes'])
            self.label_encoders[col] = le
        
        # Restore scaler
        self.scaler = StandardScaler()
        self.scaler.mean_ = np.array(state['scaler']['mean'])
        self.scaler.scale_ = np.array(state['scaler']['scale'])
        
        # Restore column lists
        self.categorical_columns = state['categorical_columns']
        self.numerical_columns = state['numerical_columns'] 