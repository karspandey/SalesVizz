import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from typing import Dict, Tuple, Any
import optuna
import pickle

class ModelTrainer:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {
            'logistic_regression': LogisticRegression(
                random_state=random_state,
                max_iter=1000,
                n_jobs=-1
            ),
            'random_forest': RandomForestClassifier(
                random_state=random_state,
                n_jobs=-1,
                n_estimators=100,
                max_depth=10
            ),
            'xgboost': xgb.XGBClassifier(
                random_state=random_state,
                n_jobs=-1,
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                tree_method='hist'  # Faster training method
            )
        }
        self.best_model = None
        self.best_model_name = None
        
    def evaluate_model(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate model performance"""
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
    
    def _preprocess_target(self, y: pd.Series) -> pd.Series:
        """Convert target variable to binary format"""
        if y.dtype == bool:
            return y.astype(int)
        
        if pd.api.types.is_numeric_dtype(y):
            # If numeric, assume binary 0/1 or similar
            unique_vals = sorted(y.unique())
            if len(unique_vals) != 2:
                raise ValueError(f"Target must have exactly 2 unique values. Found: {unique_vals}")
            mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
            return y.map(mapping)
        
        # For string/categorical values
        unique_vals = y.unique()
        if len(unique_vals) != 2:
            raise ValueError(f"Target must have exactly 2 unique values. Found: {unique_vals}")
        
        # Common churn indicators
        churn_indicators = ['yes', 'true', 'churn', '1', 'churned', 'left']
        retain_indicators = ['no', 'false', 'retain', '0', 'retained', 'stayed']
        
        # Convert to lowercase for comparison
        y_lower = y.str.lower()
        
        # Try to match with common indicators
        for val in y_lower.unique():
            if val in churn_indicators:
                return (y_lower == val).astype(int)
        
        # If no match found, take alphabetically larger value as churn
        mapping = {min(unique_vals): 0, max(unique_vals): 1}
        return y.map(mapping)
    
    def optimize_hyperparameters(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series, n_trials: int = 30, fast_mode: bool = False) -> Dict:
        """Optimize model hyperparameters using Optuna"""
        def objective(trial):
            if model_name == 'logistic_regression':
                params = {
                    'C': trial.suggest_loguniform('C', 1e-3, 1e3),
                    'max_iter': 1000,
                    'n_jobs': -1,
                    'random_state': self.random_state
                }
                model = LogisticRegression(**params)
            
            elif model_name == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 15),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                    'n_jobs': -1,
                    'random_state': self.random_state
                }
                model = RandomForestClassifier(**params)
            
            elif model_name == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 12),
                    'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 0.3),
                    'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                    'tree_method': 'hist',
                    'n_jobs': -1,
                    'random_state': self.random_state
                }
                model = xgb.XGBClassifier(**params)
            
            # Use cross-validation with fewer folds in fast mode
            cv = 3 if fast_mode else 5
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
            return scores.mean()
        
        # Create study and optimize
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        return study.best_params
    
    def train_model(self, X: pd.DataFrame, y: pd.Series, model_name: str = None, fast_mode: bool = False, use_optuna: bool = False, n_trials: int = 30) -> Tuple[Any, Dict[str, float]]:
        """Train a specific model or find the best model with optional Optuna optimization"""
        # Preprocess target variable
        y_binary = self._preprocess_target(y)
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_binary, test_size=0.2, random_state=self.random_state,
            stratify=y_binary
        )
        
        if model_name is None:
            # Try all models and select the best one
            best_score = -1
            best_model = None
            best_model_name = None
            best_metrics = None
            
            for name, model in self.models.items():
                if use_optuna and not fast_mode:
                    # Use Optuna for hyperparameter optimization
                    best_params = self.optimize_hyperparameters(name, X_train, y_train, n_trials=n_trials)
                    model.set_params(**best_params)
                
                model.fit(X_train, y_train)
                metrics = self.evaluate_model(model, X_test, y_test)
                
                if metrics['roc_auc'] > best_score:
                    best_score = metrics['roc_auc']
                    best_model = model
                    best_model_name = name
                    best_metrics = metrics
            
            self.best_model = best_model
            self.best_model_name = best_model_name
            return best_model, best_metrics
        
        else:
            # Train specific model
            model = self.models[model_name]
            
            if use_optuna and not fast_mode:
                # Use Optuna for hyperparameter optimization
                best_params = self.optimize_hyperparameters(model_name, X_train, y_train, n_trials=n_trials)
                model.set_params(**best_params)
            elif fast_mode:
                # Use minimal hyperparameters for fast training
                if model_name == 'random_forest':
                    model.set_params(n_estimators=50, max_depth=6)
                elif model_name == 'xgboost':
                    model.set_params(n_estimators=50, max_depth=4)
            
            model.fit(X_train, y_train)
            metrics = self.evaluate_model(model, X_test, y_test)
            
            self.best_model = model
            self.best_model_name = model_name
            return model, metrics
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        if self.best_model is None:
            raise ValueError("No model has been trained yet")
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.best_model,
                'model_name': self.best_model_name
            }, f)
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.best_model = data['model']
            self.best_model_name = data['model_name'] 