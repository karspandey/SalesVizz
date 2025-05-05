import pandas as pd
import numpy as np
from typing import Dict, Any
import shap

class ChurnPredictor:
    def __init__(self, model: Any, data_processor: Any):
        self.model = model
        self.data_processor = data_processor
        self.explainer = None
        
    def initialize_explainer(self, X: pd.DataFrame):
        """Initialize SHAP explainer"""
        self.explainer = shap.Explainer(self.model, X)
    
    def predict(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Make predictions with explanation"""
        try:
            # Preprocess the data
            processed_data = self.data_processor.preprocess_data(data)
            
            # Make prediction
            prediction_proba = self.model.predict_proba(processed_data)
            prediction = self.model.predict(processed_data)
            
            # Safely convert prediction values
            try:
                single_prediction = int(prediction[0])
                single_proba = float(prediction_proba[0][1])
            except (IndexError, TypeError):
                raise Exception("Invalid prediction format")
            
            # Calculate confidence score
            confidence_score = float(max(prediction_proba[0]))
            
            # Get feature importance based on model type
            feature_importance = {}
            try:
                # Try using built-in feature importance first
                if hasattr(self.model, "feature_importances_"):
                    # For Random Forest, XGBoost, etc.
                    importances = self.model.feature_importances_
                    for idx, col in enumerate(processed_data.columns):
                        feature_importance[col] = float(importances[idx])
                
                elif hasattr(self.model, "coef_"):
                    # For Logistic Regression
                    importances = np.abs(self.model.coef_[0])
                    for idx, col in enumerate(processed_data.columns):
                        feature_importance[col] = float(importances[idx])
                
                else:
                    # Use SHAP values as fallback
                    if self.explainer is None:
                        self.initialize_explainer(processed_data)
                    shap_values = self.explainer(processed_data)
                    
                    for idx, col in enumerate(processed_data.columns):
                        if isinstance(shap_values.values, np.ndarray):
                            importance_value = float(np.abs(shap_values.values[0][idx]))
                        else:
                            importance_value = float(np.abs(shap_values.values[0][idx].numpy()))
                        feature_importance[col] = importance_value
                
                # Normalize feature importance values to 0-1 scale
                max_importance = max(feature_importance.values())
                if max_importance > 0:
                    feature_importance = {k: v/max_importance for k, v in feature_importance.items()}
                
            except Exception as imp_error:
                print(f"Warning: Feature importance calculation failed: {str(imp_error)}")
                for col in processed_data.columns:
                    feature_importance[col] = 0.0
            
            # Sort features by importance
            sorted_features = dict(sorted(feature_importance.items(), 
                                       key=lambda x: abs(x[1]), 
                                       reverse=True))
            
            return {
                'prediction': bool(single_prediction),
                'churn_probability': single_proba,
                'confidence_score': confidence_score,
                'feature_importance': sorted_features
            }
            
        except Exception as e:
            raise Exception(f"Prediction error: {str(e)}") 