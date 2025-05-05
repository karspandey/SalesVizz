import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Any
import json
from datetime import datetime
import os
import streamlit as st

class ReportGenerator:
    def __init__(self, analytics: Any, predictor: Any):
        self.analytics = analytics
        self.predictor = predictor
        
    def generate_report(self, churn_column: str, model_name: str) -> Dict:
        """Generate a comprehensive report"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        report = {
            'generated_at': timestamp,
            'model_info': {
                'model_type': model_name,
                'features_used': self.analytics.numerical_columns + self.analytics.categorical_columns
            },
            'dataset_statistics': self.analytics.get_basic_stats(),
            'churn_analysis': self.analytics.analyze_churn_rate(churn_column)
        }
        
        return report
    
    def save_report(self, report: Dict, username: str, model_name: str) -> str:
        """Save report with improved naming and organization"""
        # Create reports directory if it doesn't exist
        os.makedirs('reports', exist_ok=True)
        
        # Generate filename with more information
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"report_{username}_{model_name}_{timestamp}.json"
        filepath = os.path.join('reports', filename)
        
        # Save report
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=4)
            
        return filepath
    
    def generate_feature_impact_report(self, sample_data: pd.DataFrame) -> Dict:
        """Generate a report on feature importance and impact"""
        # Get prediction and feature importance for a sample
        prediction_result = self.predictor.predict(sample_data)
        
        report = {
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'feature_importance': prediction_result['feature_importance'],
            'prediction_confidence': prediction_result['confidence_score']
        }
        
        return report
    
    def generate_full_report(self, churn_column: str, sample_data: pd.DataFrame = None) -> Dict:
        """Generate a comprehensive report including all analyses"""
        report = {
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'summary': self.generate_summary_report(churn_column)
        }
        
        if sample_data is not None:
            report['feature_impact'] = self.generate_feature_impact_report(sample_data)
        
        return report 