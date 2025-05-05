import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class ChurnAnalytics:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.numerical_columns = []
        self.categorical_columns = []
        self._identify_column_types()
    
    def _identify_column_types(self):
        """Identify numerical and categorical columns"""
        self.numerical_columns = self.data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_columns = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
    
    def get_basic_stats(self) -> Dict:
        """Get basic statistics about the dataset"""
        return {
            'total_customers': len(self.data),
            'total_features': len(self.data.columns),
            'missing_values': self.data.isnull().sum().to_dict(),
            'numerical_features': len(self.numerical_columns),
            'categorical_features': len(self.categorical_columns)
        }
    
    def analyze_churn_rate(self, churn_column: str) -> Dict:
        """Analyze overall churn rate"""
        churn_mapping = self._get_churn_mapping(churn_column)
        binary_churn = self.data[churn_column].map(churn_mapping)
        churn_counts = binary_churn.value_counts()
        total_customers = len(self.data)
        
        return {
            'churn_rate': float(churn_counts.get(1, 0) / total_customers),
            'retained_rate': float(churn_counts.get(0, 0) / total_customers),
            'churned_customers': int(churn_counts.get(1, 0)),
            'retained_customers': int(churn_counts.get(0, 0))
        }
    
    def create_churn_distribution_plot(self, churn_column: str):
        """Create churn distribution visualization"""
        # Map churn values to binary
        churn_mapping = self._get_churn_mapping(churn_column)
        churn_counts = self.data[churn_column].map(churn_mapping).value_counts()
        
        fig = px.pie(
            values=churn_counts.values,
            names=['Retained', 'Churned'],
            title='Customer Churn Distribution'
        )
        return fig
    
    def _get_churn_mapping(self, churn_column: str) -> dict:
        """Get mapping for churn values to binary format"""
        unique_values = self.data[churn_column].unique()
        if len(unique_values) != 2:
            raise ValueError(f"Churn column must have exactly 2 unique values. Found: {unique_values}")
        
        # Try to automatically determine which value represents churn
        # First try numeric values
        if pd.api.types.is_numeric_dtype(self.data[churn_column]):
            if set(unique_values) == {0, 1}:
                return {0: 0, 1: 1}
            else:
                # Assume larger value represents churn
                return {min(unique_values): 0, max(unique_values): 1}
        
        # For string/categorical values
        # Common churn indicators
        churn_indicators = ['yes', 'true', 'churn', '1', 'churned', 'left']
        retain_indicators = ['no', 'false', 'retain', '0', 'retained', 'stayed']
        
        # Convert to lowercase strings for comparison
        values_lower = [str(v).lower() for v in unique_values]
        
        # Try to match with common indicators
        churn_value = None
        for val, val_lower in zip(unique_values, values_lower):
            if val_lower in churn_indicators:
                churn_value = val
                break
        
        if churn_value is None:
            # If no match found, take alphabetically larger value as churn
            churn_value = max(unique_values)
        
        retain_value = [v for v in unique_values if v != churn_value][0]
        return {retain_value: 0, churn_value: 1}
    
    def analyze_numerical_features(self, churn_column: str) -> Dict[str, go.Figure]:
        """Analyze numerical features distribution by churn status"""
        plots = {}
        churn_mapping = self._get_churn_mapping(churn_column)
        
        for col in self.numerical_columns:
            if col != churn_column:
                # Map churn values to binary before plotting
                binary_churn = self.data[churn_column].map(churn_mapping)
                
                # Create histogram data with fixed bins for both groups
                min_val = self.data[col].min()
                max_val = self.data[col].max()
                nbins = 30
                bin_size = (max_val - min_val) / nbins
                
                # Create figure with secondary y-axis
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                # Add histogram for churned customers
                churned_data = self.data[binary_churn == 1][col]
                retained_data = self.data[binary_churn == 0][col]
                
                fig.add_trace(
                    go.Histogram(
                        x=churned_data,
                        name='Churned',
                        nbinsx=nbins,
                        marker_color='#ff7f7f',
                        opacity=0.75,
                        xbins=dict(
                            start=min_val,
                            end=max_val,
                            size=bin_size
                        )
                    )
                )
                
                # Add histogram for retained customers
                fig.add_trace(
                    go.Histogram(
                        x=retained_data,
                        name='Retained',
                        nbinsx=nbins,
                        marker_color='#7fb3ff',
                        opacity=0.75,
                        xbins=dict(
                            start=min_val,
                            end=max_val,
                            size=bin_size
                        )
                    )
                )
                
                # Update layout
                fig.update_layout(
                    title=f'Distribution of {col} by Churn Status',
                    barmode='group',  # Change to group mode
                    bargap=0.1,       # Gap between bars in the same group
                    bargroupgap=0.1,  # Gap between bar groups
                    xaxis_title=col,
                    yaxis_title='Count',
                    showlegend=True,
                    template='plotly_white',
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="right",
                        x=0.99
                    ),
                    modebar=dict(
                        orientation='v',
                        bgcolor='rgba(255, 255, 255, 0.7)',
                        color='rgba(0, 0, 0, 0.5)',
                        activecolor='rgba(68, 68, 68, 1)',
                        remove=['lasso2d', 'select2d']
                    )
                )
                
                plots[col] = fig
        
        return plots
    
    def analyze_categorical_features(self, churn_column: str) -> Dict[str, go.Figure]:
        """Analyze categorical features relationship with churn"""
        plots = {}
        churn_mapping = self._get_churn_mapping(churn_column)
        
        for col in self.categorical_columns:
            if col != churn_column:
                # Map churn values to binary before analysis
                binary_churn = self.data[churn_column].map(churn_mapping)
                
                # Calculate churn counts for each category
                churn_by_category = pd.crosstab(
                    self.data[col], 
                    binary_churn,
                    normalize='index'  # Calculate percentages
                )[1]  # Get only churn rate (1's)
                
                # Sort by churn rate
                churn_by_category = churn_by_category.sort_values(ascending=False)
                
                # Create bar plot
                fig = go.Figure(go.Bar(
                    x=churn_by_category.index,
                    y=churn_by_category.values,
                    text=[f'{v:.1%}' for v in churn_by_category.values],
                    textposition='auto'
                ))
                
                fig.update_layout(
                    title=f'Churn Rate by {col}',
                    xaxis_title=col,
                    yaxis_title='Churn Rate',
                    yaxis_tickformat=',.0%'
                )
                
                plots[col] = fig
        
        return plots
    
    def generate_correlation_matrix(self, churn_column: str):
        """Generate correlation matrix for numerical features"""
        # First convert churn column to binary if it's categorical
        churn_mapping = self._get_churn_mapping(churn_column)
        binary_churn = self.data[churn_column].map(churn_mapping)
        
        # Create a copy of numerical data
        numerical_data = self.data[self.numerical_columns].copy()
        
        # Add binary churn column
        if churn_column not in self.numerical_columns:
            numerical_data['Churn'] = binary_churn
        else:
            numerical_data[churn_column] = binary_churn
        
        # Calculate correlation matrix
        correlation_matrix = numerical_data.corr()
        
        # Create heatmap
        fig = px.imshow(
            correlation_matrix,
            title='Feature Correlation Matrix',
            color_continuous_scale='RdBu',
            aspect='auto',
            labels=dict(color="Correlation")
        )
        
        # Update layout for better readability
        fig.update_layout(
            template='plotly_white',
            xaxis_title="Features",
            yaxis_title="Features",
            modebar=dict(
                orientation='v',
                bgcolor='rgba(255, 255, 255, 0.7)',
                color='rgba(0, 0, 0, 0.5)',
                activecolor='rgba(68, 68, 68, 1)',
                remove=['lasso2d', 'select2d']
            )
        )
        
        # Rotate x-axis labels for better readability
        fig.update_xaxes(tickangle=45)
        
        return fig 