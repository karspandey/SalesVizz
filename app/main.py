import streamlit as st
import pandas as pd
from auth.auth_handler import AuthHandler
from database.models import init_db
from utils.data_processor import DataProcessor
from ml.model_trainer import ModelTrainer
from ml.predictor import ChurnPredictor
from analytics.analytics import ChurnAnalytics
from reports.report_generator import ReportGenerator
import plotly.graph_objects as go
import os
from datetime import datetime
import sqlite3
import json
import uuid
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import numpy as np
import matplotlib.pyplot as plt
import shap

# Create data directory if it doesn't exist
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
DB_PATH = os.path.join(DATA_DIR, 'salesvizz.db')

# Create data directory if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)

# Initialize database
init_db()

# Initialize authentication
auth = AuthHandler(DB_PATH)

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'role' not in st.session_state:
    st.session_state.role = None
if 'username' not in st.session_state:
    st.session_state.username = None
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = None
if 'model_name' not in st.session_state:
    st.session_state.model_name = None
if 'churn_column' not in st.session_state:
    st.session_state.churn_column = None

def login_page():
    st.title("🔐 SalesVizz - Login")
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        
        if st.button("Login"):
            result = auth.authenticate_user(username, password)
            if result:
                user_id, role = result
                st.session_state.authenticated = True
                st.session_state.user_id = user_id
                st.session_state.role = role
                st.session_state.username = username
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid username or password")
    
    with tab2:
        new_username = st.text_input("New Username", key="reg_user")
        new_password = st.text_input("New Password", type="password", key="reg_pass")
        confirm_password = st.text_input("Confirm Password", type="password", key="reg_pass_confirm")
        
        # Role selection with descriptions
        role_descriptions = {
            'viewer': 'Access to view datasets, reports, and analysis results',
            'business_analyst': 'Access to upload data, use models, and generate basic reports',
            'data_scientist': 'Full access to data science features including model training'
        }
        
        role = st.selectbox(
            "Select Role",
            options=['viewer', 'business_analyst', 'data_scientist'],
            key="reg_role",
            help="Choose your role based on your needs"
        )
        
        # Show role description
        st.caption(role_descriptions[role])
        
        if st.button("Register"):
            if new_password != confirm_password:
                st.error("Passwords don't match")
            elif auth.register_user(new_username, new_password, role):
                st.success(f"Registration successful as {role}! Please login.")
            else:
                st.error("Username already exists")

def analyze_data(data: pd.DataFrame, churn_column: str):
    """Function to perform data analysis"""
    st.header("📊 Data Analysis")
    
    if data is None:
        st.warning("Please upload data first!")
        return
        
    # Basic data info
    st.subheader("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", len(data))
    with col2:
        st.metric("Features", len(data.columns))
    with col3:
        st.metric("Missing Values", data.isnull().sum().sum())
    
    # Data preview
    with st.expander("Data Preview", expanded=False):
        st.dataframe(data.head())
        
    # Summary statistics
    with st.expander("Summary Statistics", expanded=False):
        st.dataframe(data.describe())
    
    # Churn distribution
    st.subheader("Churn Distribution")
    churn_dist = data[churn_column].value_counts()
    fig = go.Figure(data=[go.Pie(labels=churn_dist.index, values=churn_dist.values)])
    fig.update_layout(title="Churn Distribution")
    st.plotly_chart(fig)
    
    # Feature correlations
    st.subheader("Feature Correlations")
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) > 0:
        corr_matrix = data[numeric_cols].corr()
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu'
        ))
        fig.update_layout(title="Correlation Matrix")
        st.plotly_chart(fig)
    
    # Feature distributions
    st.subheader("Feature Distributions")
    selected_feature = st.selectbox("Select feature to analyze:", data.columns)
    
    if data[selected_feature].dtype in ['int64', 'float64']:
        # Numerical feature
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=data[selected_feature], nbinsx=30))
        fig.update_layout(title=f"Distribution of {selected_feature}")
        st.plotly_chart(fig)
        
        # Box plot by churn
        fig = go.Figure()
        for churn_val in data[churn_column].unique():
            fig.add_trace(go.Box(
                y=data[data[churn_column] == churn_val][selected_feature],
                name=f"Churn = {churn_val}"
            ))
        fig.update_layout(title=f"{selected_feature} by Churn Status")
        st.plotly_chart(fig)
    else:
        # Categorical feature
        cat_dist = data[selected_feature].value_counts()
        fig = go.Figure(data=[go.Bar(x=cat_dist.index, y=cat_dist.values)])
        fig.update_layout(title=f"Distribution of {selected_feature}")
        st.plotly_chart(fig)
        
        # Stacked bar by churn
        pivot_table = pd.crosstab(data[selected_feature], data[churn_column], normalize='index')
        fig = go.Figure(data=[
            go.Bar(name=str(col), x=pivot_table.index, y=pivot_table[col])
            for col in pivot_table.columns
        ])
        fig.update_layout(
            title=f"{selected_feature} vs Churn",
            barmode='stack'
        )
        st.plotly_chart(fig)

def train_model_page():
    """Page for training ML models"""
    st.subheader("🤖 Train Model")
    
    # Check if data is uploaded
    if not hasattr(st.session_state, 'data') or st.session_state.data is None:
        st.warning("⚠️ Please upload your dataset first in the Upload Data page!")
        st.info("Go to 📁 Upload Data in the navigation menu to upload your dataset.")
        return
    
    # Check if churn column is selected
    if not hasattr(st.session_state, 'churn_column') or st.session_state.churn_column is None:
        st.warning("⚠️ Please select the churn column first!")
        churn_column = st.selectbox(
            "Select the churn column from your dataset:",
            st.session_state.data.columns
        )
        if churn_column:
            st.session_state.churn_column = churn_column
        return
    
    data = st.session_state.data
    churn_column = st.session_state.churn_column
    
    # Initialize data processor and model trainer
    data_processor = DataProcessor()
    model_trainer = ModelTrainer()
    
    # Separate features and target
    X = data.drop(columns=[churn_column])
    y = data[churn_column]
    
    # Show target value distribution
    unique_vals = y.value_counts()
    st.write("📊 Churn Value Distribution:")
    st.write(pd.DataFrame({
        'Value': unique_vals.index,
        'Count': unique_vals.values,
        'Percentage': (unique_vals.values / len(y) * 100).round(2)
    }))
    
    # Process only the features, not the target
    X_processed = data_processor.preprocess_data(X)
    
    # Model selection and training mode
    col1, col2 = st.columns([3, 2])
    with col1:
        model_choice = st.selectbox(
            "Select Model",
            ["Auto-Select", "Logistic Regression", "Random Forest", "XGBoost"]
        )
    with col2:
        training_mode = st.radio(
            "Training Mode",
            ["Standard", "Fast Mode", "Optuna Optimization"],
            help="""
            Standard: Full training with default hyperparameters (2-5 minutes)
            Fast Mode: Quick training with minimal optimization (30-60 seconds)
            Optuna Optimization: Advanced hyperparameter tuning (5-10 minutes)
            """
        )
        
        # Initialize n_trials with a default value
        n_trials = 30
        if training_mode == "Optuna Optimization":
            n_trials = st.slider("Number of Optuna trials", 10, 100, 30, 
                               help="More trials = better results but longer training time")
    
    if st.button("Train Model"):
        progress_text = "Training model... This may take a few minutes."
        progress_bar = st.progress(0, text=progress_text)
        
        try:
            with st.spinner():
                # Update progress
                progress_bar.progress(10, text="Preprocessing data...")
                
                model_name = None if model_choice == "Auto-Select" else model_choice.lower().replace(" ", "_")
                
                # Update progress
                progress_bar.progress(30, text="Training model...")
                
                # Training with selected mode
                if training_mode == "Fast Mode":
                    model, metrics = model_trainer.train_model(X_processed, y, model_name, fast_mode=True)
                elif training_mode == "Optuna Optimization":
                    model, metrics = model_trainer.train_model(X_processed, y, model_name, use_optuna=True, n_trials=n_trials)
                else:
                    model, metrics = model_trainer.train_model(X_processed, y, model_name)
                
                # Update progress
                progress_bar.progress(70, text="Calculating SHAP values...")
                
                # Calculate SHAP values
                try:
                    # Create explainer based on model type
                    if isinstance(model, (RandomForestClassifier, XGBClassifier)):
                        explainer = shap.TreeExplainer(model)
                        shap_values = explainer.shap_values(X_processed)
                        if isinstance(shap_values, list):
                            shap_values = shap_values[1]  # For binary classification
                    else:  # For other models like Logistic Regression
                        explainer = shap.LinearExplainer(model, X_processed)
                        shap_values = explainer.shap_values(X_processed)
                    
                    # Store SHAP values in session state
                    st.session_state.shap_values = shap_values
                    st.session_state.feature_names = X_processed.columns
                    
                    # Store in metrics for sharing
                    metrics['shap_values'] = shap_values.tolist()
                    metrics['feature_names'] = X_processed.columns.tolist()
                    
                except Exception as e:
                    st.warning(f"Could not calculate SHAP values: {str(e)}")
                
                # Store in session state
                st.session_state.model = model
                st.session_state.data_processor = data_processor
                st.session_state.model_name = model_choice
                
                # Update progress
                progress_bar.progress(100, text="Training complete!")
                
                # Display metrics
                st.success("✨ Model trained successfully!")
                
                # Show training mode info with appropriate descriptions
                mode_descriptions = {
                    "Standard": "Standard training with default hyperparameters",
                    "Fast Mode": "Quick training with reduced optimization",
                    "Optuna Optimization": f"Advanced optimization with {n_trials} trials"
                }
                st.info(f"Training Mode: {mode_descriptions[training_mode]}")
                
                # Show model info
                st.info(f"Selected Model: {model_choice if model_choice != 'Auto-Select' else model_trainer.best_model_name.replace('_', ' ').title()}")
                
                # Display metrics with better formatting
                st.subheader("📊 Model Performance")
                
                # Metrics in columns
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
                col2.metric("Precision", f"{metrics['precision']:.2%}")
                col3.metric("Recall", f"{metrics['recall']:.2%}")
                col4.metric("ROC AUC", f"{metrics['roc_auc']:.2%}")
                
                # SHAP Analysis Section
                st.subheader("🔍 Model Interpretability (SHAP)")
                
                if hasattr(st.session_state, 'shap_values'):
                    tab1, tab2, tab3 = st.tabs(["Summary Plot", "Feature Importance", "Individual Predictions"])
                    
                    with tab1:
                        st.write("SHAP Summary Plot (Impact on Model Output)")
                        fig_summary = plt.figure(figsize=(10, 8))
                        shap.summary_plot(st.session_state.shap_values, 
                                        X_processed,
                                        plot_type="bar",
                                        show=False)
                        st.pyplot(fig_summary)
                        plt.clf()
                    
                    with tab2:
                        st.write("Feature Importance based on SHAP values")
                        fig_importance = plt.figure(figsize=(10, 8))
                        shap.summary_plot(st.session_state.shap_values,
                                        X_processed,
                                        plot_type="violin",
                                        show=False)
                        st.pyplot(fig_importance)
                        plt.clf()
                    
                    with tab3:
                        st.write("Individual Prediction Explanations")
                        sample_idx = st.slider("Select a sample to explain:", 
                                             0, len(X_processed)-1, 0)
                        
                        try:
                            fig_force = plt.figure(figsize=(10, 3))
                            
                            # Handle different model types and their expected values
                            if isinstance(model, RandomForestClassifier):
                                # For Random Forest, use the first class's expected value
                                if isinstance(explainer.expected_value, list):
                                    expected_value = explainer.expected_value[1]  # For binary classification
                                else:
                                    expected_value = explainer.expected_value
                                
                                if isinstance(st.session_state.shap_values, list):
                                    shap_values = st.session_state.shap_values[1]  # For binary classification
                                else:
                                    shap_values = st.session_state.shap_values
                            else:
                                # For other models like Logistic Regression
                                expected_value = explainer.expected_value
                                shap_values = st.session_state.shap_values
                            
                            # Create force plot with correct parameters
                            shap.plots.force(
                                base_value=expected_value,
                                shap_values=shap_values[sample_idx,:],
                                features=X_processed.iloc[sample_idx,:],
                                matplotlib=True,
                                show=False
                            )
                            st.pyplot(fig_force)
                            plt.clf()
                            
                            # Show actual feature values for the selected sample
                            st.write("Feature Values for Selected Sample:")
                            sample_data = pd.DataFrame({
                                'Feature': X_processed.columns,
                                'Value': X_processed.iloc[sample_idx].values,
                                'SHAP Value': shap_values[sample_idx]
                            }).sort_values('SHAP Value', key=abs, ascending=False)
                            st.dataframe(sample_data)
                        except Exception as e:
                            st.error(f"Error creating force plot: {str(e)}")
                            st.info("This might be due to incompatible model type or SHAP values format.")
                else:
                    st.info("SHAP analysis is not available for this model.")
                
        except ValueError as ve:
            st.error(f"❌ Error: {str(ve)}")
            st.info("""
            Please make sure your churn column:
            1. Has exactly two unique values
            2. Uses consistent values (e.g., Yes/No, True/False, 1/0)
            3. Has no missing values
            """)
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")
            st.info("""
            Suggestions:
            1. Check your data for missing or invalid values
            2. Try a different model
            3. Enable fast mode for quicker training
            4. Reduce your dataset size if it's very large
            """)
        finally:
            # Clear progress bar
            progress_bar.empty()

def display_feature_importance(importance_dict):
    """Display feature importance with better visualization"""
    st.subheader("🎯 Feature Importance Analysis")
    
    # Create DataFrame for plotting
    importance_df = pd.DataFrame(
        importance_dict.items(),
        columns=['Feature', 'Importance']
    ).sort_values('Importance', ascending=True)  # Ascending for better visualization
    
    # Create a horizontal bar chart using plotly
    import plotly.graph_objects as go
    
    fig = go.Figure(go.Bar(
        x=importance_df['Importance'],
        y=importance_df['Feature'],
        orientation='h',
        marker=dict(
            color=importance_df['Importance'],
            colorscale='Viridis'
        )
    ))
    
    fig.update_layout(
        title="Feature Importance Scores",
        xaxis_title="Relative Importance",
        yaxis_title="Features",
        height=max(400, len(importance_df) * 25),  # Dynamic height based on number of features
        yaxis={'categoryorder': 'total ascending'},
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display detailed breakdown
    st.subheader("📊 Detailed Feature Importance")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Top 5 Most Important Features:")
        top_features = importance_df.tail(5)
        for idx, row in top_features.iloc[::-1].iterrows():
            st.write(f"- {row['Feature']}: {row['Importance']:.3f}")
    
    with col2:
        st.write("Bottom 5 Least Important Features:")
        bottom_features = importance_df.head(5)
        for idx, row in bottom_features.iterrows():
            st.write(f"- {row['Feature']}: {row['Importance']:.3f}")

def predict_page(data=None, churn_column=None):
    st.subheader("🔮 Make Predictions")
    
    if st.session_state.model is None or st.session_state.data_processor is None:
        st.warning("Please train a model first!")
        return

    predictor = ChurnPredictor(st.session_state.model, st.session_state.data_processor)
    
    # Get all features from training data except churn column
    features = [col for col in st.session_state.data.columns 
               if col != st.session_state.churn_column]
    
    # Create input form with all required features
    st.write("Enter customer information:")
    input_data = {}
    
    # Create input fields for each feature
    for feature in features:
        if feature in st.session_state.data_processor.categorical_columns:
            input_type = st.radio(
                f"How would you like to input {feature}?",
                ["Select from known values", "Enter new value"],
                key=f"input_type_{feature}"
            )
            
            if input_type == "Select from known values":
                known_values = st.session_state.data[feature].unique().tolist()
                input_data[feature] = st.selectbox(
                    f"Select {feature}",
                    options=known_values,
                    key=f"input_{feature}"
                )
            else:
                input_data[feature] = st.text_input(
                    f"Enter new value for {feature}",
                    key=f"input_{feature}_new"
                )
        else:
            # For numerical features only
            try:
                # Convert to numeric first to ensure it's a numerical column
                numeric_values = pd.to_numeric(st.session_state.data[feature])
                avg_value = numeric_values.mean()
                input_data[feature] = st.number_input(
                    f"Enter {feature}",
                    value=float(avg_value),
                    key=f"input_{feature}"
                )
            except (ValueError, TypeError):
                # If conversion to numeric fails, treat as string input
                input_data[feature] = st.text_input(
                    f"Enter {feature}",
                    key=f"input_{feature}"
                )
    
    if st.button("Predict"):
        try:
            # Create DataFrame with all required features in correct order
            input_df = pd.DataFrame([input_data], columns=features)
            
            # Make prediction
            result = predictor.predict(input_df)
            
            # Display results
            st.subheader("Prediction Results")
            col1, col2, col3 = st.columns(3)
            col1.metric(
                "Prediction",
                "Will Churn" if result['prediction'] else "Will Not Churn"
            )
            col2.metric("Churn Probability", f"{result['churn_probability']:.2%}")
            col3.metric("Confidence Score", f"{result['confidence_score']:.2%}")
            
            # Use the new feature importance visualization
            display_feature_importance(result['feature_importance'])
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.info("""
            Suggestions to resolve this:
            1. Make sure all required features are filled
            2. Try using Random Forest or XGBoost models instead of Logistic Regression
            3. For categorical features, consider using known values if possible
            """)

def generate_report_page():
    st.subheader("📊 Generate Report")
    
    if st.session_state.model is None:
        st.warning("Please train a model first!")
        return
    
    analytics = ChurnAnalytics(st.session_state.data)
    predictor = ChurnPredictor(st.session_state.model, st.session_state.data_processor)
    report_gen = ReportGenerator(analytics, predictor)
    
    # Use default values if not available
    model_name = st.session_state.model_name if st.session_state.model_name else "Unknown_Model"
    username = st.session_state.username if st.session_state.username else "anonymous"
    
    # Generate report
    report = report_gen.generate_report(
        st.session_state.churn_column,
        model_name
    )
    
    # Display report summary
    st.json(report)
    
    # Save report button
    if st.button("Save Report"):
        try:
            filepath = report_gen.save_report(
                report,
                username,
                model_name
            )
            
            # Read the saved report for download
            with open(filepath, "rb") as file:
                btn = st.download_button(
                    label="Download Report",
                    data=file,
                    file_name=os.path.basename(filepath),
                    mime="application/json"
                )
            
            st.success(f"Report saved successfully! You can download it using the button above.")
            
        except Exception as e:
            st.error(f"Error saving report: {str(e)}")

def check_permission(required_permission: str):
    """Decorator to check role permissions"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            from config import ROLES
            user_role = st.session_state.role
            
            # Admin has access to everything
            if user_role == 'admin':
                return func(*args, **kwargs)
            
            # Check if user's role has the required permission
            user_permissions = ROLES.get(user_role, [])
            if required_permission in user_permissions:
                return func(*args, **kwargs)
            else:
                st.error("You don't have permission to access this feature!")
                return None
        return wrapper
    return decorator

def generate_analysis_key():
    """Generate a unique analysis key"""
    return str(uuid.uuid4())[:8].upper()

def shared_analysis_page():
    st.header("🔗 Shared Analysis")
    
    # Create tabs for different functions
    tab1, tab2 = st.tabs(["View Shared Analysis", "Share Your Analysis"])
    
    with tab1:
        st.subheader("Access Shared Analysis")
        
        # Option to view by entering key or browsing public analyses
        view_option = st.radio("Choose how to view:", ["Enter Analysis Key", "Browse Public Analyses"])
        
        if view_option == "Enter Analysis Key":
            analysis_key = st.text_input("Enter Analysis Key", key="analysis_key_input")
            if analysis_key:
                try:
                    conn = sqlite3.connect(DB_PATH)
                    cursor = conn.cursor()
                    
                    # Get analysis details
                    cursor.execute("""
                        SELECT sa.*, u.username 
                        FROM shared_analysis sa 
                        JOIN users u ON sa.user_id = u.id 
                        WHERE analysis_key = ?
                    """, (analysis_key,))
                    result = cursor.fetchone()
                    
                    if result:
                        # Update view count
                        cursor.execute("UPDATE shared_analysis SET view_count = view_count + 1 WHERE analysis_key = ?", 
                                    (analysis_key,))
                        conn.commit()
                        
                        # Display analysis details
                        st.success(f"Analysis found! Created by: {result[13]}")
                        
                        with st.expander("📊 Analysis Details", expanded=True):
                            st.write(f"**Title:** {result[3]}")
                            st.write(f"**Description:** {result[4]}")
                            st.write(f"**Created:** {result[10]}")
                            st.write(f"**Views:** {result[12]}")
                            
                            # Display dataset preview
                            if result[5]:  # dataset_preview
                                st.subheader("Dataset Preview")
                                df_preview = pd.read_json(result[5])
                                st.dataframe(df_preview)
                            
                            # Display analysis results
                            if result[6]:  # analysis_results
                                st.subheader("Analysis Results")
                                analysis_results = json.loads(result[6])
                                for key, value in analysis_results.items():
                                    st.write(f"**{key}:** {value}")
                            
                            # Display model metrics
                            if result[7]:  # model_metrics
                                st.subheader("Model Performance")
                                metrics = json.loads(result[7])
                                cols = st.columns(4)
                                metrics_display = {
                                    "Accuracy": metrics.get('accuracy', 0),
                                    "Precision": metrics.get('precision', 0),
                                    "Recall": metrics.get('recall', 0),
                                    "ROC AUC": metrics.get('roc_auc', 0)
                                }
                                for i, (metric, value) in enumerate(metrics_display.items()):
                                    cols[i].metric(metric, f"{value:.2%}")
                            
                            # Display feature importance
                            if result[8]:  # feature_importance
                                st.subheader("🎯 Feature Importance")
                                try:
                                    feature_imp = json.loads(result[8])
                                    if feature_imp:
                                        # Create a bar chart using plotly
                                        fig = go.Figure([go.Bar(
                                            x=list(feature_imp.values()),
                                            y=list(feature_imp.keys()),
                                            orientation='h',
                                            marker=dict(
                                                color=list(feature_imp.values()),
                                                colorscale='Viridis'
                                            )
                                        )])
                                        
                                        fig.update_layout(
                                            title="Feature Importance Scores",
                                            xaxis_title="Relative Importance",
                                            yaxis_title="Features",
                                            height=max(400, len(feature_imp) * 25),
                                            yaxis={'categoryorder':'total ascending'},
                                            template="plotly_dark",
                                            showlegend=False
                                        )
                                        
                                        st.plotly_chart(fig, use_container_width=True)
                                        
                                        # Display top and bottom features
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.write("Top 5 Most Important Features:")
                                            for feature, importance in list(feature_imp.items())[:5]:
                                                st.write(f"- {feature}: {importance:.3f}")
                                        
                                        with col2:
                                            st.write("Bottom 5 Least Important Features:")
                                            for feature, importance in list(feature_imp.items())[-5:]:
                                                st.write(f"- {feature}: {importance:.3f}")
                                    else:
                                        st.info("No feature importance data available for this analysis.")
                                except Exception as e:
                                    st.error(f"Error displaying feature importance: {str(e)}")
                                    st.info("Could not display feature importance. The data might be in an incorrect format.")
                    else:
                        st.error("Analysis not found. Please check the key and try again.")
                    
                    conn.close()
                except Exception as e:
                    st.error(f"Error accessing analysis: {str(e)}")
        
        else:  # Browse Public Analyses
            try:
                conn = sqlite3.connect(DB_PATH)
                public_analyses = pd.read_sql("""
                    SELECT sa.analysis_key, sa.title, sa.description, sa.created_at, 
                           sa.view_count, u.username as creator
                    FROM shared_analysis sa
                    JOIN users u ON sa.user_id = u.id
                    WHERE sa.is_public = TRUE
                    ORDER BY sa.created_at DESC
                """, conn)
                
                if not public_analyses.empty:
                    st.dataframe(public_analyses)
                    
                    # Allow viewing selected analysis
                    selected_key = st.selectbox("Select analysis to view:", 
                                              public_analyses['analysis_key'].tolist())
                    if selected_key:
                        # Instead of rerunning, directly show the analysis
                        cursor = conn.cursor()
                        cursor.execute("""
                            SELECT sa.*, u.username 
                            FROM shared_analysis sa 
                            JOIN users u ON sa.user_id = u.id 
                            WHERE analysis_key = ?
                        """, (selected_key,))
                        result = cursor.fetchone()
                        
                        if result:
                            # Update view count
                            cursor.execute("UPDATE shared_analysis SET view_count = view_count + 1 WHERE analysis_key = ?", 
                                        (selected_key,))
                            conn.commit()
                            
                            # Display analysis details
                            st.success(f"Analysis found! Created by: {result[13]}")
                            
                            with st.expander("📊 Analysis Details", expanded=True):
                                st.write(f"**Title:** {result[3]}")
                                st.write(f"**Description:** {result[4]}")
                                st.write(f"**Created:** {result[10]}")
                                st.write(f"**Views:** {result[12]}")
                                
                                # Display dataset preview
                                if result[5]:  # dataset_preview
                                    st.subheader("Dataset Preview")
                                    df_preview = pd.read_json(result[5])
                                    st.dataframe(df_preview)
                                
                                # Display analysis results
                                if result[6]:  # analysis_results
                                    st.subheader("Analysis Results")
                                    analysis_results = json.loads(result[6])
                                    for key, value in analysis_results.items():
                                        st.write(f"**{key}:** {value}")
                                
                                # Display model metrics
                                if result[7]:  # model_metrics
                                    st.subheader("Model Performance")
                                    metrics = json.loads(result[7])
                                    cols = st.columns(4)
                                    metrics_display = {
                                        "Accuracy": metrics.get('accuracy', 0),
                                        "Precision": metrics.get('precision', 0),
                                        "Recall": metrics.get('recall', 0),
                                        "ROC AUC": metrics.get('roc_auc', 0)
                                    }
                                    for i, (metric, value) in enumerate(metrics_display.items()):
                                        cols[i].metric(metric, f"{value:.2%}")
                                
                                # Display feature importance
                                if result[8]:  # feature_importance
                                    st.subheader("🎯 Feature Importance")
                                    try:
                                        feature_imp = json.loads(result[8])
                                        if feature_imp:
                                            # Create a bar chart using plotly
                                            fig = go.Figure([go.Bar(
                                                x=list(feature_imp.values()),
                                                y=list(feature_imp.keys()),
                                                orientation='h',
                                                marker=dict(
                                                    color=list(feature_imp.values()),
                                                    colorscale='Viridis'
                                                )
                                            )])
                                            
                                            fig.update_layout(
                                                title="Feature Importance Scores",
                                                xaxis_title="Relative Importance",
                                                yaxis_title="Features",
                                                height=max(400, len(feature_imp) * 25),
                                                yaxis={'categoryorder':'total ascending'},
                                                template="plotly_dark",
                                                showlegend=False
                                            )
                                            
                                            st.plotly_chart(fig, use_container_width=True)
                                            
                                            # Display top and bottom features
                                            col1, col2 = st.columns(2)
                                            with col1:
                                                st.write("Top 5 Most Important Features:")
                                                for feature, importance in list(feature_imp.items())[:5]:
                                                    st.write(f"- {feature}: {importance:.3f}")
                                            
                                            with col2:
                                                st.write("Bottom 5 Least Important Features:")
                                                for feature, importance in list(feature_imp.items())[-5:]:
                                                    st.write(f"- {feature}: {importance:.3f}")
                                        else:
                                            st.info("No feature importance data available for this analysis.")
                                    except Exception as e:
                                        st.error(f"Error displaying feature importance: {str(e)}")
                                        st.info("Could not display feature importance. The data might be in an incorrect format.")
                else:
                    st.info("No public analyses available yet.")
                
                conn.close()
            except Exception as e:
                st.error(f"Error loading public analyses: {str(e)}")
    
    with tab2:
        if st.session_state.role in ['admin', 'data_scientist', 'business_analyst']:
            st.subheader("Share Your Analysis")
            
            if st.session_state.data is not None and hasattr(st.session_state, 'model'):
                # Form for sharing analysis
                title = st.text_input("Analysis Title", 
                                    placeholder="Enter a descriptive title for your analysis")
                description = st.text_area("Description", 
                                         placeholder="Describe your analysis and key findings")
                is_public = st.checkbox("Make this analysis public", 
                                      help="Public analyses can be viewed by anyone")
                
                if st.button("Generate Analysis Key"):
                    try:
                        # Generate unique key
                        analysis_key = generate_analysis_key()
                        
                        # Prepare data for storage
                        dataset_preview = st.session_state.data.head(5).to_json()
                        
                        # Get analysis results
                        analysis_results = {
                            "Total Records": len(st.session_state.data),
                            "Features Used": list(st.session_state.data.columns),
                            "Model Type": st.session_state.model_name if hasattr(st.session_state, 'model_name') else "Unknown Model"
                        }
                        
                        # Get model metrics
                        X = st.session_state.data.drop(columns=[st.session_state.churn_column])
                        y = st.session_state.data[st.session_state.churn_column]
                        
                        # Preprocess data using stored data_processor
                        if hasattr(st.session_state, 'data_processor'):
                            X_processed = st.session_state.data_processor.preprocess_data(X)
                        else:
                            X_processed = X
                            
                        # Get predictions
                        y_pred = st.session_state.model.predict(X_processed)
                        y_prob = st.session_state.model.predict_proba(X_processed)[:, 1]
                        
                        # Calculate metrics
                        metrics = {
                            'accuracy': float(accuracy_score(y, y_pred)),
                            'precision': float(precision_score(y, y_pred)),
                            'recall': float(recall_score(y, y_pred)),
                            'roc_auc': float(roc_auc_score(y, y_prob))
                        }
                        
                        # Get feature importance
                        if hasattr(st.session_state.model, 'feature_importances_'):
                            feature_importance = {
                                str(col): float(imp) 
                                for col, imp in zip(
                                    [c for c in st.session_state.data.columns if c != st.session_state.churn_column],
                                    st.session_state.model.feature_importances_
                                )
                            }
                        else:
                            feature_importance = {}
                        
                        conn = sqlite3.connect(DB_PATH)
                        cursor = conn.cursor()
                        
                        # Store in database with metrics
                        cursor.execute("""
                            INSERT INTO shared_analysis (
                                analysis_key, user_id, title, description, dataset_preview,
                                analysis_results, model_metrics, feature_importance, is_public
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            analysis_key,
                            st.session_state.user_id,
                            title,
                            description,
                            dataset_preview,
                            json.dumps(analysis_results),
                            json.dumps(metrics),
                            json.dumps(st.session_state.feature_importance if hasattr(st.session_state, 'feature_importance') else {}),
                            is_public
                        ))
                        
                        conn.commit()
                        conn.close()
                        
                        # Show success message with metrics preview
                        st.success(f"Analysis shared successfully! Your analysis key is: {analysis_key}")
                        
                        # Display metrics preview
                        st.subheader("Model Performance Preview")
                        cols = st.columns(4)
                        cols[0].metric("Accuracy", f"{metrics['accuracy']:.2%}")
                        cols[1].metric("Precision", f"{metrics['precision']:.2%}")
                        cols[2].metric("Recall", f"{metrics['recall']:.2%}")
                        cols[3].metric("ROC AUC", f"{metrics['roc_auc']:.2%}")
                        
                        st.info("Share this key with others to let them view your analysis.")
                        
                    except Exception as e:
                        st.error(f"Error sharing analysis: {str(e)}")
                        st.info("Please make sure your model is properly trained and data is preprocessed correctly.")
            else:
                st.warning("Please upload data and train a model before sharing analysis.")
        else:
            st.error("You don't have permission to share analysis. Please upgrade your role to share.")

def analysis_page():
    st.header("📊 Analysis")
    
    if st.session_state.data is None:
        st.warning("Please upload data first!")
        return
        
    # Get the churn column if not already set
    if 'churn_column' not in st.session_state:
        st.session_state.churn_column = st.selectbox(
            "Select the churn column",
            st.session_state.data.columns
        )
    
    # Perform analysis
    analyze_data(st.session_state.data, st.session_state.churn_column)

def report_page():
    st.header("📈 Reports")
    
    if st.session_state.data is None:
        st.warning("Please upload data first!")
        return
        
    # Report generation options
    report_type = st.selectbox(
        "Select Report Type",
        ["Churn Analysis Summary", "Feature Impact Report"]
    )
    
    if report_type == "Churn Analysis Summary":
        generate_churn_summary()
    elif report_type == "Feature Impact Report":
        generate_feature_impact()

def generate_churn_summary():
    """Generate a summary report of churn analysis"""
    st.subheader("Churn Analysis Summary")
    
    data = st.session_state.data
    
    # Check if churn column is selected
    if 'churn_column' not in st.session_state:
        st.warning("Please select the churn column first!")
        churn_column = st.selectbox(
            "Select the churn column",
            data.columns
        )
        st.session_state.churn_column = churn_column
    else:
        churn_column = st.session_state.churn_column
    
    # Verify churn column exists in data
    if churn_column not in data.columns:
        st.error(f"Churn column '{churn_column}' not found in the data!")
        return
    
    # Overall metrics
    col1, col2, col3 = st.columns(3)
    
    # Convert churn column to numeric if it's not already
    if data[churn_column].dtype not in ['int64', 'float64']:
        # Try to convert unique values to 0 and 1
        unique_vals = data[churn_column].unique()
        if len(unique_vals) == 2:
            churn_map = {unique_vals[0]: 0, unique_vals[1]: 1}
            numeric_churn = data[churn_column].map(churn_map)
        else:
            st.error("Churn column must have exactly two unique values!")
            return
    else:
        numeric_churn = data[churn_column]
    
    # Churn rate
    churn_rate = numeric_churn.mean()
    with col1:
        st.metric("Overall Churn Rate", f"{churn_rate:.1%}")
    
    # Total customers
    with col2:
        st.metric("Total Customers", len(data))
    
    # Churned customers
    with col3:
        st.metric("Churned Customers", numeric_churn.sum())
    
    # Churn trend if time data available
    if 'date' in data.columns or 'month' in data.columns or 'year' in data.columns:
        st.subheader("Churn Trend")
        time_col = [col for col in data.columns if col in ['date', 'month', 'year']][0]
        churn_trend = data.groupby(time_col)[churn_column].mean()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=churn_trend.index, y=churn_trend.values))
        fig.update_layout(title="Churn Rate Over Time")
        st.plotly_chart(fig)
    
    # Top factors correlation with churn
    st.subheader("Top Factors Influencing Churn")
    
    # Get numeric columns excluding the churn column
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
    numeric_cols = [col for col in numeric_cols if col != churn_column]
    
    if len(numeric_cols) > 0:
        # Calculate correlations with numeric churn values
        correlations = data[numeric_cols].apply(lambda x: x.corr(numeric_churn))
        correlations = correlations.sort_values(ascending=False)
        
        fig = go.Figure(go.Bar(
            x=correlations.index,
            y=correlations.values,
            text=correlations.values.round(3),
            textposition='auto',
        ))
        fig.update_layout(title="Feature Correlations with Churn")
        st.plotly_chart(fig)
    else:
        st.warning("No numeric features found for correlation analysis!")

def generate_feature_impact():
    """Generate a report on feature impact on churn"""
    st.subheader("Feature Impact Analysis")
    
    # Check if data is uploaded
    if not hasattr(st.session_state, 'data') or st.session_state.data is None:
        st.warning("⚠️ Please upload your dataset first in the Upload Data page!")
        st.info("Go to 📁 Upload Data in the navigation menu to upload your dataset.")
        return
    
    # Check if churn column is selected
    if not hasattr(st.session_state, 'churn_column') or st.session_state.churn_column is None:
        st.warning("⚠️ Please select the churn column first!")
        churn_column = st.selectbox(
            "Select the churn column from your dataset:",
            st.session_state.data.columns
        )
        if churn_column:
            st.session_state.churn_column = churn_column
        return
    
    data = st.session_state.data
    churn_column = st.session_state.churn_column
    
    # Convert churn column to numeric if it's not already
    if data[churn_column].dtype not in ['int64', 'float64']:
        # Try to convert unique values to 0 and 1
        unique_vals = data[churn_column].unique()
        if len(unique_vals) == 2:
            churn_map = {unique_vals[0]: 0, unique_vals[1]: 1}
            data = data.copy()
            data[churn_column] = data[churn_column].map(churn_map)
            st.info(f"Converted churn values: {unique_vals[0]} → 0, {unique_vals[1]} → 1")
        else:
            st.error("Churn column must have exactly two unique values!")
            return
    
    # If model exists and user is data scientist, use model-based feature importance
    if hasattr(st.session_state, 'model') and st.session_state.model is not None and st.session_state.role == 'data_scientist':
        if hasattr(st.session_state.model, 'feature_importances_'):
            # Get feature names excluding the churn column
            feature_names = [col for col in data.columns if col != churn_column]
            importances = st.session_state.model.feature_importances_
            
            # Create DataFrame ensuring both arrays have the same length
            feature_imp = pd.DataFrame({
                'feature': feature_names[:len(importances)],
                'importance': importances[:len(feature_names)]
            }).sort_values('importance', ascending=True)
            
            fig = go.Figure(go.Bar(
                x=feature_imp['importance'],
                y=feature_imp['feature'],
                orientation='h'
            ))
            fig.update_layout(title="Model-based Feature Importance")
            st.plotly_chart(fig)
    else:
        # For business analysts or when no model is available, use correlation-based importance
        st.info("Using correlation-based feature importance analysis")
        
        # Get numeric columns excluding the churn column
        numeric_cols = [col for col in data.select_dtypes(include=['int64', 'float64']).columns 
                       if col != churn_column]
        
        if not numeric_cols:
            st.warning("No numeric features found for correlation analysis!")
            return
        
        # Calculate correlations with numeric churn values
        correlations = pd.DataFrame()
        for col in numeric_cols:
            correlations[col] = [abs(data[col].corr(data[churn_column]))]
        
        # Sort correlations
        correlations = correlations.T.sort_values(by=0, ascending=True)
        
        # Create correlation-based importance visualization
        fig = go.Figure(go.Bar(
            x=correlations[0].values,
            y=correlations.index,
            orientation='h'
        ))
        fig.update_layout(
            title="Feature Importance (based on correlation with churn)",
            xaxis_title="Absolute Correlation",
            yaxis_title="Features"
        )
        st.plotly_chart(fig)
        
        # Display detailed analysis
        st.subheader("Feature Impact Details")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Top 5 Most Important Features:")
            top_features = correlations.tail(5)
            for feature in top_features.index:
                st.write(f"- {feature}: {correlations.loc[feature, 0]:.3f}")
        
        with col2:
            st.write("Bottom 5 Least Important Features:")
            bottom_features = correlations.head(5)
            for feature in bottom_features.index:
                st.write(f"- {feature}: {correlations.loc[feature, 0]:.3f}")
        
        # Add feature impact explanations
        st.subheader("Feature Impact Interpretation")
        selected_feature = st.selectbox(
            "Select a feature to analyze:",
            numeric_cols
        )
        
        # Calculate and display specific feature analysis
        feature_corr = data[selected_feature].corr(data[churn_column])
        direction = "positive" if feature_corr > 0 else "negative"
        strength = abs(feature_corr)
        
        st.write(f"### Impact of {selected_feature}")
        st.write(f"""
        - Correlation with churn: {feature_corr:.3f}
        - Direction: {direction} correlation
        - Impact strength: {'High' if strength > 0.5 else 'Medium' if strength > 0.3 else 'Low'}
        """)
        
        # Show distribution comparison
        fig = go.Figure()
        for churn_val in [0, 1]:
            fig.add_trace(go.Box(
                y=data[data[churn_column] == churn_val][selected_feature],
                name=f"{'Churned' if churn_val == 1 else 'Not Churned'}"
            ))
        fig.update_layout(title=f"Distribution of {selected_feature} by Churn Status")
        st.plotly_chart(fig)

def upload_data_page():
    st.header("📁 Upload Data")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.data = df
            
            # Select churn column
            churn_column = st.selectbox(
                "Select the churn column",
                df.columns
            )
            if churn_column:
                st.session_state.churn_column = churn_column
            
            st.success("Data uploaded successfully!")
            st.write("Preview of your data:")
            st.write(df.head())
            
            # Display basic statistics
            st.subheader("Dataset Overview")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", len(df))
            with col2:
                st.metric("Features", len(df.columns))
            with col3:
                st.metric("Missing Values", df.isnull().sum().sum())
            
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            st.info("""
            Please make sure:
            1. The file is in CSV format
            2. The file is not corrupted
            3. The file contains valid data
            """)

def main():
    st.set_page_config(page_title="SalesVizz", page_icon="📊", layout="wide")
    
    # Initialize session state
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'role' not in st.session_state:
        st.session_state.role = None
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'data_processor' not in st.session_state:
        st.session_state.data_processor = None
    if 'model_name' not in st.session_state:
        st.session_state.model_name = None
    if 'churn_column' not in st.session_state:
        st.session_state.churn_column = None

    # Initialize database
    init_db()
    
    # Display login/register form if not authenticated
    if not st.session_state.authenticated:
        login_page()
        return
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    
    # Show user info
    st.sidebar.markdown(f"**Logged in as:** {st.session_state.username}")
    st.sidebar.markdown(f"**Role:** {st.session_state.role}")
    
    # Role-specific pages
    if st.session_state.role == 'viewer':
        pages = {
            "🔗 Shared Analysis": shared_analysis_page
        }
    elif st.session_state.role == 'data_scientist':
        pages = {
            "📁 Upload Data": upload_data_page,
            "📊 Analysis": analysis_page,
            "🤖 Train Model": train_model_page,
            "🔮 Make Predictions": predict_page,
            "📈 Reports": report_page,
            "🔬 Advanced Analytics": generate_feature_impact,
            "🔗 Shared Analysis": shared_analysis_page
        }
    elif st.session_state.role == 'business_analyst':
        pages = {
            "📁 Upload Data": upload_data_page,
            "📊 Analysis": analysis_page,
            "📈 Reports": report_page,
            "🔗 Shared Analysis": shared_analysis_page
        }
    elif st.session_state.role == 'admin':
        pages = {
            "👥 User Management": user_management_page,
            "📁 Upload Data": upload_data_page,
            "📊 Analysis": analysis_page,
            "🤖 Train Model": train_model_page,
            "🔮 Make Predictions": predict_page,
            "📈 Reports": report_page,
            "🔬 Advanced Analytics": generate_feature_impact,
            "⚙️ System Settings": system_settings_page,
            "📊 Usage Analytics": usage_analytics_page,
            "🔗 Shared Analysis": shared_analysis_page
        }
    else:  # default
        pages = {
            "📁 Upload Data": upload_data_page,
            "📊 Analysis": analysis_page,
            "🤖 Train Model": train_model_page,
            "🔮 Make Predictions": predict_page,
            "📈 Reports": report_page,
            "🔗 Shared Analysis": shared_analysis_page
        }
    
    page = st.sidebar.radio("Select a page:", list(pages.keys()))
    
    # Logout button
    if st.sidebar.button("Logout"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.experimental_rerun()
    
    # Display selected page
    st.title("SalesVizz")
    pages[page]()

def user_management_page():
    """Admin page for managing users"""
    st.header("👥 User Management")
    
    if st.session_state.role != 'admin':
        st.error("Access denied. Admin privileges required.")
        return
    
    # Get all users from database
    conn = sqlite3.connect(DB_PATH)
    users_df = pd.read_sql("SELECT id, username, role FROM users", conn)
    
    # Display users in an interactive table
    st.subheader("Current Users")
    st.dataframe(users_df)
    
    # User actions
    st.subheader("User Actions")
    
    # Create three columns for different actions
    col1, col2, col3 = st.columns(3)
    
    # New User Creation (First Column)
    with col1:
        st.subheader("Create New User")
        new_username = st.text_input("Username", key="new_user_username")
        new_password = st.text_input("Password", type="password", key="new_user_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="new_user_confirm")
        new_role = st.selectbox(
            "Role",
            ['viewer', 'business_analyst', 'data_scientist', 'admin'],
            key="new_user_role"
        )
        
        if st.button("Create User"):
            if not new_username or not new_password:
                st.error("Username and password are required!")
            elif new_password != confirm_password:
                st.error("Passwords don't match!")
            else:
                try:
                    cursor = conn.cursor()
                    # Check if username exists
                    cursor.execute("SELECT id FROM users WHERE username = ?", (new_username,))
                    if cursor.fetchone() is not None:
                        st.error("Username already exists!")
                    else:
                        # Create new user
                        auth.register_user(new_username, new_password, new_role)
                        st.success(f"Created new user {new_username} with role {new_role}")
                        st.rerun()
                except Exception as e:
                    st.error(f"Error creating user: {str(e)}")
    
    # Modify User Role (Second Column)
    with col2:
        st.subheader("Modify User Role")
        user_to_modify = st.selectbox(
            "Select user",
            users_df['username'].tolist(),
            key="modify_user"
        )
        new_role = st.selectbox(
            "New role",
            ['viewer', 'business_analyst', 'data_scientist', 'admin'],
            key="new_role"
        )
        if st.button("Update Role"):
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE users SET role = ? WHERE username = ?",
                    (new_role, user_to_modify)
                )
                conn.commit()
                st.success(f"Updated role for {user_to_modify} to {new_role}")
                st.rerun()
            except Exception as e:
                st.error(f"Error updating role: {str(e)}")
    
    # Delete User (Third Column)
    with col3:
        st.subheader("Delete User")
        user_to_delete = st.selectbox(
            "Select user to delete",
            users_df['username'].tolist(),
            key="delete_user"
        )
        if st.button("Delete User", type="primary"):
            if user_to_delete == st.session_state.username:
                st.error("Cannot delete your own account!")
            else:
                try:
                    cursor = conn.cursor()
                    cursor.execute(
                        "DELETE FROM users WHERE username = ?",
                        (user_to_delete,)
                    )
                    conn.commit()
                    st.success(f"Deleted user {user_to_delete}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error deleting user: {str(e)}")
    
    # User activity monitoring
    st.subheader("User Activity")
    activity_df = pd.read_sql("""
        SELECT u.username, sa.title, sa.created_at, sa.view_count
        FROM shared_analysis sa
        JOIN users u ON sa.user_id = u.id
        ORDER BY sa.created_at DESC
        LIMIT 10
    """, conn)
    
    if not activity_df.empty:
        st.dataframe(activity_df)
    else:
        st.info("No user activity recorded yet")
    
    conn.close()

def system_settings_page():
    """Admin page for system settings"""
    st.header("⚙️ System Settings")
    
    if st.session_state.role != 'admin':
        st.error("Access denied. Admin privileges required.")
        return
    
    # Database Management
    st.subheader("Database Management")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Backup Database"):
            try:
                backup_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = os.path.join(DATA_DIR, f'backup_salesvizz_{backup_time}.db')
                import shutil
                shutil.copy2(DB_PATH, backup_path)
                st.success(f"Database backed up successfully to {backup_path}")
            except Exception as e:
                st.error(f"Error backing up database: {str(e)}")
    
    with col2:
        if st.button("Clear Analysis History"):
            if st.checkbox("I understand this will delete all shared analyses"):
                try:
                    conn = sqlite3.connect(DB_PATH)
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM shared_analysis")
                    conn.commit()
                    conn.close()
                    st.success("Analysis history cleared successfully")
                except Exception as e:
                    st.error(f"Error clearing history: {str(e)}")
    
    # System Configuration
    st.subheader("System Configuration")
    
    # Model settings
    st.write("Model Training Settings")
    max_trials = st.slider("Maximum Optuna Trials", 10, 200, 100)
    if st.button("Save Settings"):
        st.session_state.max_optuna_trials = max_trials
        st.success("Settings saved successfully")

def usage_analytics_page():
    """Admin page for usage analytics"""
    st.header("📊 Usage Analytics")
    
    if st.session_state.role != 'admin':
        st.error("Access denied. Admin privileges required.")
        return
    
    conn = sqlite3.connect(DB_PATH)
    
    # User Statistics
    st.subheader("User Statistics")
    col1, col2, col3 = st.columns(3)
    
    # Total users
    total_users = pd.read_sql("SELECT COUNT(*) as count FROM users", conn).iloc[0]['count']
    col1.metric("Total Users", total_users)
    
    # Users by role
    users_by_role = pd.read_sql(
        "SELECT role, COUNT(*) as count FROM users GROUP BY role",
        conn
    )
    col2.metric(
        "Most Common Role",
        users_by_role.iloc[users_by_role['count'].argmax()]['role']
    )
    
    # Active users (users with shared analyses)
    active_users = pd.read_sql(
        "SELECT COUNT(DISTINCT user_id) as count FROM shared_analysis",
        conn
    ).iloc[0]['count']
    col3.metric("Active Users", active_users)
    
    # Analysis Statistics
    st.subheader("Analysis Statistics")
    
    # Analysis trends
    analysis_trend = pd.read_sql("""
        SELECT DATE(created_at) as date, COUNT(*) as count
        FROM shared_analysis
        GROUP BY DATE(created_at)
        ORDER BY date DESC
        LIMIT 30
    """, conn)
    
    if not analysis_trend.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=analysis_trend['date'],
            y=analysis_trend['count'],
            mode='lines+markers'
        ))
        fig.update_layout(title="Daily Analysis Creation Trend")
        st.plotly_chart(fig)
    
    # Popular analyses
    st.subheader("Most Viewed Analyses")
    popular_analyses = pd.read_sql("""
        SELECT sa.title, u.username, sa.view_count, sa.created_at
        FROM shared_analysis sa
        JOIN users u ON sa.user_id = u.id
        ORDER BY sa.view_count DESC
        LIMIT 5
    """, conn)
    
    if not popular_analyses.empty:
        st.dataframe(popular_analyses)
    
    conn.close()

if __name__ == "__main__":
    main() 