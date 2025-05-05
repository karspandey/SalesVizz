# SalesVizz - Customer Churn Analysis Platform

SalesVizz is a comprehensive data analytics and churn prediction platform built with Streamlit. It helps businesses analyze customer behavior, predict churn, and generate actionable insights.

## Features

- 🔐 User Authentication System
- 📊 Interactive Data Analysis
- 🤖 Machine Learning Models
- 📈 Churn Predictions
- 📋 Business Intelligence Reports
- 📉 Feature Importance Analysis

## Setup Instructions

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
cd app
streamlit run main.py #The app will be accessible in your web browser at http://localhost:8501.
```

## Usage
Once the app is running, you can use it to:
Upload customer data to analyze and visualize.
Evaluate churn prediction models with automated metrics.
Tune model hyperparameters for optimal performance.
Visualize feature importance to understand key drivers of customer churn.
Generate business intelligence reports to make informed decisions.
The user interface is interactive and easy to use, with prompts guiding you through each step of the process.



## Project Structure
Here’s an overview of the project structure:
SalesVizz/
```bash
├── app/                        # Contains the Streamlit app code
│   ├── main.py                 # Main file to run the Streamlit app
│   ├── analytics/              # Contains analytics-related scripts
│   │   └── analytics.py        # Handles analytics visualizations and metric calculations
│   ├── auth/                   # Manages authentication and user sessions
│   │   └── auth_handler.py     # Manages user authentication and security flows
│   ├── database/               # Contains database models and schemas
│   │   └── models.py           # Defines database schemas and ORM models
│   ├── ml/                     # Contains machine learning-related code
│   │   ├── model_trainer.py    # Trains machine learning models for churn prediction
│   │   └── predictor.py        # Generates predictions using trained models
│   ├── reports/                # Contains report generation scripts
│   │   └── report_generator.py # Creates business intelligence reports
│   ├── utils/                  # Utility scripts for data handling and validation
│   │   ├── data_processor.py   # Cleans and preprocesses input data
│   │   ├── error_messages.py   # Centralizes error message definitions
│   │   ├── logger.py           # Configures logging for the application
│   │   └── validators.py       # Validates user inputs and data formats
│   ├── config.py               # Application configuration settings
│   └── main.py                 # Entry point for the Streamlit app
├── .gitignore                 # Git ignore file for unnecessary files
├── LICENSE                    # Project license information
├── README.md                  # Project documentation (this file)
└── requirements.txt           # List of dependencies
```



## Contributing
We welcome contributions to SalesVizz! If you would like to contribute, please follow these steps:
Fork the repository.
Create a new branch for your feature or bug fix.
Write tests for your changes (if applicable).
Make your changes and commit them with descriptive messages.
Push your changes and submit a pull request.
We’ll review your changes and get back to you as soon as possible.

## License
SalesVizz is open-source and licensed under the MIT License. See the LICENSE file for more details.