# Email Campaign KPI Predictor

## 📧 Advanced ML tool for predicting and optimizing email marketing campaigns

![Email Campaign KPI Predictor Banner](https://via.placeholder.com/1200x300)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.15.0-FF4B4B)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.6.2-006ACC)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-3.3.2-3DAD47)](https://lightgbm.readthedocs.io/)
[![Plotly](https://img.shields.io/badge/Plotly-5.9.0-3F4F75)](https://plotly.com/)

## 📋 Overview

Email Campaign KPI Predictor is a machine learning-powered tool for email marketers to predict open rates and optimize campaign performance. Built specifically for Adobe Campaign data, this application helps marketing teams make data-driven decisions about targeting, subject lines, and delivery timing.

## ✨ Features

- **Predictive Analytics**: ML model predicts open rates based on targeting, content, and timing
- **Smart Recommendations**: 
  - Targeting optimization based on historical performance
  - Subject line recommendations using NLP and clustering techniques
  - Send time optimization
- **Interactive Visualizations**:
  - Performance comparison charts
  - Feature importance analysis
  - Performance radar charts
  - Historical trend analysis
- **Comprehensive Reporting**:
  - Exportable predictions and recommendations
  - Downloadable campaign reports
  - Model performance metrics

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/email-campaign-kpi-predictor.git
cd email-campaign-kpi-predictor
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## 📊 Data Requirements

The application requires two CSV files exported from Adobe Campaign:

### Delivery Level Data
File name: `delivery_data.csv`
Required columns:
- `InternalName` (delivery identifier)
- `subject` (email subject line)
- `Contact date` (date and time of delivery)
- `Utskick` (sends count)
- `Opens` (opens count)
- `Clicks` (clicks count)
- `Optout` (unsubscribe count)
- `county` (targeting region)
- `Dialog`, `Syfte`, `Produkt`, `Bolag` (campaign metadata)

### Customer Level Data
File name: `customer_data.csv`
Required columns:
- `Primary k isWoman OptOut` (customer identifier)
- `Gender` (customer gender)
- `Age` (customer age)
- `InternalName` (delivery identifier to link with delivery data)
- `Dialog`, `Syfte`, `Product`, `Bolag` (campaign metadata)

## 🚀 Usage

1. Place your data files (`delivery_data.csv` and `customer_data.csv`) in the application directory.

2. Run the Streamlit app:
```bash
streamlit run app.py
```

3. Access the app through your browser at `http://localhost:8501`

4. Use the interface to:
   - Input campaign parameters
   - View predictions and recommendations
   - Explore performance insights
   - Export reports and predictions

## 🔍 Model Information

The application trains and evaluates multiple machine learning models:

- **XGBoost Regressor**: Gradient boosting framework for regression
- **LightGBM**: Gradient boosting framework optimized for efficiency
- **Random Forest**: Ensemble learning method

Feature engineering includes:
- Temporal features (day of week, time of day)
- Subject line analysis (length, personalization, question marks, etc.)
- Demographic features (age, gender distribution)
- Campaign metadata (dialog type, purpose, product)

## 📋 Project Structure

```
email-campaign-kpi-predictor/
├── app.py                     # Main Streamlit application
├── feature_engineering.py     # Feature engineering functions
├── model_training.py          # Model training and evaluation
├── subject_recommendation.py  # Subject line recommendation logic
├── visualizations.py          # Visualization functions
├── requirements.txt           # Python dependencies
├── delivery_data.csv          # Delivery-level data from Adobe Campaign
├── customer_data.csv          # Customer-level data from Adobe Campaign
└── README.md                  # This file
```

## 🧩 Core Components

### Feature Engineering Module
- Processes raw data from Adobe Campaign
- Creates derived features and time-based metrics
- Handles text analysis for subject lines
- Manages categorical encoding

### Model Training Module
- Trains multiple model types and selects the best performer
- Performs hyperparameter tuning with cross-validation
- Calculates feature importance
- Evaluates model performance

### Subject Recommendation Engine
- Uses clustering to identify successful patterns
- Extracts common elements from high-performing subject lines
- Provides data-driven recommendations for new campaigns

### Visualization Dashboard
- Creates interactive charts for campaign analysis
- Visualizes predicted performance improvements
- Shows historical trends and patterns
- Highlights feature importance

## 📈 Example Results

The application provides several key insights:

1. **Predicted Open Rate**: Baseline prediction for current settings
2. **Targeting Recommendations**: Optimal targeting to maximize open rate
3. **Subject Line Optimization**: Recommended subject line patterns and specific text
4. **Combined Improvement Potential**: Expected lift from implementing all recommendations
5. **Feature Importance**: Understanding which factors most impact open rates

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

*Note: This project is not officially affiliated with Adobe Campaign.*