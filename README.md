# Email Campaign KPI Predictor

## 📧 Advanced ML tool for predicting and optimizing email marketing campaigns
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.15.0-FF4B4B)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.6.2-006ACC)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-3.3.2-3DAD47)](https://lightgbm.readthedocs.io/)
[![Plotly](https://img.shields.io/badge/Plotly-5.9.0-3F4F75)](https://plotly.com/)

## 📋 Overview

Email Campaign KPI Predictor is a machine learning-powered tool for email marketers to predict and optimize campaign performance. Built specifically for Adobe Campaign data, this application helps marketing teams make data-driven decisions about targeting, subject lines, and delivery timing with clear metrics predictions:

- **For Subject Line Optimization**: Predict Open Rate only
- **For Targeting Optimization**: Predict Open Rate, Click Rate, and Optout Rate

## ✨ Features

- **Multi-Metric Predictive Analytics**:
  - Open Rate predictions for all scenarios
  - Click Rate predictions for targeting options
  - Optout Rate predictions for targeting options
- **Smart Recommendations**: 
  - Targeting optimization with complete KPI impact assessment
  - Subject line recommendations focusing on open rate optimization
  - Combined recommendations for maximum impact
- **Interactive Visualizations**:
  - Performance comparison charts
  - Feature importance analysis
  - Performance radar charts
  - Metric-specific historical trends
  - Company (Bolag) segment analysis
- **Comprehensive Reporting**:
  - Exportable predictions with detailed metrics breakdown
  - Downloadable campaign reports
  - Model performance metrics

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/6ogo/KPI-Predictor.git
cd KPI-Predictor
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
- `InternalName` (Delivery identifier)
- `Subject` (Email subject line)
- `Date` (Date and time of delivery)
- `Sendouts` (Total sendout count for delivery)
- `Opens` (Total Opens count in delivery)
- `Clicks` (Total Clicks count in delivery)
- `Optouts` (Total unsubscribe count in delivery)
- `Dialog`, `Syfte`, `Produkt` (Campaign metadata)

### Customer Level Data
File name: `customer_data.csv`
Required columns:
- `Primary key` (Customer identifier)
- `InternalName` (Delivery identifier to link with delivery data)
- `OptOut` (OptOut in delivery 1/0)
- `Open` (Opened delivery 1/0)
- `Click` (Clicked in delivery 1/0)
- `Gender` (Customer gender)
- `Age` (Customer age)
- `Bolag` (Customer company connection)

## 🚀 Usage

1. Place your data files (`delivery_data.csv` and `customer_data.csv`) in the application directory.

2. Run the Streamlit app:
```bash
streamlit run app.py
```

3. Access the app through your browser at `http://localhost:8501`

4. Use the interface to:
   - Input campaign parameters
   - View predictions and recommendations for:
     - Subject line (open rate only)
     - Targeting (open, click, and optout rates)
     - Combined approach (all metrics)
   - Explore performance insights by specific metrics
   - Export detailed reports and predictions

## 🔍 Model Information

The application trains separate models for different metrics:

- **Open Rate Model**: Predicts email open rates based on all features
- **Click Rate Model**: Predicts click rates based on targeting and content features
- **Optout Rate Model**: Predicts unsubscribe rates based on targeting and content features

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
├── multi_metric_model.py      # Models for different metrics
├── recommendations.py         # Recommendation generation
├── subject_recommendation.py  # Subject line optimization
├── visualizations.py          # Visualization functions
├── requirements.txt           # Python dependencies
├── delivery_data.csv          # Delivery-level data from Adobe Campaign
├── customer_data.csv          # Customer-level data from Adobe Campaign
└── README.md                  # This file
```

## 🧩 Core Components

### Multi-Metric Model Module
- Trains separate models for open rate, click rate, and optout rate
- Provides specialized predictions for different optimization scenarios
- Handles different feature importance for different metrics

### Feature Engineering Module
- Processes raw data from Adobe Campaign
- Creates derived features and time-based metrics
- Specializes in subject line analysis for open rate prediction
- Manages categorical encoding

### Recommendations Engine
- Delivers metric-specific optimizations:
  - Subject recommendations focus on open rate
  - Targeting recommendations provide complete KPI assessment
- Combines recommendations for maximum overall impact

### Visualization Dashboard
- Creates metric-specific visualizations
- Shows impact on different KPIs separately
- Provides comparative analysis across recommendations
- Visualizes historical trends by metric

## 📈 Example Results

The application provides several key insights:

1. **Predicted Open Rate for Subject Line Optimization**:
   - Current subject performance
   - Recommended subject with predicted open rate improvement
   - No predictions for click or optout rates (as subject primarily affects opens)

2. **Full KPI Predictions for Targeting Optimization**:
   - Predicted open rates
   - Predicted click rates
   - Predicted optout rates

3. **Combined Recommendations**:
   - Best targeting option with best subject line
   - Complete KPI impact assessment (open, click, optout)
   - Relative improvement over current campaign

4. **Customer Segment Analysis by Company (Bolag)**:
   - Open rates by company/organization
   - Click rates by company/organization
   - Optout rates by company/organization
   - Comparative analysis across top companies

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

*Note: This project is not officially affiliated with Adobe Campaign.*