def train_multi_metric_models(delivery_df, customer_df):
    """Train separate models for predicting open rate, click rate, and optout rate"""
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score, GridSearchCV
    from sklearn.metrics import mean_absolute_error
    import xgboost as xgb
    from lightgbm import LGBMRegressor
    
    # Process data with enhanced feature engineering
    from feature_engineering import enhanced_feature_engineering
    processed_data = enhanced_feature_engineering(delivery_df, customer_df)
    
    # Calculate metrics if not already present
    if 'open_rate' not in processed_data.columns:
        processed_data['open_rate'] = (processed_data['Opens'] / processed_data['Utskick']) * 100
    
    if 'click_rate' not in processed_data.columns:
        processed_data['click_rate'] = (processed_data['Clicks'] / processed_data['Opens']) * 100
    
    if 'optout_rate' not in processed_data.columns:
        processed_data['optout_rate'] = (processed_data['Optout'] / processed_data['Opens']) * 100
    
    # Replace infinities and NaNs with 0 for click_rate and optout_rate (when Opens = 0)
    processed_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    processed_data.fillna({
        'click_rate': 0,
        'optout_rate': 0
    }, inplace=True)
    
    # Define features for all models
    features = processed_data.drop(['open_rate', 'click_rate', 'optout_rate', 
                                  'InternalName', 'subject', 'contact_date'], 
                                  axis=1, errors='ignore')
    
    # Define targets
    targets = {
        'open_rate': processed_data['open_rate'],
        'click_rate': processed_data['click_rate'],
        'optout_rate': processed_data['optout_rate']
    }
    
    # Identify feature types
    categorical_features = features.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_features = features.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Create preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', StandardScaler(), numerical_features)
        ]
    )
    
    # Define model parameters
    xgb_params = {
        'n_estimators': 100,
        'learning_rate': 0.05,
        'max_depth': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }
    
    # Train models for each target metric
    models = {}
    performance = {}
    
    for metric, target in targets.items():
        print(f"Training model for {metric}...")
        
        # Create pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', xgb.XGBRegressor(**xgb_params))
        ])
        
        # Fit model
        pipeline.fit(features, target)
        
        # Evaluate with cross-validation
        cv_scores = cross_val_score(
            pipeline, features, target,
            cv=5, scoring='neg_mean_absolute_error'
        )
        
        # Store model and performance
        models[metric] = pipeline
        performance[metric] = {
            'mae': -np.mean(cv_scores),
            'std': np.std(cv_scores)
        }
        
        print(f"  {metric} model MAE: {-np.mean(cv_scores):.4f}")
    
    # For subject line optimization, we only need the open_rate model
    # Create a specialized model for subject line features only
    subject_features = [
        col for col in features.columns 
        if col.startswith('length') or col.startswith('has_') or col.startswith('word_count')
    ]
    
    if subject_features:
        # If we have subject-specific features, train a specialized model
        subject_pipeline = Pipeline([
            ('preprocessor', StandardScaler()),
            ('regressor', xgb.XGBRegressor(**xgb_params))
        ])
        
        subject_pipeline.fit(features[subject_features], targets['open_rate'])
        models['subject_open_rate'] = subject_pipeline
    
    return {
        'models': models,
        'performance': performance,
        'feature_names': features.columns.tolist(),
        'categorical_values': {
            col: features[col].unique().tolist() 
            for col in categorical_features
        }
    }


def predict_metrics(input_data, models, metrics=None):
    """
    Predict multiple metrics based on input data
    
    Parameters:
    - input_data: DataFrame with input features
    - models: Dict of trained models for different metrics
    - metrics: List of metrics to predict (default: all available)
    
    Returns:
    - Dictionary with predicted values for each metric
    """
    if metrics is None:
        metrics = list(models.keys())
    
    predictions = {}
    
    for metric in metrics:
        if metric in models:
            predictions[metric] = models[metric].predict(input_data)[0]
    
    return predictions