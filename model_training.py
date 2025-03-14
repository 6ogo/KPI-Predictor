def train_and_evaluate_models(X, y):
    """Train multiple models and select the best one based on cross-validation"""
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score, GridSearchCV
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import xgboost as xgb
    from lightgbm import LGBMRegressor
    from sklearn.ensemble import RandomForestRegressor
    import matplotlib.pyplot as plt
    
    # Identify feature types
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Create preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', StandardScaler(), numerical_features)
        ]
    )
    
    # Define models to test
    models = {
        'XGBoost': xgb.XGBRegressor(objective='reg:squarederror'),
        'LightGBM': LGBMRegressor(),
        'RandomForest': RandomForestRegressor(random_state=42)
    }
    
    # Model parameters for grid search
    param_grids = {
        'XGBoost': {
            'regressor__n_estimators': [50, 100, 200],
            'regressor__learning_rate': [0.01, 0.05, 0.1],
            'regressor__max_depth': [3, 5, 7]
        },
        'LightGBM': {
            'regressor__n_estimators': [50, 100, 200],
            'regressor__learning_rate': [0.01, 0.05, 0.1],
            'regressor__num_leaves': [31, 63, 127]
        },
        'RandomForest': {
            'regressor__n_estimators': [50, 100, 200],
            'regressor__max_depth': [None, 10, 20],
            'regressor__min_samples_split': [2, 5, 10]
        }
    }
    
    # Store results
    results = {}
    best_models = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"Training {name}...")
        
        # Create pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        
        # Perform GridSearch
        grid_search = GridSearchCV(
            pipeline,
            param_grids[name],
            cv=5,
            scoring='neg_mean_absolute_error',
            n_jobs=-1
        )
        
        # Fit grid search
        grid_search.fit(X, y)
        
        # Store best model
        best_models[name] = grid_search.best_estimator_
        
        # Get cross-validation results
        cv_results = cross_val_score(
            grid_search.best_estimator_,
            X, y,
            cv=5,
            scoring='neg_mean_absolute_error'
        )
        
        # Store results
        results[name] = {
            'mae': -np.mean(cv_results),
            'best_params': grid_search.best_params_
        }
        
        print(f"{name} MAE: {-np.mean(cv_results):.4f}")
    
    # Determine best model
    best_model_name = min(results, key=lambda x: results[x]['mae'])
    best_model = best_models[best_model_name]
    
    print(f"\nBest model: {best_model_name} with MAE: {results[best_model_name]['mae']:.4f}")
    print(f"Best parameters: {results[best_model_name]['best_params']}")
    
    # Feature importance analysis (for tree-based models)
    if hasattr(best_model.named_steps['regressor'], 'feature_importances_'):
        # Get feature names after preprocessing
        feature_names = []
        
        # Get the OneHotEncoder
        ohe = best_model.named_steps['preprocessor'].transformers_[0][1]
        
        # Get the categorical feature names after one-hot encoding
        if hasattr(ohe, 'get_feature_names_out'):
            cat_feature_names = ohe.get_feature_names_out(categorical_features)
            feature_names.extend(cat_feature_names)
        
        # Add numerical feature names
        feature_names.extend(numerical_features)
        
        # Get feature importances
        importances = best_model.named_steps['regressor'].feature_importances_
        
        # Match feature names and importances (might need adjustment based on actual pipeline)
        if len(feature_names) == len(importances):
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            print("\nTop 10 important features:")
            print(feature_importance.head(10))
        
    return best_model, results