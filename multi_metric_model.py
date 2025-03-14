import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import logging
import traceback
import re

# Setup detailed logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import functions from model_metadata
from model_metadata import get_model_version, save_model_metadata

def clean_subject_text(text):
    """
    Clean and normalize subject text to handle special characters like en dash (–)
    
    Parameters:
    - text: The subject line text to clean
    
    Returns:
    - Cleaned text with special characters normalized
    """
    if not isinstance(text, str):
        return ""
    
    try:
        # Replace en dash (–) with regular hyphen (-)
        text = text.replace('–', '-')
        
        # Replace em dash (—) with regular hyphen (-)
        text = text.replace('—', '-')
        
        # Replace other potentially problematic characters
        text = text.replace('\n', ' ').replace('\r', ' ')
        
        # Remove control characters
        text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    except Exception as e:
        logging.error(f"Error cleaning subject text: {e}")
        return ""

def safe_extract_features(extract_fn, subject):
    """
    Safely extract features from a subject line with detailed error handling
    
    Parameters:
    - extract_fn: Feature extraction function
    - subject: Subject line text
    
    Returns:
    - Dictionary of features or default features on error
    """
    default_features = {
        'length': 0,
        'has_personalization': 0,
        'has_question': 0,
        'has_numbers': 0,
        'has_uppercase_words': 0,
        'has_emoji': 0,
        'word_count': 0
    }
    
    try:
        # First clean the subject text
        cleaned_subject = clean_subject_text(subject)
        
        # Then extract features from the cleaned text
        return extract_fn(cleaned_subject)
    except Exception as e:
        logging.error(f"Feature extraction error: {e}")
        logging.error(f"Problem subject: '{subject}'")
        logging.error(traceback.format_exc())
        return default_features

def train_multi_metric_models(delivery_df, customer_df):
    """Train separate models for predicting open rate, click rate, and optout rate"""
    logging.info("Starting model training process")
    
    try:
        # Process data with enhanced error handling
        processed_data = process_data_directly(delivery_df, customer_df)
        
        # Calculate metrics if not already present
        if 'open_rate' not in processed_data.columns:
            processed_data['open_rate'] = (processed_data['Opens'] / processed_data['Sendouts']) * 100
        
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
        columns_to_drop = ['open_rate', 'click_rate', 'optout_rate', 
                           'InternalName', 'Subject', 'subject', 'Date', 
                           'Opens', 'Clicks', 'Optout', 'Sendouts']
        
        features = processed_data.drop([col for col in columns_to_drop if col in processed_data.columns], 
                                      axis=1, errors='ignore')
        
        # Check for duplicate columns and remove them
        if len(features.columns) != len(set(features.columns)):
            duplicate_cols = features.columns[features.columns.duplicated()].tolist()
            logging.info(f"Found duplicate columns: {duplicate_cols}")
            features = features.loc[:, ~features.columns.duplicated()]
        
        # Define targets
        targets = {
            'open_rate': processed_data['open_rate'],
            'click_rate': processed_data['click_rate'],
            'optout_rate': processed_data['optout_rate']
        }
        
        # Identify feature types
        categorical_features = features.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_features = features.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Log feature information
        logging.info(f"Using {len(categorical_features)} categorical features: {categorical_features}")
        logging.info(f"Using {len(numerical_features)} numerical features: {numerical_features}")
        
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
            logging.info(f"Training model for {metric}...")
            
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
            
            logging.info(f"  {metric} model MAE: {-np.mean(cv_scores):.4f}")
        
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
        
        logging.info("Model training completed successfully!")
        
        return {
            'models': models,
            'performance': performance,
            'feature_names': features.columns.tolist(),
            'categorical_values': {
                col: features[col].unique().tolist() 
                for col in categorical_features
            }
        }
    except Exception as e:
        logging.error(f"Error in train_multi_metric_models: {e}")
        logging.error(traceback.format_exc())
        raise Exception(f"Model training failed: {e}")

def process_data_directly(delivery_df, customer_df):
    """
    Process the data directly with enhanced error handling for special characters
    """
    logging.info("Starting process_data_directly function")
    
    try:
        # Make copies to avoid modifying originals
        delivery_df = delivery_df.copy()
        customer_df = customer_df.copy()
        
        # Log column names for debugging
        logging.info(f"Delivery columns: {delivery_df.columns.tolist()}")
        
        # CRITICAL FIX: Handle the subject column with special character cleaning
        if 'subject' not in delivery_df.columns and 'Subject' in delivery_df.columns:
            logging.info("Converting 'Subject' to lowercase 'subject' with cleaning")
            # Clean each subject line while creating the lowercase column
            delivery_df['subject'] = delivery_df['Subject'].apply(clean_subject_text)
        elif 'subject' in delivery_df.columns:
            logging.info("Cleaning existing 'subject' column")
            # Clean the existing subject column
            delivery_df['subject'] = delivery_df['subject'].apply(clean_subject_text)
        else:
            logging.warning("No subject column found! Creating empty 'subject' column")
            delivery_df['subject'] = ""
        
        # Basic time features from Date if available
        if 'Date' in delivery_df.columns:
            try:
                delivery_df['Date'] = pd.to_datetime(delivery_df['Date'])
                delivery_df['day_of_week'] = delivery_df['Date'].dt.dayofweek
                delivery_df['hour_of_day'] = delivery_df['Date'].dt.hour
                delivery_df['is_weekend'] = delivery_df['day_of_week'].isin([5, 6]).astype(int)
            except Exception as e:
                logging.warning(f"Could not process Date column: {e}")
                # Add default values
                delivery_df['day_of_week'] = 0
                delivery_df['hour_of_day'] = 12
                delivery_df['is_weekend'] = 0
        else:
            # Add default values if Date column is missing
            delivery_df['day_of_week'] = 0
            delivery_df['hour_of_day'] = 12
            delivery_df['is_weekend'] = 0
        
        # Add subject features - with enhanced error handling
        subject_feature_names = ['length', 'has_personalization', 'has_question', 
                               'has_numbers', 'has_uppercase_words', 'has_emoji', 'word_count']
        
        # Import feature extraction function
        from feature_engineering import extract_subject_features
        
        # CRITICAL FIX: Use safe feature extraction with detailed error handling
        logging.info("Generating subject features with safer extraction")
        try:
            # Process each subject line individually with error handling
            subject_features_list = []
            
            for idx, subj in enumerate(delivery_df['subject']):
                try:
                    # Try to extract features from each subject line
                    features = safe_extract_features(extract_subject_features, subj)
                    subject_features_list.append(features)
                except Exception as e:
                    logging.error(f"Error extracting features for subject at index {idx}: {e}")
                    # Use default features on error
                    subject_features_list.append({
                        'length': 0, 'has_personalization': 0, 'has_question': 0,
                        'has_numbers': 0, 'has_uppercase_words': 0, 'has_emoji': 0, 'word_count': 0
                    })
            
            # Convert list of dictionaries to DataFrame
            subject_features = pd.DataFrame(subject_features_list)
            
            # Add each feature to the dataframe
            for feature in subject_feature_names:
                if feature in subject_features.columns:
                    delivery_df[feature] = subject_features[feature]
                else:
                    # Add default values for missing features
                    if feature == 'length':
                        delivery_df[feature] = delivery_df['subject'].fillna("").str.len()
                    elif feature == 'word_count':
                        delivery_df[feature] = delivery_df['subject'].fillna("").str.split().str.len()
                    else:
                        delivery_df[feature] = 0  # Default for boolean features
        except Exception as e:
            logging.error(f"Error in subject feature extraction: {e}")
            logging.error(traceback.format_exc())
            # Add default values for all subject features as a fallback
            for feature in subject_feature_names:
                delivery_df[feature] = 0
        
        # Check that subject features were added correctly
        for feature in subject_feature_names:
            if feature not in delivery_df.columns:
                logging.error(f"Subject feature '{feature}' was not added. Adding default values.")
                delivery_df[feature] = 0
        
        # Calculate aggregate customer demographics if available
        if 'Bolag' in customer_df.columns:
            # Check Age column
            age_col = None
            for col_name in ['Age', 'age', 'AGE']:
                if col_name in customer_df.columns:
                    age_col = col_name
                    break
                    
            if age_col:
                # We can calculate age stats
                customer_aggs = customer_df.groupby('InternalName').agg({
                    age_col: ['mean', 'std', 'min', 'max']
                })
                
                # Flatten multi-level column index
                customer_aggs.columns = ['_'.join(col).strip() for col in customer_aggs.columns.values]
                customer_aggs.reset_index(inplace=True)
                
                # Rename columns for clarity
                age_mean_col = f"{age_col}_mean"
                age_std_col = f"{age_col}_std"
                age_min_col = f"{age_col}_min"
                age_max_col = f"{age_col}_max"
                
                column_renames = {
                    age_mean_col: 'avg_age',
                    age_std_col: 'std_age',
                    age_min_col: 'min_age',
                    age_max_col: 'max_age'
                }
                
                customer_aggs.rename(columns=column_renames, inplace=True)
                
                # Merge with delivery data
                delivery_df = delivery_df.merge(customer_aggs, on='InternalName', how='left')
            
            # Check Gender column
            gender_col = None
            for col_name in ['Gender', 'gender', 'GENDER']:
                if col_name in customer_df.columns:
                    gender_col = col_name
                    break
                    
            if gender_col:
                # Calculate percentage of women
                gender_aggs = customer_df.groupby('InternalName').apply(
                    lambda x: (x[gender_col].str.lower() == 'f').mean() * 100 if len(x) > 0 else 50
                ).reset_index(name='pct_women')
                
                # Merge with delivery data
                delivery_df = delivery_df.merge(gender_aggs, on='InternalName', how='left')
            
            # Fill missing values with reasonable defaults
            delivery_df.fillna({
                'avg_age': 40,
                'std_age': 15,
                'min_age': 18,
                'max_age': 80,
                'pct_women': 50
            }, inplace=True)
        else:
            # Add default demographics if not available
            if 'avg_age' not in delivery_df.columns:
                delivery_df['avg_age'] = 40
            if 'std_age' not in delivery_df.columns:
                delivery_df['std_age'] = 15
            if 'min_age' not in delivery_df.columns:
                delivery_df['min_age'] = 18
            if 'max_age' not in delivery_df.columns:
                delivery_df['max_age'] = 80
            if 'pct_women' not in delivery_df.columns:
                delivery_df['pct_women'] = 50
        
        # Map Dialog, Syfte, and Produkt to expected lowercase names if needed
        column_mappings = {
            'Dialog': 'dialog',
            'Syfte': 'syfte',
            'Produkt': 'product'
        }
        
        for orig_col, new_col in column_mappings.items():
            if orig_col in delivery_df.columns and new_col not in delivery_df.columns:
                delivery_df[new_col] = delivery_df[orig_col]
        
        # Handle any missing columns that the model might expect
        expected_columns = ['dialog', 'syfte', 'product', 'county', 'day_of_week', 'hour_of_day', 'is_weekend']
        
        for col in expected_columns:
            if col not in delivery_df.columns:
                delivery_df[col] = 'Unknown'
        
        logging.info(f"Processed data shape: {delivery_df.shape}")
        return delivery_df
    except Exception as e:
        logging.error(f"Error in process_data_directly: {e}")
        logging.error(traceback.format_exc())
        raise Exception(f"Data processing failed: {e}")

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
            try:
                predictions[metric] = models[metric].predict(input_data)[0]
            except Exception as e:
                logging.error(f"Error predicting {metric}: {e}")
                # Fall back to a default value
                predictions[metric] = 0.0
    
    return predictions

def enhanced_train_multi_metric_models(delivery_df, customer_df):
    """Enhanced version of train_multi_metric_models that adds versioning and metadata"""
    try:
        # Log detailed information about the data
        logging.info(f"Starting enhanced training with {len(delivery_df)} deliveries")
        logging.info(f"Delivery columns: {delivery_df.columns.tolist()}")
        
        if 'Subject' in delivery_df.columns:
            # Check for potential problematic subjects
            logging.info("Checking subject lines for potential issues")
            sample_subjects = delivery_df['Subject'].head(10).tolist()
            for i, subj in enumerate(sample_subjects):
                if isinstance(subj, str) and ('–' in subj or '—' in subj):
                    logging.info(f"Found en/em dash in subject {i}: '{subj}'")
        
        # Call the function directly
        model_results = train_multi_metric_models(delivery_df, customer_df)
        
        # Add version using imported function from model_metadata
        model_results['version'] = get_model_version()
        
        # Add data stats
        model_results['num_campaigns'] = len(delivery_df)
        
        # Save metadata using imported function from model_metadata
        save_model_metadata(model_results)
        
        return model_results
    except Exception as e:
        logging.error(f"Enhanced training failed: {e}")
        logging.error(traceback.format_exc())
        raise Exception(f"Enhanced training failed: {e}")