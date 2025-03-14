import pandas as pd
import numpy as np
from datetime import datetime
import re
import logging
import traceback

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_subject_features(subject):
    """
    Extract features from email subject lines with enhanced error handling
    
    Parameters:
    - subject: Email subject line text
    
    Returns:
    - Dictionary of extracted features
    """
    # Default features (used when errors occur)
    default_features = {
        'length': 0,
        'has_personalization': 0,
        'has_question': 0,
        'has_numbers': 0,
        'has_uppercase_words': 0,
        'has_emoji': 0,
        'word_count': 0,
        'has_exclamation': 0,
        'has_special_chars': 0,
        'has_cta': 0,
        'has_urgency': 0,
        'has_benefits': 0,
        'is_short': 0,
        'is_medium': 0,
        'is_long': 0
    }
    
    # Handle non-string inputs
    if not isinstance(subject, str):
        logging.warning(f"Non-string subject encountered: {type(subject)}. Using default features.")
        if subject is None:
            return default_features
        try:
            # Try to convert to string
            subject = str(subject)
        except:
            return default_features
    
    # Empty string check
    if not subject.strip():
        return default_features
    
    try:
        features = {}
        
        # Basic features
        features['length'] = len(subject)
        
        # Check for personalization (Swedish and English)
        personalization_words = r'\b(your|you|du|din|ditt|dina)\b'
        features['has_personalization'] = 1 if re.search(personalization_words, subject.lower()) else 0
        
        # Question mark presence
        features['has_question'] = 1 if '?' in subject else 0
        
        # Numbers presence
        features['has_numbers'] = 1 if re.search(r'\d', subject) else 0
        
        # Uppercase words (for emphasis)
        features['has_uppercase_words'] = 1 if re.search(r'\b[A-Z]{2,}\b', subject) else 0
        
        # Emoji presence - with error handling for Unicode issues
        try:
            # Use a simplified emoji detection approach
            emoji_pattern = r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF]'
            features['has_emoji'] = 1 if re.search(emoji_pattern, subject) else 0
        except Exception as e:
            logging.warning(f"Error detecting emoji: {e}")
            features['has_emoji'] = 0
        
        # Word count
        features['word_count'] = len(subject.split())
        
        # Additional features
        # Exclamation mark presence
        features['has_exclamation'] = 1 if '!' in subject else 0
        
        # Special character presence (excluding common punctuation)
        special_chars = r'[#$%&*+<=>@^_`{|}~]'
        features['has_special_chars'] = 1 if re.search(special_chars, subject) else 0
        
        # Specific call-to-action words
        cta_words = r'\b(köp|buy|get|få|spara|save|discover|upptäck|click|klicka|check|se|läs|read)\b'
        features['has_cta'] = 1 if re.search(cta_words, subject.lower()) else 0
        
        # Urgency words
        urgency_words = r'\b(nu|now|idag|today|sista|last|limited|begränsat|snabbt|quick|fast|soon|snart)\b'
        features['has_urgency'] = 1 if re.search(urgency_words, subject.lower()) else 0
        
        # Benefit words
        benefit_words = r'\b(gratis|free|spara|save|extra|mer|more|bäst|best|exclusive|exklusiv|special|erbjudande|offer)\b'
        features['has_benefits'] = 1 if re.search(benefit_words, subject.lower()) else 0
        
        # Length categories
        features['is_short'] = 1 if len(subject) < 30 else 0
        features['is_medium'] = 1 if 30 <= len(subject) < 60 else 0
        features['is_long'] = 1 if len(subject) >= 60 else 0
        
        return features
    except Exception as e:
        logging.error(f"Error extracting subject features: {e}")
        logging.error(traceback.format_exc())
        # Return default features in case of errors
        return default_features

def enhanced_feature_engineering(delivery_df, customer_df):
    """Enhanced feature engineering for email campaign data with robust error handling"""
    logging.info("Starting enhanced feature engineering")
    
    try:
        # Create copies to avoid modifying originals
        delivery_df = delivery_df.copy()
        customer_df = customer_df.copy()
        
        # --- Standardize column names ---
        # Convert all column names to lowercase for consistency
        delivery_df.columns = [col.lower() for col in delivery_df.columns]
        if customer_df is not None and len(customer_df) > 0:
            customer_df.columns = [col.lower() for col in customer_df.columns]
        
        # Log column names after standardization
        logging.info(f"Standardized delivery columns: {delivery_df.columns.tolist()}")
        
        # --- Handle subject column ---
        # Ensure 'subject' column exists
        if 'subject' not in delivery_df.columns:
            logging.warning("Subject column not found. Creating empty 'subject' column.")
            delivery_df['subject'] = ""
        
        # --- Time-based features ---
        date_col = None
        for col_name in ['date', 'Date', 'DATE']:
            if col_name.lower() in delivery_df.columns:
                date_col = col_name.lower()
                break
                
        if date_col:
            try:
                delivery_df['date'] = pd.to_datetime(delivery_df[date_col], errors='coerce')
                delivery_df['day_of_week'] = delivery_df['date'].dt.dayofweek
                delivery_df['hour_of_day'] = delivery_df['date'].dt.hour
                delivery_df['is_weekend'] = delivery_df['day_of_week'].isin([5, 6]).astype(int)
                delivery_df['month'] = delivery_df['date'].dt.month
                delivery_df['is_morning'] = (delivery_df['hour_of_day'] >= 6) & (delivery_df['hour_of_day'] < 12)
                delivery_df['is_afternoon'] = (delivery_df['hour_of_day'] >= 12) & (delivery_df['hour_of_day'] < 18)
                delivery_df['is_evening'] = (delivery_df['hour_of_day'] >= 18) & (delivery_df['hour_of_day'] < 22)
                delivery_df['is_night'] = (delivery_df['hour_of_day'] >= 22) | (delivery_df['hour_of_day'] < 6)
                
                # Convert to numeric values (0/1 instead of True/False)
                for col in ['is_morning', 'is_afternoon', 'is_evening', 'is_night']:
                    delivery_df[col] = delivery_df[col].astype(int)
            except Exception as e:
                logging.error(f"Error processing Date column: {e}")
                logging.error(traceback.format_exc())
                # Add default time features
                for col in ['day_of_week', 'hour_of_day', 'is_weekend', 'month']:
                    if col not in delivery_df.columns:
                        delivery_df[col] = 0
                
                # Add time of day indicators
                delivery_df['is_morning'] = 1
                delivery_df['is_afternoon'] = 0
                delivery_df['is_evening'] = 0
                delivery_df['is_night'] = 0
        else:
            # Add default time features if Date column is missing
            logging.warning("Date column not found. Adding default time features.")
            delivery_df['day_of_week'] = 0
            delivery_df['hour_of_day'] = 9  # Default to morning
            delivery_df['is_weekend'] = 0
            delivery_df['month'] = datetime.now().month
            delivery_df['is_morning'] = 1
            delivery_df['is_afternoon'] = 0
            delivery_df['is_evening'] = 0
            delivery_df['is_night'] = 0
        
        # --- Subject line features ---
        if 'subject' in delivery_df.columns:
            try:
                # Clean up subject column - replace NaN with empty string
                delivery_df['subject'] = delivery_df['subject'].fillna('').astype(str)
                
                # Apply enhanced subject features extraction (with error handling per row)
                logging.info("Extracting subject line features")
                subject_features_list = []
                
                for idx, subj in enumerate(delivery_df['subject']):
                    try:
                        # Extract features for each subject
                        features = extract_subject_features(subj)
                        subject_features_list.append(features)
                    except Exception as e:
                        logging.error(f"Error extracting features for subject at index {idx}: {e}")
                        # Use default features on error
                        subject_features_list.append({
                            'length': 0, 
                            'has_personalization': 0, 
                            'has_question': 0,
                            'has_numbers': 0, 
                            'has_uppercase_words': 0, 
                            'has_emoji': 0, 
                            'word_count': 0,
                            'has_exclamation': 0,
                            'has_special_chars': 0,
                            'has_cta': 0,
                            'has_urgency': 0,
                            'has_benefits': 0,
                            'is_short': 0,
                            'is_medium': 0,
                            'is_long': 0
                        })
                
                # Convert list of dictionaries to DataFrame
                subject_features = pd.DataFrame(subject_features_list)
                
                # Join the subject features to the main dataframe
                for col in subject_features.columns:
                    delivery_df[col] = subject_features[col]
                
                # Create additional derived features
                # Average word length
                delivery_df['avg_word_length'] = delivery_df['subject'].apply(
                    lambda x: np.mean([len(w) for w in str(x).split()]) if len(str(x).split()) > 0 else 0
                )
                
                # First word is verb (simplified approximation)
                common_verbs = ['check', 'discover', 'get', 'see', 'find', 'try', 'read', 'buy', 'learn', 'start', 
                               'köp', 'se', 'hitta', 'prova', 'läs', 'börja', 'upptäck', 'få']
                delivery_df['starts_with_verb'] = delivery_df['subject'].apply(
                    lambda x: 1 if str(x).split()[0].lower() in common_verbs and len(str(x).split()) > 0 else 0
                )
                
            except Exception as e:
                logging.error(f"Error processing subject features as a whole: {e}")
                logging.error(traceback.format_exc())
                
                # Add default subject features
                default_features = [
                    'length', 'has_personalization', 'has_question', 
                    'has_numbers', 'has_uppercase_words', 'has_emoji', 'word_count',
                    'has_exclamation', 'has_special_chars', 'has_cta',
                    'has_urgency', 'has_benefits', 'is_short', 'is_medium', 'is_long'
                ]
                
                for feature in default_features:
                    delivery_df[feature] = 0
                
                # Calculate length and word count directly
                delivery_df['length'] = delivery_df['subject'].fillna('').str.len()
                delivery_df['word_count'] = delivery_df['subject'].fillna('').apply(lambda x: len(str(x).split()))
                delivery_df['avg_word_length'] = 0
                delivery_df['starts_with_verb'] = 0
                
        else:
            logging.warning("Subject column not found. Adding default subject feature values.")
            # Add default subject features
            default_features = [
                'length', 'has_personalization', 'has_question', 
                'has_numbers', 'has_uppercase_words', 'has_emoji', 'word_count',
                'has_exclamation', 'has_special_chars', 'has_cta',
                'has_urgency', 'has_benefits', 'is_short', 'is_medium', 'is_long',
                'avg_word_length', 'starts_with_verb'
            ]
            
            for feature in default_features:
                delivery_df[feature] = 0
        
        # --- Customer demographics features ---
        if customer_df is not None and len(customer_df) > 0:
            # Check for bolag column in different cases
            bolag_col = None
            for col_name in ['bolag', 'Bolag', 'BOLAG']:
                if col_name.lower() in customer_df.columns:
                    bolag_col = col_name.lower()
                    break
                    
            # Check for internalname column in different cases
            internal_col = None
            for col_name in ['internalname', 'InternalName', 'INTERNALNAME']:
                if col_name.lower() in customer_df.columns:
                    internal_col = col_name.lower()
                    break
                    
            if bolag_col and internal_col:
                try:
                    logging.info("Processing customer demographics")
                    
                    # Check Age column
                    age_col = None
                    for col_name in ['age', 'Age', 'AGE']:
                        if col_name.lower() in customer_df.columns:
                            age_col = col_name.lower()
                            break
                    
                    if age_col:
                        # Create aggregated customer features by delivery
                        try:
                            # Convert age to numeric first
                            customer_df[age_col] = pd.to_numeric(customer_df[age_col], errors='coerce')
                            
                            # Group by internal name and calculate age statistics
                            customer_aggs = customer_df.groupby(internal_col).agg({
                                age_col: ['mean', 'median', 'std'],
                            })
                            
                            # Flatten multi-index columns
                            customer_aggs.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in customer_aggs.columns]
                            
                            # Rename for clarity
                            column_renames = {
                                f'{age_col}_mean': 'avg_age',
                                f'{age_col}_median': 'median_age',
                                f'{age_col}_std': 'std_age'
                            }
                            
                            customer_aggs.rename(columns=column_renames, inplace=True)
                            customer_aggs.reset_index(inplace=True)
                            
                            # Merge with delivery data
                            delivery_df = delivery_df.merge(customer_aggs, on=internal_col, how='left')
                        except Exception as e:
                            logging.error(f"Error processing age demographics: {e}")
                            logging.error(traceback.format_exc())
                    
                    # Calculate gender ratio if Gender column exists
                    gender_col = None
                    for col_name in ['gender', 'Gender', 'GENDER']:
                        if col_name.lower() in customer_df.columns:
                            gender_col = col_name.lower()
                            break
                            
                    if gender_col:
                        try:
                            gender_aggs = customer_df.groupby(internal_col).apply(
                                lambda x: (x[gender_col].str.lower() == 'f').mean() * 100
                            ).reset_index(name='pct_women')
                            
                            # Merge with delivery data
                            delivery_df = delivery_df.merge(gender_aggs, on=internal_col, how='left')
                        except Exception as e:
                            logging.error(f"Error processing gender demographics: {e}")
                            logging.error(traceback.format_exc())
                
                except Exception as e:
                    logging.error(f"Error in overall customer demographics processing: {e}")
                    logging.error(traceback.format_exc())
            
            # Compute open/click/optout rates from customer-level data if needed
            try:
                # Find open, click, optout columns in customer data
                open_col = None
                click_col = None
                optout_col = None
                
                for col_name in ['open', 'Open', 'OPEN']:
                    if col_name.lower() in customer_df.columns:
                        open_col = col_name.lower()
                        break
                        
                for col_name in ['click', 'Click', 'CLICK']:
                    if col_name.lower() in customer_df.columns:
                        click_col = col_name.lower()
                        break
                        
                for col_name in ['optout', 'Optout', 'OptOut', 'OPTOUT']:
                    if col_name.lower() in customer_df.columns:
                        optout_col = col_name.lower()
                        break
                
                if internal_col and all(col is not None for col in [open_col, click_col, optout_col]):
                    # Aggregate customer-level metrics to delivery level
                    customer_metrics = customer_df.groupby(internal_col).agg({
                        open_col: 'mean',
                        click_col: 'mean',
                        optout_col: 'mean'
                    })
                    
                    # Rename columns
                    customer_metrics.columns = ['customer_open_rate', 'customer_click_rate', 'customer_optout_rate']
                    customer_metrics.reset_index(inplace=True)
                    
                    # Multiply by 100 to get percentages
                    for col in ['customer_open_rate', 'customer_click_rate', 'customer_optout_rate']:
                        customer_metrics[col] = customer_metrics[col] * 100
                    
                    # Merge with delivery data
                    delivery_df = delivery_df.merge(customer_metrics, on=internal_col, how='left')
            except Exception as e:
                logging.error(f"Error computing customer-level metrics: {e}")
        
        # Fill missing values with reasonable defaults
        demographic_defaults = {
            'avg_age': 40,
            'median_age': 38,
            'std_age': 15,
            'pct_women': 50,
            'customer_open_rate': 20,
            'customer_click_rate': 3,
            'customer_optout_rate': 0.2
        }
        
        delivery_df.fillna(demographic_defaults, inplace=True)
        
        # --- Calculate engagement metrics if they don't exist ---
        try:
            # Look for sendouts, opens, clicks, optouts in various capitalizations
            fields = ['sendouts', 'opens', 'clicks', 'optouts']
            
            # Dictionary to store found column names
            found_cols = {}
            
            # Find columns with different capitalization
            for field in fields:
                for col in delivery_df.columns:
                    if col.lower() == field:
                        found_cols[field] = col
                        break
            
            # Calculate open_rate if not present
            if 'open_rate' not in delivery_df.columns and 'opens' in found_cols and 'sendouts' in found_cols:
                # Convert to numeric first
                delivery_df[found_cols['opens']] = pd.to_numeric(delivery_df[found_cols['opens']], errors='coerce').fillna(0)
                delivery_df[found_cols['sendouts']] = pd.to_numeric(delivery_df[found_cols['sendouts']], errors='coerce').fillna(0)
                
                # Calculate open rate (avoid division by zero)
                delivery_df['open_rate'] = np.where(
                    delivery_df[found_cols['sendouts']] > 0,
                    (delivery_df[found_cols['opens']] / delivery_df[found_cols['sendouts']]) * 100,
                    0
                )
            
            # Calculate click_rate if not present
            if 'click_rate' not in delivery_df.columns and 'clicks' in found_cols and 'opens' in found_cols:
                # Convert to numeric first
                delivery_df[found_cols['clicks']] = pd.to_numeric(delivery_df[found_cols['clicks']], errors='coerce').fillna(0)
                
                # Calculate click rate (avoid division by zero)
                delivery_df['click_rate'] = np.where(
                    delivery_df[found_cols['opens']] > 0,
                    (delivery_df[found_cols['clicks']] / delivery_df[found_cols['opens']]) * 100,
                    0
                )
            
            # Calculate optout_rate if not present
            if 'optout_rate' not in delivery_df.columns and 'optouts' in found_cols and 'opens' in found_cols:
                # Convert to numeric first
                delivery_df[found_cols['optouts']] = pd.to_numeric(delivery_df[found_cols['optouts']], errors='coerce').fillna(0)
                
                # Calculate optout rate (avoid division by zero)
                delivery_df['optout_rate'] = np.where(
                    delivery_df[found_cols['opens']] > 0,
                    (delivery_df[found_cols['optouts']] / delivery_df[found_cols['opens']]) * 100,
                    0
                )
        except Exception as e:
            logging.error(f"Error calculating rates: {e}")
            logging.error(traceback.format_exc())
        
        # Make sure rate columns exist
        rate_cols = ['open_rate', 'click_rate', 'optout_rate']
        rate_defaults = [20.0, 3.0, 0.2]  # Default values
        
        for col, default in zip(rate_cols, rate_defaults):
            if col not in delivery_df.columns:
                logging.warning(f"Adding default values for missing {col} column")
                delivery_df[col] = default
        
        # Replace infinities and NaNs
        delivery_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Fill missing rates with median or default values
        for col, default in zip(rate_cols, rate_defaults):
            median_value = delivery_df[col].median()
            delivery_df[col].fillna(median_value if not pd.isna(median_value) else default, inplace=True)
            
            # Clip values to reasonable ranges
            delivery_df[col] = delivery_df[col].clip(0, 100)
        
        logging.info(f"Enhanced feature engineering completed. Shape: {delivery_df.shape}")
        return delivery_df
        
    except Exception as e:
        logging.error(f"Error in enhanced_feature_engineering: {e}")
        logging.error(traceback.format_exc())
        # Return the original dataframe if processing fails
        logging.info("Returning original delivery_df due to error")
        return delivery_df