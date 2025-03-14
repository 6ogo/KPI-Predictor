import pandas as pd
import numpy as np
from datetime import datetime
import re
import logging
from sklearn.feature_extraction.text import TfidfVectorizer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_subject_features(subject):
    """
    Extract features from email subject lines
    
    Parameters:
    - subject: Email subject line text
    
    Returns:
    - Dictionary of extracted features
    """
    # Handle non-string inputs
    if not isinstance(subject, str):
        logging.warning(f"Non-string subject encountered: {type(subject)}. Using default features.")
        return {
            'length': 0,
            'has_personalization': 0,
            'has_question': 0,
            'has_numbers': 0,
            'has_uppercase_words': 0,
            'has_emoji': 0,
            'word_count': 0
        }
    
    # Empty string check
    if not subject.strip():
        return {
            'length': 0,
            'has_personalization': 0,
            'has_question': 0,
            'has_numbers': 0,
            'has_uppercase_words': 0,
            'has_emoji': 0,
            'word_count': 0
        }
    
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
        
        # Emoji presence - extended range to catch more emoji characters
        emoji_pattern = r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002702-\U000027B0\U000024C2-\U0001F251]'
        features['has_emoji'] = 1 if re.search(emoji_pattern, subject) else 0
        
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
        # Return basic features in case of errors
        return {
            'length': len(subject) if isinstance(subject, str) else 0,
            'has_personalization': 0,
            'has_question': 0,
            'has_numbers': 0,
            'has_uppercase_words': 0,
            'has_emoji': 0,
            'word_count': len(subject.split()) if isinstance(subject, str) else 0
        }

def enhanced_feature_engineering(delivery_df, customer_df):
    """Enhanced feature engineering for email campaign data"""
    logging.info("Starting enhanced feature engineering")
    
    try:
        # Create copies to avoid modifying originals
        delivery_df = delivery_df.copy()
        customer_df = customer_df.copy()
        
        # --- Standardize column names ---
        # Ensure 'subject' column exists (lowercase)
        if 'subject' not in delivery_df.columns and 'Subject' in delivery_df.columns:
            delivery_df['subject'] = delivery_df['Subject']
            logging.info("Created lowercase 'subject' column from 'Subject'")
        
        # --- Time-based features ---
        if 'Date' in delivery_df.columns:
            try:
                delivery_df['Date'] = pd.to_datetime(delivery_df['Date'])
                delivery_df['day_of_week'] = delivery_df['Date'].dt.dayofweek
                delivery_df['hour_of_day'] = delivery_df['Date'].dt.hour
                delivery_df['is_weekend'] = delivery_df['day_of_week'].isin([5, 6]).astype(int)
                delivery_df['month'] = delivery_df['Date'].dt.month
                delivery_df['is_morning'] = (delivery_df['hour_of_day'] >= 6) & (delivery_df['hour_of_day'] < 12)
                delivery_df['is_afternoon'] = (delivery_df['hour_of_day'] >= 12) & (delivery_df['hour_of_day'] < 18)
                delivery_df['is_evening'] = (delivery_df['hour_of_day'] >= 18) & (delivery_df['hour_of_day'] < 22)
                delivery_df['is_night'] = (delivery_df['hour_of_day'] >= 22) | (delivery_df['hour_of_day'] < 6)
                
                # Convert to dummies
                for col in ['is_morning', 'is_afternoon', 'is_evening', 'is_night']:
                    delivery_df[col] = delivery_df[col].astype(int)
            except Exception as e:
                logging.error(f"Error processing Date column: {e}")
                # Add default time features
                for col in ['day_of_week', 'hour_of_day', 'is_weekend', 'month']:
                    if col not in delivery_df.columns:
                        delivery_df[col] = 0
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
            # Apply enhanced subject features extraction
            logging.info("Extracting subject line features")
            try:
                subject_features = delivery_df['subject'].apply(extract_subject_features).apply(pd.Series)
                
                # Join the subject features to the main dataframe
                for col in subject_features.columns:
                    delivery_df[col] = subject_features[col]
                
                # Create additional derived features from subject text
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
                
                # --- Advanced NLP for subject lines ---
                if len(delivery_df) >= 10:  # Only if we have enough samples
                    try:
                        # Use TF-IDF to extract important words from subject lines
                        logging.info("Performing TF-IDF analysis on subject lines")
                        tfidf = TfidfVectorizer(
                            max_features=10, 
                            stop_words=['english', 'swedish', 'och', 'för', 'med', 'att', 'den', 'det', 'du', 'är']
                        )
                        subject_tfidf = tfidf.fit_transform(delivery_df['subject'].fillna(''))
                        
                        # Convert to dataframe
                        tfidf_cols = [f'tfidf_{word}' for word in tfidf.get_feature_names_out()]
                        tfidf_df = pd.DataFrame(subject_tfidf.toarray(), columns=tfidf_cols)
                        
                        # Add TF-IDF features to main dataframe
                        for col in tfidf_cols:
                            delivery_df[col] = tfidf_df[col].values
                    except Exception as e:
                        logging.error(f"Error in TF-IDF processing: {e}")
            except Exception as e:
                logging.error(f"Error processing subject features: {e}")
        else:
            logging.warning("Subject column not found. Skipping subject feature extraction.")
        
        # --- Customer demographics features ---
        if 'Bolag' in customer_df.columns and 'InternalName' in customer_df.columns:
            try:
                logging.info("Processing customer demographics")
                # Create aggregated customer features by delivery
                customer_aggs = customer_df.groupby('InternalName').agg({
                    'Bolag': lambda x: x.mode()[0] if not x.mode().empty else 'Unknown',
                    'Age': ['mean', 'median', 'std'],
                    'Optout': 'mean',
                    'Open': 'mean',
                    'Click': 'mean'
                })
                
                # Flatten multi-index columns
                customer_aggs.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in customer_aggs.columns]
                
                # Rename for clarity
                column_renames = {
                    'Bolag_<lambda>': 'main_bolag',
                    'Age_mean': 'avg_age',
                    'Age_median': 'median_age',
                    'Age_std': 'std_age',
                    'Optout_mean': 'historical_optout_rate',
                    'Open_mean': 'historical_open_rate',
                    'Click_mean': 'historical_click_rate'
                }
                
                customer_aggs.rename(columns=column_renames, inplace=True)
                customer_aggs.reset_index(inplace=True)
                
                # Calculate gender ratio if Gender column exists
                if 'Gender' in customer_df.columns:
                    gender_aggs = customer_df.groupby('InternalName').apply(
                        lambda x: (x['Gender'].str.lower() == 'f').mean() * 100
                    ).reset_index(name='pct_women')
                    
                    # Join with other aggregations
                    customer_aggs = customer_aggs.merge(gender_aggs, on='InternalName', how='left')
                
                # Merge with delivery data
                delivery_df = delivery_df.merge(customer_aggs, on='InternalName', how='left')
                
                # Fill missing values with reasonable defaults
                demographic_defaults = {
                    'avg_age': 40,
                    'median_age': 38,
                    'std_age': 15,
                    'pct_women': 50,
                    'historical_optout_rate': 0.01,
                    'historical_open_rate': 0.2,
                    'historical_click_rate': 0.1
                }
                
                delivery_df.fillna(demographic_defaults, inplace=True)
            except Exception as e:
                logging.error(f"Error processing customer demographics: {e}")
        
        # --- Calculate engagement metrics if they don't exist ---
        if 'open_rate' not in delivery_df.columns:
            if 'Opens' in delivery_df.columns and 'Sendouts' in delivery_df.columns:
                delivery_df['open_rate'] = (delivery_df['Opens'] / delivery_df['Sendouts']) * 100
        
        if 'click_rate' not in delivery_df.columns:
            if 'Clicks' in delivery_df.columns and 'Opens' in delivery_df.columns:
                delivery_df['click_rate'] = (delivery_df['Clicks'] / delivery_df['Opens']) * 100
        
        if 'optout_rate' not in delivery_df.columns:
            if 'Optout' in delivery_df.columns and 'Opens' in delivery_df.columns:
                delivery_df['optout_rate'] = (delivery_df['Optout'] / delivery_df['Opens']) * 100
        
        # Replace infinities and NaNs
        delivery_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Fill missing rates with median or 0
        if 'open_rate' in delivery_df.columns:
            median_open = delivery_df['open_rate'].median()
            delivery_df['open_rate'].fillna(median_open if not pd.isna(median_open) else 0, inplace=True)
            
        if 'click_rate' in delivery_df.columns:
            median_click = delivery_df['click_rate'].median()
            delivery_df['click_rate'].fillna(median_click if not pd.isna(median_click) else 0, inplace=True)
            
        if 'optout_rate' in delivery_df.columns:
            median_optout = delivery_df['optout_rate'].median()
            delivery_df['optout_rate'].fillna(median_optout if not pd.isna(median_optout) else 0, inplace=True)
        
        logging.info(f"Enhanced feature engineering completed. Shape: {delivery_df.shape}")
        return delivery_df
        
    except Exception as e:
        logging.error(f"Error in enhanced_feature_engineering: {e}")
        # Return the original dataframe if processing fails
        logging.info("Returning original delivery_df due to error")
        return delivery_df