import pandas as pd
import numpy as np
from datetime import datetime
import re
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_subject_features(subject):
    """
    Extract features from email subject lines
    
    Parameters:
    - subject: Email subject line text
    
    Returns:
    - Dictionary of extracted features
    """
    if not isinstance(subject, str):
        return {
            'length': 0,
            'has_personalization': 0,
            'has_question': 0,
            'has_numbers': 0,
            'has_uppercase_words': 0,
            'has_emoji': 0,
            'word_count': 0
        }
        
    features = {}
    features['length'] = len(subject)
    features['has_personalization'] = 1 if re.search(r'\b(your|you|du|din|ditt|dina)\b', subject.lower()) else 0
    features['has_question'] = 1 if '?' in subject else 0
    features['has_numbers'] = 1 if re.search(r'\d', subject) else 0
    features['has_uppercase_words'] = 1 if re.search(r'\b[A-Z]{2,}\b', subject) else 0
    features['has_emoji'] = 1 if re.search(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF]', subject) else 0
    features['word_count'] = len(subject.split())
    return features

def enhanced_feature_engineering(delivery_df, customer_df):
    """Enhanced feature engineering for email campaign data"""
    import pandas as pd
    import numpy as np
    from datetime import datetime
    import re
    from sklearn.feature_extraction.text import TfidfVectorizer

    # --- Time-based features ---
    delivery_df['Date'] = pd.to_datetime(delivery_df['Date'])
    delivery_df['day_of_week'] = delivery_df['Date'].dt.dayofweek
    delivery_df['hour_of_day'] = delivery_df['Date'].dt.hour
    delivery_df['is_weekend'] = delivery_df['day_of_week'].isin([5, 6]).astype(int)
    
    # --- Subject line features ---
    # Apply subject features extraction
    subject_features = delivery_df['subject'].apply(extract_subject_features).apply(pd.Series)
    delivery_df = pd.concat([delivery_df, subject_features], axis=1)
    
    # --- Advanced NLP for subject lines ---
    # Use TF-IDF to extract important words from subject lines
    tfidf = TfidfVectorizer(max_features=10, stop_words=['english', 'swedish'])  # Adjust stop_words as needed
    if len(delivery_df) > 10:  # Need sufficient samples
        subject_tfidf = tfidf.fit_transform(delivery_df['subject'].fillna(''))
        tfidf_df = pd.DataFrame(subject_tfidf.toarray(), columns=tfidf.get_feature_names_out())
        delivery_df = pd.concat([delivery_df, tfidf_df], axis=1)
    
    # --- Customer engagement features ---
    # Calculate customer-level engagement metrics
    customer_engagement = customer_df.groupby('Primary k isWoman OptOut').agg(
        past_opens=pd.NamedAgg(column='opens_count', aggfunc='mean'),  # Assuming these columns exist
        past_clicks=pd.NamedAgg(column='clicks_count', aggfunc='mean'),
        engagement_score=pd.NamedAgg(column='engagement_score', aggfunc='mean')
    ).reset_index()
    
    # --- Merge with delivery data ---
    # First link customers to deliveries
    delivery_customer = delivery_df.merge(
        customer_df[['Primary k isWoman OptOut', 'InternalName']], 
        on='InternalName', 
        how='left'
    )
    
    # Then add customer engagement metrics
    merged_df = delivery_customer.merge(
        customer_engagement,
        on='Primary k isWoman OptOut',
        how='left'
    )
    
    # --- Calculate derived metrics ---
    merged_df['open_rate'] = (merged_df['Opens'] / merged_df['Utskick']) * 100
    merged_df['click_rate'] = (merged_df['Clicks'] / merged_df['Opens']) * 100
    merged_df['click_to_open_rate'] = (merged_df['Clicks'] / merged_df['Opens']) * 100
    merged_df['optout_rate'] = (merged_df['Optout'] / merged_df['Opens']) * 100
    
    # --- Handle missing values ---
    merged_df = merged_df.fillna({
        'past_opens': merged_df['past_opens'].median(),
        'past_clicks': merged_df['past_clicks'].median(),
        'engagement_score': merged_df['engagement_score'].median(),
        'click_rate': 0,
        'click_to_open_rate': 0,
        'optout_rate': 0
    })
    
    return merged_df