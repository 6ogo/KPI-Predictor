"""
Subject line recommendation module for email campaigns
"""
import pandas as pd
import numpy as np
import re
import logging
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from feature_engineering import extract_subject_features

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def recommend_subject(current_subject, delivery_df, subject_patterns=None):
    """
    Recommend a better subject line based on historical data and patterns.
    """
    import re

    if subject_patterns and isinstance(subject_patterns, list) and len(subject_patterns) > 0:
        best_pattern = subject_patterns[0]
        words = re.findall(r'\b\w+\b', current_subject)
        if len(words) >= 3:
            recommended = best_pattern.replace('[PRODUCT]', words[0])
            recommended = recommended.replace('[ACTION]', words[1] if len(words) > 1 else 'check')
            recommended = recommended.replace('[BENEFIT]', words[2] if len(words) > 2 else 'today')
        else:
            recommended = best_pattern.replace('[PRODUCT]', 'our products')
            recommended = recommended.replace('[ACTION]', 'check out')
            recommended = recommended.replace('[BENEFIT]', 'special offers')
        return recommended

    # Fallback if no patterns are available
    from feature_engineering import extract_subject_features
    features = extract_subject_features(current_subject)
    if features['has_personalization'] == 0:
        return "Din " + current_subject  # Swedish "Your"
    if features['has_question'] == 0:
        return current_subject + "?"
    if features['has_emoji'] == 0:
        return current_subject + " ✨"
    if len(current_subject) > 50:
        words = current_subject.split()
        return " ".join(words[:7]) + "..."
    return current_subject

def build_subject_recommendation_model(delivery_df):
    """
    Build a subject line recommendation model based on historical data using clustering.
    """
    if 'Subject' in delivery_df.columns:
        tfidf = TfidfVectorizer(max_features=50)
        tfidf_matrix = tfidf.fit_transform(delivery_df['Subject'].fillna(''))
        kmeans = KMeans(n_clusters=5, random_state=42)
        clusters = kmeans.fit_predict(tfidf_matrix)
        delivery_df['subject_cluster'] = clusters

        # Find best cluster by open rate
        cluster_performance = delivery_df.groupby('subject_cluster')['open_rate'].mean()
        best_cluster = cluster_performance.idxmax()
        best_subjects = delivery_df[delivery_df['subject_cluster'] == best_cluster]['Subject'].tolist()
        patterns = best_subjects[:3]  # Top 3 subjects as patterns
    else:
        patterns = ["Din [PRODUCT] [ACTION] för [BENEFIT]"]

    # Create subject recommendations by category
    recommendations = {
        'informational': "Viktig information om din tjänst",
        'promotional': "Specialerbjudande: Ta del av våra förmåner idag",
        'newsletter': "Din månatliga uppdatering",
        'announcement': "Nyheter: Nya tjänster och erbjudanden",
        'reminder': "Påminnelse: Viktig information om ditt konto"
    }

    return recommendations, patterns

def generate_recommendations(input_data, models, delivery_df, subject_patterns=None):
    """
    Generate recommendations for targeting and subject lines
    
    Parameters:
    - input_data: DataFrame with current input parameters
    - models: Dict of trained models for different metrics
    - delivery_df: DataFrame with historical delivery data
    - subject_patterns: Optional subject line patterns from clustering
    
    Returns:
    - Dictionary with recommendations and predictions
    """
    results = {}
    
    # Use the predict_metrics function from multi_metric_model
    from multi_metric_model import predict_metrics
    
    # 1. Current predictions (baseline)
    current_predictions = predict_metrics(input_data, models)
    results['current'] = current_predictions
    
    # 2. Targeting recommendations
    # Find best performing county based on historical data
    best_county = delivery_df.groupby('county')['open_rate'].mean().idxmax() if 'county' in delivery_df.columns else "Stockholm"
    
    # Create input with recommended targeting
    targeting_data = input_data.copy()
    targeting_data['county'] = best_county
    
    # Predict all metrics for targeting recommendations
    targeting_predictions = predict_metrics(targeting_data, models, 
                                            metrics=['open_rate', 'click_rate', 'optout_rate'])
    
    results['targeting'] = {
        'county': best_county,
        'predictions': targeting_predictions
    }
    
    # Get the current subject (could be 'subject' or 'Subject' column)
    current_subject = ""
    if 'subject' in input_data.columns:
        current_subject = input_data['subject'].iloc[0]
    elif 'Subject' in input_data.columns:
        current_subject = input_data['Subject'].iloc[0]
    
    # Build subject recommendation model
    subject_recommendations, pattern_list = build_subject_recommendation_model(delivery_df)
        
    # Get recommended subject
    recommended_subject = recommend_subject(
        current_subject, 
        delivery_df, 
        subject_patterns=pattern_list if subject_patterns is None else subject_patterns,
        subject_recommendations=subject_recommendations
    )
    
    # 3. Extract features from recommended subject
    from feature_engineering import extract_subject_features
    recommended_subject_features = extract_subject_features(recommended_subject)
    
    # Create input with recommended subject
    subject_data = input_data.copy()
    for feature, value in recommended_subject_features.items():
        if feature in subject_data.columns:
            subject_data[feature] = value
    
    # Only predict open_rate for subject line recommendations
    subject_prediction = predict_metrics(subject_data, models, metrics=['open_rate'])
    
    results['subject'] = {
        'text': recommended_subject,
        'open_rate': subject_prediction.get('open_rate', 0)
    }
    
    # 4. Combined recommendation (targeting + subject)
    # Create combined input data
    combined_data = targeting_data.copy()
    for feature, value in recommended_subject_features.items():
        if feature in combined_data.columns:
            combined_data[feature] = value
    
    # Predict all metrics for the combined recommendation
    combined_predictions = predict_metrics(combined_data, models, 
                                          metrics=['open_rate', 'click_rate', 'optout_rate'])
    
    results['combined'] = {
        'county': best_county,
        'subject': recommended_subject,
        'predictions': combined_predictions
    }
    
    return results


def format_predictions(recommendations):
    """Format prediction results for display"""
    formatted = {}
    
    # Current predictions
    formatted['current'] = {
        'open_rate': recommendations['current'].get('open_rate', 0),
        'click_rate': recommendations['current'].get('click_rate', 0),
        'optout_rate': recommendations['current'].get('optout_rate', 0)
    }
    
    # Targeting recommendations
    formatted['targeting'] = {
        'county': recommendations['targeting']['county'],
        'open_rate': recommendations['targeting']['predictions'].get('open_rate', 0),
        'click_rate': recommendations['targeting']['predictions'].get('click_rate', 0),
        'optout_rate': recommendations['targeting']['predictions'].get('optout_rate', 0),
        'open_rate_diff': recommendations['targeting']['predictions'].get('open_rate', 0) - formatted['current']['open_rate'],
        'click_rate_diff': recommendations['targeting']['predictions'].get('click_rate', 0) - formatted['current']['click_rate'],
        'optout_rate_diff': recommendations['targeting']['predictions'].get('optout_rate', 0) - formatted['current']['optout_rate']
    }
    
    # Subject recommendations (only open rate)
    formatted['subject'] = {
        'text': recommendations['subject']['text'],
        'open_rate': recommendations['subject']['open_rate'],
        'open_rate_diff': recommendations['subject']['open_rate'] - formatted['current']['open_rate']
    }
    
    # Combined recommendations
    formatted['combined'] = {
        'county': recommendations['combined']['county'],
        'subject': recommendations['combined']['subject'],
        'open_rate': recommendations['combined']['predictions'].get('open_rate', 0),
        'click_rate': recommendations['combined']['predictions'].get('click_rate', 0),
        'optout_rate': recommendations['combined']['predictions'].get('optout_rate', 0),
        'open_rate_diff': recommendations['combined']['predictions'].get('open_rate', 0) - formatted['current']['open_rate'],
        'click_rate_diff': recommendations['combined']['predictions'].get('click_rate', 0) - formatted['current']['click_rate'],
        'optout_rate_diff': recommendations['combined']['predictions'].get('optout_rate', 0) - formatted['current']['optout_rate']
    }
    
    return formatted