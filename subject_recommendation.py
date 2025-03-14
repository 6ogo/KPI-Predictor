def recommend_subject(current_subject, delivery_df, subject_patterns=None):
    """
    Recommend a better subject line based on historical data
    
    Parameters:
    - current_subject: Current subject line
    - delivery_df: DataFrame with historical delivery data
    - subject_patterns: Optional patterns from subject line clustering
    
    Returns:
    - Recommended subject line text
    """
    import re
    
    # If we have predefined patterns from clustering, use those
    if subject_patterns and isinstance(subject_patterns, list) and len(subject_patterns) > 0:
        # Find the best performing pattern based on historical data
        best_pattern = subject_patterns[0]  # Default to first pattern
        
        # Substitute the current subject content into the pattern
        words = re.findall(r'\b\w+\b', current_subject)
        if len(words) >= 3:
            recommended = best_pattern.replace('[PRODUCT]', words[0])
            recommended = recommended.replace('[ACTION]', words[1] if len(words) > 1 else 'check')
            recommended = recommended.replace('[BENEFIT]', words[2] if len(words) > 2 else 'today')
        else:
            # Not enough words in current subject, use the pattern with placeholders
            recommended = best_pattern.replace('[PRODUCT]', 'our products')
            recommended = recommended.replace('[ACTION]', 'check out')
            recommended = recommended.replace('[BENEFIT]', 'special offers')
        
        return recommended
    
    # If no patterns available, try to enhance the current subject
    # Add personalization if not present
    from feature_engineering import extract_subject_features
    features = extract_subject_features(current_subject)
    
    if features['has_personalization'] == 0:
        return "Your " + current_subject
    
    if features['has_question'] == 0:
        return current_subject + "?"
    
    if features['has_emoji'] == 0:
        return current_subject + " ✨"
    
    # If all enhancements already present, try to make it more concise
    if len(current_subject) > 50:
        words = current_subject.split()
        return " ".join(words[:7]) + "..."
    
    # No changes needed
    return current_subject


def build_subject_recommendation_model(delivery_df):
    """
    Build a subject line recommendation model based on historical data
    
    Parameters:
    - delivery_df: DataFrame with delivery data including subject lines and open rates
    
    Returns:
    - Tuple of (subject_recommendations, subject_patterns)
        - subject_recommendations: Dictionary of subject line recommendations by category
        - subject_patterns: List of effective subject line patterns
    """
    import pandas as pd
    import numpy as np
    from feature_engineering import extract_subject_features
    
    # Extract features from all historical subject lines
    subject_features_df = pd.DataFrame(delivery_df['subject'].apply(extract_subject_features).tolist())
    
    # Add open rates to the features
    subject_features_df['open_rate'] = delivery_df['open_rate'].values
    
    # Find patterns that correlate with high open rates
    patterns = []
    
    # 1. Questions in subject line
    if 'has_question' in subject_features_df.columns:
        question_impact = subject_features_df.groupby('has_question')['open_rate'].mean()
        
        if question_impact.get(1, 0) > question_impact.get(0, 0):
            patterns.append("Why [PRODUCT] [ACTION] can [BENEFIT] you?")
    
    # 2. Personalization in subject line
    if 'has_personalization' in subject_features_df.columns:
        personalization_impact = subject_features_df.groupby('has_personalization')['open_rate'].mean()
        
        if personalization_impact.get(1, 0) > personalization_impact.get(0, 0):
            patterns.append("Your [PRODUCT] [ACTION] is ready with [BENEFIT]")
    
    # 3. Numbers in subject line
    if 'has_numbers' in subject_features_df.columns:
        numbers_impact = subject_features_df.groupby('has_numbers')['open_rate'].mean()
        
        if numbers_impact.get(1, 0) > numbers_impact.get(0, 0):
            patterns.append("3 ways to [ACTION] your [PRODUCT] for [BENEFIT]")
    
    # 4. Optimal length
    if 'length' in subject_features_df.columns:
        # Bin the lengths
        subject_features_df['length_bin'] = pd.cut(
            subject_features_df['length'], 
            bins=[0, 20, 40, 60, 100], 
            labels=['very_short', 'short', 'medium', 'long']
        )
        
        length_impact = subject_features_df.groupby('length_bin')['open_rate'].mean()
        best_length = length_impact.idxmax() if not length_impact.empty else 'short'
        
        if best_length == 'very_short':
            patterns.append("[ACTION] [PRODUCT] now")
        elif best_length == 'short':
            patterns.append("[ACTION] your [PRODUCT] for [BENEFIT]")
        elif best_length == 'medium':
            patterns.append("The best way to [ACTION] your [PRODUCT] for amazing [BENEFIT]")
        else:  # long
            patterns.append("Discover how you can [ACTION] your [PRODUCT] to achieve the greatest [BENEFIT] this month")
    
    # 5. Default pattern if no patterns generated
    if not patterns:
        patterns = [
            "Discover your [PRODUCT] [BENEFIT] today",
            "How to [ACTION] your [PRODUCT] for better [BENEFIT]",
            "Your [PRODUCT]: [ACTION] for [BENEFIT] now"
        ]
    
    # Create subject recommendations by category
    recommendations = {
        'informational': "[PRODUCT] update: Important information about your [BENEFIT]",
        'promotional': "Special offer: [ACTION] [PRODUCT] and get [BENEFIT]",
        'newsletter': "Your monthly [PRODUCT] update: [BENEFIT] and more",
        'announcement': "Announcing: New [PRODUCT] [BENEFIT] - [ACTION] now",
        'reminder': "Reminder: [ACTION] your [PRODUCT] for [BENEFIT] by Friday"
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
        
    current_subject = input_data['subject'][0] if 'subject' in input_data else ""
    recommended_subject = recommend_subject(current_subject, delivery_df, 
                                            subject_patterns=subject_patterns)
    
    # 3. Extract features from recommended subject
    from feature_engineering import extract_subject_features
    recommended_subject_features = extract_subject_features(recommended_subject)
    
    # Create input with recommended subject
    subject_data = input_data.copy()
    for feature, value in recommended_subject_features.items():
        if feature in subject_data.columns:
            subject_data[feature] = value
    
    # Only predict open_rate for subject line recommendations
    subject_prediction = models['open_rate'].predict(subject_data)[0]
    
    results['subject'] = {
        'text': recommended_subject,
        'open_rate': subject_prediction
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
        
    current_subject = input_data['subject'][0] if 'subject' in input_data else ""
    recommended_subject = recommend_subject(current_subject, delivery_df, 
                                            subject_patterns=subject_patterns)
    
    # 3. Extract features from recommended subject
    from feature_engineering import extract_subject_features
    recommended_subject_features = extract_subject_features(recommended_subject)
    
    # Create input with recommended subject
    subject_data = input_data.copy()
    for feature, value in recommended_subject_features.items():
        if feature in subject_data.columns:
            subject_data[feature] = value
    
    # Only predict open_rate for subject line recommendations
    subject_prediction = models['open_rate'].predict(subject_data)[0]
    
    results['subject'] = {
        'text': recommended_subject,
        'open_rate': subject_prediction
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