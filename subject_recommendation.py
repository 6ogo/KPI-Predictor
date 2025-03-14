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

def recommend_subject(current_subject, delivery_df, subject_patterns=None, subject_recommendations=None):
    """
    Recommend a better subject line based on historical data
    
    Parameters:
    - current_subject: Current subject line
    - delivery_df: DataFrame with historical delivery data
    - subject_patterns: Optional patterns from subject line clustering
    - subject_recommendations: Optional pre-built recommendations
    
    Returns:
    - Recommended subject line text
    """
    if not current_subject or current_subject.strip() == "":
        current_subject = "Newsletter"
    
    # If we have specific recommendations based on category, use those
    if subject_recommendations and isinstance(subject_recommendations, dict) and len(subject_recommendations) > 0:
        # Try to match current subject to a category
        categories = list(subject_recommendations.keys())
        best_category = categories[0]  # Default to first category
        
        # Check for category keywords in the current subject
        lower_subject = current_subject.lower()
        for category, recommendation in subject_recommendations.items():
            category_keywords = {
                'informational': ['information', 'update', 'notice', 'important'],
                'promotional': ['offer', 'special', 'discount', 'save', 'deal'],
                'newsletter': ['newsletter', 'weekly', 'monthly', 'news'],
                'announcement': ['announcement', 'introducing', 'new', 'launch'],
                'reminder': ['reminder', 'don\'t forget', 'deadline', 'remember']
            }
            
            if category in category_keywords:
                for keyword in category_keywords[category]:
                    if keyword in lower_subject:
                        best_category = category
                        break
        
        # Get the recommendation for the best matching category
        template = subject_recommendations[best_category]
        
        # Extract key terms from current subject to use in the template
        words = re.findall(r'\b\w+\b', current_subject)
        product_terms = [w for w in words if len(w) > 3][:2]  # Use longer words as product terms
        product = ' '.join(product_terms) if product_terms else 'products'
        
        # Replace placeholders with extracted or default terms
        recommendation = template
        recommendation = recommendation.replace('[PRODUCT]', product)
        recommendation = recommendation.replace('[ACTION]', 'check out')
        recommendation = recommendation.replace('[BENEFIT]', 'special benefits')
        
        return recommendation
    
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
    features = extract_subject_features(current_subject)
    
    if features['has_personalization'] == 0:
        return "Din " + current_subject  # Swedish "Your"
    
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
    logging.info("Building subject recommendation model")
    
    # Get the subject column (either 'subject' or 'Subject')
    if 'subject' in delivery_df.columns:
        subject_col = 'subject'
    elif 'Subject' in delivery_df.columns:
        subject_col = 'Subject'
    else:
        logging.warning("No subject column found! Creating default recommendations.")
        # Create empty recommendations if no subject column exists
        return {
            'informational': "Viktig information om din tjänst",
            'promotional': "Specialerbjudande: Ta del av våra förmåner idag",
            'newsletter': "Din månatliga uppdatering",
            'announcement': "Nyheter: Nya tjänster och erbjudanden",
            'reminder': "Påminnelse: Viktig information om ditt konto"
        }, ["Din [PRODUCT] [ACTION] för [BENEFIT]"]
    
    # Make sure we have open_rate column
    if 'open_rate' not in delivery_df.columns:
        if 'Opens' in delivery_df.columns and 'Sendouts' in delivery_df.columns:
            delivery_df['open_rate'] = (delivery_df['Opens'] / delivery_df['Sendouts']) * 100
        else:
            logging.warning("Cannot calculate open_rate! Using default recommendations.")
            return {
                'informational': "Viktig information om din tjänst",
                'promotional': "Specialerbjudande: Ta del av våra förmåner idag",
                'newsletter': "Din månatliga uppdatering",
                'announcement': "Nyheter: Nya tjänster och erbjudanden",
                'reminder': "Påminnelse: Viktig information om ditt konto"
            }, ["Din [PRODUCT] [ACTION] för [BENEFIT]"]
    
    # Clean and prepare subject lines
    delivery_df['clean_subject'] = delivery_df[subject_col].fillna("")
    
    # Skip if not enough data
    if len(delivery_df) < 5:
        logging.warning("Not enough data for subject analysis. Using default recommendations.")
        return {
            'informational': "Viktig information om din tjänst",
            'promotional': "Specialerbjudande: Ta del av våra förmåner idag",
            'newsletter': "Din månatliga uppdatering",
            'announcement': "Nyheter: Nya tjänster och erbjudanden",
            'reminder': "Påminnelse: Viktig information om ditt konto"
        }, ["Din [PRODUCT] [ACTION] för [BENEFIT]"]
    
    # Extract features from all historical subject lines
    try:
        subject_features_df = pd.DataFrame(delivery_df['clean_subject'].apply(extract_subject_features).tolist())
        
        # Add open rates to the features
        subject_features_df['open_rate'] = delivery_df['open_rate'].values
        
        # Analyze the relationship between features and open_rate
        feature_impact = {}
        
        for feature in ['has_personalization', 'has_question', 'has_numbers', 'has_uppercase_words', 'has_emoji']:
            if feature in subject_features_df.columns:
                # Compare means for feature present (1) vs absent (0)
                impact = subject_features_df.groupby(feature)['open_rate'].mean()
                
                # Calculate the difference
                if 1 in impact and 0 in impact:
                    feature_impact[feature] = impact[1] - impact[0]
                    logging.info(f"Impact of {feature}: {feature_impact[feature]:.2f} percentage points")
        
        # For word count and length, find the optimal values
        for feature in ['word_count', 'length']:
            if feature in subject_features_df.columns:
                # Group into bins for analysis
                if feature == 'word_count':
                    bins = [0, 3, 6, 9, 12, 15, 100]
                    labels = ['1-3', '4-6', '7-9', '10-12', '13-15', '15+']
                else:  # length
                    bins = [0, 20, 40, 60, 80, 100, 1000]
                    labels = ['1-20', '21-40', '41-60', '61-80', '81-100', '100+']
                
                subject_features_df[f'{feature}_bin'] = pd.cut(
                    subject_features_df[feature], 
                    bins=bins, 
                    labels=labels
                )
                
                # Get average open rate by bin
                bin_impact = subject_features_df.groupby(f'{feature}_bin')['open_rate'].mean()
                
                # Find the optimal bin
                if not bin_impact.empty:
                    optimal_bin = bin_impact.idxmax()
                    feature_impact[f'optimal_{feature}'] = optimal_bin
                    logging.info(f"Optimal {feature}: {optimal_bin} (open rate: {bin_impact[optimal_bin]:.2f}%)")
    except Exception as e:
        logging.error(f"Error in subject feature analysis: {e}")
        feature_impact = {}
    
    # Advanced text analysis of subject lines
    try:
        # Only proceed if we have enough data
        if len(delivery_df) >= 10:
            # Create a corpus of subject lines
            corpus = delivery_df['clean_subject'].fillna("").tolist()
            
            # Extract frequent words using CountVectorizer
            vectorizer = CountVectorizer(max_features=50, stop_words=['english', 'swedish', 'din', 'och', 'för', 'att', 'med', 'det', 'är'])
            X = vectorizer.fit_transform(corpus)
            words = vectorizer.get_feature_names_out()
            
            # Get the word counts
            word_counts = X.sum(axis=0).A1
            
            # Create a dictionary of word to count
            word_freq = dict(zip(words, word_counts))
            
            # Find which words are associated with higher open rates
            word_performance = {}
            
            for word in words:
                # Find subjects containing this word
                mask = delivery_df['clean_subject'].str.contains(r'\b' + word + r'\b', case=False, regex=True)
                
                # Skip if word is too rare
                if mask.sum() < 3:
                    continue
                
                # Compare open rates
                with_word = delivery_df.loc[mask, 'open_rate'].mean()
                without_word = delivery_df.loc[~mask, 'open_rate'].mean()
                
                # Calculate the impact
                impact = with_word - without_word
                word_performance[word] = impact
            
            # Find the best performing words
            best_words = sorted(word_performance.items(), key=lambda x: x[1], reverse=True)[:10]
            worst_words = sorted(word_performance.items(), key=lambda x: x[1])[:5]
            
            logging.info(f"Best performing words: {best_words}")
            logging.info(f"Worst performing words: {worst_words}")
            
            # Try to identify topic clusters using LDA
            try:
                # Use TF-IDF to focus on important words
                tfidf_vectorizer = TfidfVectorizer(max_features=100, stop_words=['english', 'swedish', 'din', 'och', 'för', 'att', 'med', 'det', 'är'])
                tfidf = tfidf_vectorizer.fit_transform(corpus)
                
                # Apply LDA
                lda = LatentDirichletAllocation(n_components=5, random_state=42)
                lda.fit(tfidf)
                
                # Get topics
                topics = []
                feature_names = tfidf_vectorizer.get_feature_names_out()
                
                for topic_idx, topic in enumerate(lda.components_):
                    top_words = [feature_names[i] for i in topic.argsort()[:-11:-1]]
                    topics.append(top_words)
                
                logging.info(f"Identified topics: {topics}")
            except Exception as e:
                logging.warning(f"Error in topic extraction: {e}")
                topics = []
        else:
            word_performance = {}
            best_words = []
            worst_words = []
            topics = []
    except Exception as e:
        logging.error(f"Error in text analysis: {e}")
        word_performance = {}
        best_words = []
        worst_words = []
        topics = []
    
    # Generate subject line patterns based on findings
    patterns = []
    
    # 1. Create pattern based on feature impact
    if feature_impact:
        # Start with basic patterns
        if feature_impact.get('has_personalization', 0) > 0:
            patterns.append("Din [PRODUCT] [ACTION] för [BENEFIT]")
        
        if feature_impact.get('has_question', 0) > 0:
            patterns.append("Vill du [ACTION] din [PRODUCT] för bättre [BENEFIT]?")
        
        if feature_impact.get('has_numbers', 0) > 0:
            patterns.append("5 sätt att [ACTION] din [PRODUCT] för bättre [BENEFIT]")
            
        # Add patterns based on optimal length/word count
        optimal_length = feature_impact.get('optimal_length', '21-40')
        
        if optimal_length == '1-20':
            patterns.append("[ACTION] [PRODUCT] nu")
        elif optimal_length == '21-40':
            patterns.append("[ACTION] din [PRODUCT] för [BENEFIT]")
        elif optimal_length == '41-60':
            patterns.append("Bästa sättet att [ACTION] din [PRODUCT] för bra [BENEFIT]")
        else:  # longer
            patterns.append("Upptäck hur du kan [ACTION] din [PRODUCT] för att uppnå bästa [BENEFIT] denna månad")
    
    # 2. Create patterns using best performing words
    if best_words:
        top_words = [word for word, _ in best_words[:5]]
        
        # Create pattern using top performing words
        if len(top_words) >= 3:
            pattern = f"{top_words[0]} [PRODUCT] {top_words[1]} för {top_words[2]}"
            patterns.append(pattern)
    
    # 3. Create patterns based on topic clusters
    if topics:
        for topic_words in topics:
            if len(topic_words) >= 3:
                pattern = f"{topic_words[0]} {topic_words[1]} [PRODUCT] för {topic_words[2]}"
                patterns.append(pattern)
    
    # 4. Default patterns if none generated
    if not patterns:
        patterns = [
            "Upptäck din [PRODUCT] [BENEFIT] idag",
            "Hur du kan [ACTION] din [PRODUCT] för bättre [BENEFIT]",
            "Din [PRODUCT]: [ACTION] för [BENEFIT] nu"
        ]
    
    # Create subject recommendations by category - using Swedish text
    recommendations = {
        'informational': "Information om [PRODUCT]: Viktig information om din [BENEFIT]",
        'promotional': "Specialerbjudande: [ACTION] [PRODUCT] och få [BENEFIT]",
        'newsletter': "Din månatliga [PRODUCT] uppdatering: [BENEFIT] och mer",
        'announcement': "Nyheter: Nya [PRODUCT] [BENEFIT] - [ACTION] nu",
        'reminder': "Påminnelse: [ACTION] din [PRODUCT] för [BENEFIT] innan fredag"
    }
    
    # Enhance category-specific recommendations based on word performance
    if best_words:
        for category in recommendations:
            # Get original template
            template = recommendations[category]
            
            # Try to incorporate good performing words
            for word, impact in best_words:
                if impact > 0:
                    if '[ACTION]' in template and len(word) > 3:
                        template = template.replace('[ACTION]', f"{word}")
                        break
            
            # Update recommendation
            recommendations[category] = template
    
    logging.info(f"Generated {len(patterns)} subject patterns")
    logging.info(f"Generated {len(recommendations)} category recommendations")
    
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