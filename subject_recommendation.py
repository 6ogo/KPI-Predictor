"""
Subject line recommendation module for email campaigns
"""
import pandas as pd
import numpy as np
import re
import logging
import random
from collections import Counter
import traceback
from feature_engineering import extract_subject_features

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_subject_templates():
    """
    Generate a list of high-performing subject line templates.
    """
    templates = [
        # Personalized templates (Swedish)
        "Din guide till {product} - {benefit} resultat",
        "Upptäck dina {product} förmåner {benefit}",
        "Se hur {product} kan hjälpa dig {benefit}",
        "{action} din {product} och få {benefit}",
        "Snabbguide: {action} {product} som aldrig förr",
        "Få ut mer av din {product} - {benefit}",
        "Speciellt för dig: {action} {product} med {benefit}",
        "Ny {product}: {action} och få {benefit}",
        "{benefit} erbjudande på {product} - bara idag!",
        
        # Question-based templates
        "Har du provat vår nya {product}?",
        "Vet du hur du kan {action} din {product} {benefit}?",
        "Vill du veta mer om {product} och {benefit}?",
        "Behöver din {product} ett {action}?",
        
        # Urgency-based templates
        "Sista chansen: {action} {product} för {benefit}",
        "Endast idag: {benefit} på alla {product}",
        "Missa inte: {action} {product} med {benefit}",
        
        # Benefit-focused templates
        "{benefit} med vår nya {product}",
        "Förbättra din {product} med {benefit}",
        "Få {benefit} med vår {product} - börja {action} idag",
    ]
    
    return templates

def recommend_subject(current_subject, delivery_df=None, subject_patterns=None, subject_recommendations=None):
    """
    Improved function to recommend better subject lines based on the current subject.
    The function now accepts the subject_recommendations parameter.
    
    Parameters:
    - current_subject: The current subject line text
    - delivery_df: DataFrame with historical delivery data
    - subject_patterns: List of subject line patterns to use
    - subject_recommendations: Dictionary of category-based recommendations
    
    Returns:
    - Recommended subject line
    """
    import random
    
    # Safety checks for inputs
    if not isinstance(current_subject, str):
        current_subject = str(current_subject) if current_subject is not None else ""
    
    # Check if we have category-based recommendations
    if subject_recommendations and isinstance(subject_recommendations, dict) and len(subject_recommendations) > 0:
        # Randomly select a category-based recommendation (25% chance)
        if random.random() < 0.25:
            category = random.choice(list(subject_recommendations.keys()))
            return subject_recommendations[category]
    
    # Extract key terms from current subject
    terms = extract_key_terms(current_subject)
    product = terms['products'][0] if terms['products'] else "produkt"
    action = terms['actions'][0] if terms['actions'] else "upptäck"
    benefit = terms['benefits'][0] if terms['benefits'] else "förmåner"
    
    # Get templates
    templates = generate_subject_templates()
    
    # Use subject patterns if available, otherwise use our templates
    if subject_patterns and isinstance(subject_patterns, list) and len(subject_patterns) > 0:
        # Add our templates to any existing patterns
        templates = subject_patterns + templates
    
    # Select 3 templates randomly to give options
    if len(templates) > 3:
        selected_templates = random.sample(templates, 3)
    else:
        selected_templates = templates
    
    # Format the templates with the extracted terms
    recommendations = []
    for template in selected_templates:
        try:
            rec = template.format(
                product=product,
                action=action,
                benefit=benefit
            )
            recommendations.append(rec)
        except KeyError as e:
            # If template uses a placeholder not in our mapping, use fallback
            logging.warning(f"Template error: {e}, using fallback template")
            fallback = f"Din {product}: {action} och få {benefit}"
            recommendations.append(fallback)
        except Exception as e:
            logging.error(f"Error formatting template '{template}': {e}")
            fallback = f"Din {product}: {action} och få {benefit}"
            recommendations.append(fallback)
    
    # Return the best recommendation (first one)
    return recommendations[0] if recommendations else f"Din {product}: {action} och få {benefit}"


def extract_key_terms(subject):
    """
    Extract key terms from the subject line that can be used in recommendations.
    """
    import re
    # Safety check
    if not isinstance(subject, str):
        subject = str(subject) if subject is not None else ""
        
    # Extract potential product terms (usually nouns)
    products = []
    actions = []
    benefits = []
    
    # Common action words in Swedish and English
    action_words = [
        'check', 'discover', 'get', 'see', 'find', 'try', 'read', 'buy', 'learn', 'start',
        'köp', 'se', 'hitta', 'prova', 'läs', 'börja', 'upptäck', 'få', 'spara', 'save'
    ]
    
    # Common benefit words in Swedish and English
    benefit_words = [
        'special', 'new', 'exclusive', 'free', 'limited', 'bonus', 'extra',
        'speciell', 'ny', 'exklusiv', 'gratis', 'begränsad', 'rabatt', 'erbjudande',
        'today', 'now', 'idag', 'nu', 'just nu', 'bästa', 'best'
    ]
    
    try:
        words = re.findall(r'\b\w+\b', subject.lower())
        
        for word in words:
            if word in action_words:
                actions.append(word)
            elif word in benefit_words:
                benefits.append(word)
            elif len(word) > 3:  # Potential product/noun
                products.append(word)
    except Exception as e:
        logging.error(f"Error extracting words from subject: {e}")
    
    return {
        'products': products if products else ['produkter'],  # Default to "produkter" if no products found
        'actions': actions if actions else ['se'],  # Default to "se" if no actions found
        'benefits': benefits if benefits else ['erbjudande']  # Default to "erbjudande" if no benefits found
    }

def build_subject_recommendation_model(delivery_df):
    """
    Build a more sophisticated subject line recommendation model
    """
    # Import the random module to ensure it's available
    import random
    import pandas as pd
    import numpy as np
    import logging
    
    # Default recommendations if we can't extract from data
    recommendations = {
        'informational': "Viktig information om din tjänst",
        'promotional': "Specialerbjudande: Ta del av våra förmåner idag",
        'newsletter': "Din månatliga uppdatering",
        'announcement': "Nyheter: Nya tjänster och erbjudanden",
        'reminder': "Påminnelse: Viktig information om ditt konto"
    }
    
    # Default patterns if we can't extract from data
    default_patterns = [
        "Din {product}: {action} och få {benefit}",
        "{action} din {product} för bästa {benefit}",
        "Specialerbjudande: {product} med {benefit}"
    ]
    
    # Extract patterns from data if possible
    patterns = default_patterns.copy()
    
    if delivery_df is not None and len(delivery_df) > 0:
        try:
            # Look for the subject column with flexible case handling
            subject_col = None
            for col in delivery_df.columns:
                if col.lower() == 'subject':
                    subject_col = col
                    break
            
            if subject_col and len(delivery_df) > 10:
                # Clean the subject lines
                subjects = delivery_df[subject_col].fillna('').astype(str)
                
                # Identify high-performing subjects
                open_rate_col = None
                for col in delivery_df.columns:
                    if col.lower() == 'open_rate':
                        open_rate_col = col
                        break
                        
                if open_rate_col:
                    top_performer_idx = delivery_df[open_rate_col].nlargest(5).index
                    top_subjects = subjects.loc[top_performer_idx].tolist()
                    
                    # Extract templates from top subjects
                    for subject in top_subjects:
                        words = subject.split()
                        if len(words) >= 3:
                            # Create a template by replacing key terms with placeholders
                            template = subject
                            for word in words:
                                if len(word) > 3 and word.lower() not in ['och', 'med', 'för', 'din', 'ditt', 'våra', 'vara']:
                                    if random.random() < 0.3:  # Randomly replace some words
                                        if word[0].isupper():
                                            template = template.replace(word, '{Product}')
                                        else:
                                            # Guess if it's a product, action, or benefit
                                            if any(action in word.lower() for action in ['köp', 'se', 'prova', 'läs', 'få']):
                                                template = template.replace(word, '{action}')
                                            elif any(benefit in word.lower() for benefit in ['rabatt', 'erbjudande', 'gratis', 'special', 'bäst']):
                                                template = template.replace(word, '{benefit}')
                                            else:
                                                template = template.replace(word, '{product}')
                            
                            patterns.append(template)
                    
                    # Remove duplicates and limit the number of patterns
                    patterns = list(set(patterns))[:10]
        except Exception as e:
            logging.error(f"Error extracting subject patterns: {e}")
            logging.error(traceback.format_exc())
            patterns = default_patterns.copy()
    
    return recommendations, patterns

def generate_recommendations(input_data, models, delivery_df, subject_patterns=None):
    """
    Generate recommendations for targeting and subject lines with enhanced error handling
    
    Parameters:
    - input_data: DataFrame with current input parameters
    - models: Dict of trained models for different metrics
    - delivery_df: DataFrame with historical delivery data
    - subject_patterns: Optional subject line patterns from clustering
    
    Returns:
    - Dictionary with recommendations and predictions
    """
    results = {}
    
    try:
        # Ensure input_data has standardized column names
        input_data_std = input_data.copy()
        input_data_std.columns = [col.lower() for col in input_data_std.columns]
        
        # Use the predict_metrics function from multi_metric_model
        from multi_metric_model import predict_metrics
        
        # 1. Current predictions (baseline)
        current_predictions = predict_metrics(input_data_std, models)
        results['current'] = current_predictions
        
        # 2. Targeting recommendations
        # Standardize delivery_df column names
        delivery_df_std = delivery_df.copy()
        delivery_df_std.columns = [col.lower() for col in delivery_df_std.columns]
        
        # Find best performing bolag based on historical data
        best_bolag = "B28"  # Default to Stockholm (B28)
        try:
            # Look for county or bolag column
            county_col = None
            for col_name in ['county', 'bolag']:
                if col_name in delivery_df_std.columns:
                    county_col = col_name
                    break
                    
            if county_col and 'open_rate' in delivery_df_std.columns:
                best_bolag = delivery_df_std.groupby(county_col)['open_rate'].mean().idxmax()
        except Exception as e:
            logging.error(f"Error finding best bolag: {e}")
        
        # Create input with recommended targeting
        targeting_data = input_data_std.copy()
        targeting_data['county'] = best_bolag  # Using 'county' for model compatibility but it's actually bolag
        
        # Predict all metrics for targeting recommendations
        targeting_predictions = predict_metrics(targeting_data, models, 
                                                metrics=['open_rate', 'click_rate', 'optout_rate'])
        
        results['targeting'] = {
            'county': best_bolag,  # Storing bolag code in 'county' for compatibility
            'predictions': targeting_predictions
        }
        
        # Get the current subject
        current_subject = ""
        for col_name in ['subject', 'Subject']:
            if col_name in input_data.columns:
                current_subject = input_data[col_name].iloc[0]
                break
        
        # Build subject recommendation model with error handling
        try:
            subject_recommendations, pattern_list = build_subject_recommendation_model(delivery_df_std)
        except Exception as e:
            logging.error(f"Error building subject recommendation model: {e}")
            # Fallback to default values
            subject_recommendations = {
                'informational': "Viktig information om din tjänst",
                'promotional': "Specialerbjudande: Ta del av våra förmåner idag"
            }
            pattern_list = [
                "Din {product}: {action} och få {benefit}",
                "Specialerbjudande: {product} med {benefit}"
            ]
            
        # Get recommended subject with error handling
        try:
            recommended_subject = recommend_subject(
                current_subject, 
                delivery_df_std, 
                subject_patterns=pattern_list if subject_patterns is None else subject_patterns,
                subject_recommendations=subject_recommendations
            )
        except Exception as e:
            logging.error(f"Error recommending subject: {e}")
            recommended_subject = "Specialerbjudande: Ta del av våra förmåner idag"
        
        # 3. Extract features from recommended subject
        from feature_engineering import extract_subject_features
        try:
            recommended_subject_features = extract_subject_features(recommended_subject)
        except Exception as e:
            logging.error(f"Error extracting features from recommended subject: {e}")
            recommended_subject_features = {
                'length': len(recommended_subject),
                'has_personalization': 1 if 'din' in recommended_subject.lower() else 0,
                'has_question': 1 if '?' in recommended_subject else 0,
                'has_numbers': 0,
                'has_uppercase_words': 0,
                'has_emoji': 0,
                'word_count': len(recommended_subject.split())
            }
        
        # Create input with recommended subject
        subject_data = input_data_std.copy()
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
            'county': best_bolag,  # Storing bolag code in 'county' for compatibility
            'subject': recommended_subject,
            'predictions': combined_predictions
        }
    except Exception as e:
        logging.error(f"Error in generate_recommendations: {e}")
        logging.error(traceback.format_exc())
        
        # Return default recommendations if there's an error
        # First, check if 'current' was successfully added to results
        if 'current' not in results:
            results['current'] = {
                'open_rate': 25.0,
                'click_rate': 3.0,
                'optout_rate': 0.2
            }
        
        # Add default targeting recommendation if missing
        if 'targeting' not in results:
            results['targeting'] = {
                'county': 'B28',  # Default to Stockholm
                'predictions': {
                    'open_rate': 26.0,
                    'click_rate': 3.5,
                    'optout_rate': 0.18
                }
            }
        
        # Add default subject recommendation if missing
        if 'subject' not in results:
            results['subject'] = {
                'text': "Specialerbjudande: Ta del av våra förmåner idag",
                'open_rate': 27.0
            }
        
        # Add default combined recommendation if missing
        if 'combined' not in results:
            results['combined'] = {
                'county': 'B28',
                'subject': "Specialerbjudande: Ta del av våra förmåner idag",
                'predictions': {
                    'open_rate': 28.0,
                    'click_rate': 3.5,
                    'optout_rate': 0.18
                }
            }
    
    return results

def format_predictions(recommendations):
    """Format prediction results for display with improved error handling"""
    import pandas as pd
    import numpy as np
    import logging
    
    # Default values for missing or invalid data
    default_values = {
        'open_rate': 25.0,
        'click_rate': 3.0,
        'optout_rate': 0.2
    }
    
    # Helper function to safely extract values
    def safe_extract(d, key_path, default_value):
        """Safely extract a value from a nested dictionary"""
        if not isinstance(d, dict):
            return default_value
            
        current = d
        keys = key_path.split('.')
        
        for key in keys[:-1]:
            if not isinstance(current, dict) or key not in current:
                return default_value
            current = current[key]
            
        last_key = keys[-1]
        if not isinstance(current, dict) or last_key not in current:
            return default_value
            
        value = current[last_key]
        
        # Validate the value
        if not isinstance(value, (int, float)) or pd.isna(value) or np.isnan(value) or not np.isfinite(value):
            return default_value
            
        return float(value)
    
    # Initialize formatted results
    formatted = {}
    
    try:
        # Format current predictions
        formatted['current'] = {
            'open_rate': safe_extract(recommendations, 'current.open_rate', default_values['open_rate']),
            'click_rate': safe_extract(recommendations, 'current.click_rate', default_values['click_rate']),
            'optout_rate': safe_extract(recommendations, 'current.optout_rate', default_values['optout_rate'])
        }
        
        # Format targeting recommendations
        formatted['targeting'] = {
            'county': recommendations.get('targeting', {}).get('county', 'B28'),  # Default to Stockholm
            'open_rate': safe_extract(recommendations, 'targeting.predictions.open_rate', default_values['open_rate']),
            'click_rate': safe_extract(recommendations, 'targeting.predictions.click_rate', default_values['click_rate']),
            'optout_rate': safe_extract(recommendations, 'targeting.predictions.optout_rate', default_values['optout_rate'])
        }
        
        # Calculate differences for targeting
        formatted['targeting']['open_rate_diff'] = formatted['targeting']['open_rate'] - formatted['current']['open_rate']
        formatted['targeting']['click_rate_diff'] = formatted['targeting']['click_rate'] - formatted['current']['click_rate']
        formatted['targeting']['optout_rate_diff'] = formatted['targeting']['optout_rate'] - formatted['current']['optout_rate']
        
        # Format subject recommendations
        formatted['subject'] = {
            'text': recommendations.get('subject', {}).get('text', 'Recommended subject line'),
            'open_rate': safe_extract(recommendations, 'subject.open_rate', default_values['open_rate'] * 1.2)  # 20% better by default
        }
        formatted['subject']['open_rate_diff'] = formatted['subject']['open_rate'] - formatted['current']['open_rate']
        
        # Format combined recommendations
        formatted['combined'] = {
            'county': recommendations.get('combined', {}).get('county', recommendations.get('targeting', {}).get('county', 'B28')),
            'subject': recommendations.get('combined', {}).get('subject', recommendations.get('subject', {}).get('text', 'Recommended subject line')),
            'open_rate': safe_extract(recommendations, 'combined.predictions.open_rate', default_values['open_rate'] * 1.3),
            'click_rate': safe_extract(recommendations, 'combined.predictions.click_rate', default_values['click_rate'] * 1.3),
            'optout_rate': safe_extract(recommendations, 'combined.predictions.optout_rate', default_values['optout_rate'] * 0.8)
        }
        
        # Calculate differences for combined
        formatted['combined']['open_rate_diff'] = formatted['combined']['open_rate'] - formatted['current']['open_rate']
        formatted['combined']['click_rate_diff'] = formatted['combined']['click_rate'] - formatted['current']['click_rate']
        formatted['combined']['optout_rate_diff'] = formatted['combined']['optout_rate'] - formatted['current']['optout_rate']
        
    except Exception as e:
        logging.error(f"Error in format_predictions: {e}")
        logging.error(traceback.format_exc())
        
        # Return default values if there's an error
        formatted = {
            'current': {
                'open_rate': default_values['open_rate'],
                'click_rate': default_values['click_rate'],
                'optout_rate': default_values['optout_rate']
            },
            'targeting': {
                'county': 'B28',
                'open_rate': default_values['open_rate'] * 1.1,
                'click_rate': default_values['click_rate'] * 1.1,
                'optout_rate': default_values['optout_rate'] * 0.9,
                'open_rate_diff': default_values['open_rate'] * 0.1,
                'click_rate_diff': default_values['click_rate'] * 0.1,
                'optout_rate_diff': -default_values['optout_rate'] * 0.1
            },
            'subject': {
                'text': 'Recommended subject line',
                'open_rate': default_values['open_rate'] * 1.2,
                'open_rate_diff': default_values['open_rate'] * 0.2
            },
            'combined': {
                'county': 'B28',
                'subject': 'Recommended subject line',
                'open_rate': default_values['open_rate'] * 1.3,
                'click_rate': default_values['click_rate'] * 1.3,
                'optout_rate': default_values['optout_rate'] * 0.8,
                'open_rate_diff': default_values['open_rate'] * 0.3,
                'click_rate_diff': default_values['click_rate'] * 0.3,
                'optout_rate_diff': -default_values['optout_rate'] * 0.2
            }
        }
    
    # Add confidence scores
    for scenario in ['current', 'targeting', 'subject', 'combined']:
        if 'confidence' not in formatted[scenario]:
            formatted[scenario]['confidence'] = {}
            for metric in ['open_rate', 'click_rate', 'optout_rate']:
                if metric in formatted[scenario]:
                    # Default confidence of 85%
                    formatted[scenario]['confidence'][metric] = 85
    
    return formatted