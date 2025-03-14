def build_subject_recommendation_model(delivery_df):
    """Build a model to recommend high-performing subject lines"""
    import pandas as pd
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    
    # Check if we have enough data for meaningful analysis
    if len(delivery_df) < 10:
        return None, None
    
    # Process subject lines using TF-IDF
    tfidf = TfidfVectorizer(max_features=50, stop_words=['english', 'swedish'])
    subject_vectors = tfidf.fit_transform(delivery_df['subject'].fillna(''))
    
    # Combine with open rate for analysis
    subject_data = pd.DataFrame(subject_vectors.toarray(), columns=tfidf.get_feature_names_out())
    subject_data['open_rate'] = delivery_df['open_rate']
    
    # Cluster subject lines (find patterns)
    n_clusters = min(5, len(delivery_df) // 2)  # Avoid too many clusters for small datasets
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    subject_data['cluster'] = kmeans.fit_predict(subject_vectors)
    
    # Find best performing clusters
    cluster_performance = subject_data.groupby('cluster')['open_rate'].mean().sort_values(ascending=False)
    
    # For each cluster, get example subject lines that performed well
    recommended_subjects = []
    for cluster in cluster_performance.index:
        # Get subjects from this cluster, sorted by open rate
        cluster_mask = subject_data['cluster'] == cluster
        if cluster_mask.sum() > 0:
            cluster_subjects = pd.DataFrame({
                'subject': delivery_df.loc[cluster_mask.values, 'subject'],
                'open_rate': subject_data.loc[cluster_mask, 'open_rate']
            }).sort_values('open_rate', ascending=False)
            
            # Take the top performing subject from this cluster
            if len(cluster_subjects) > 0:
                top_subject = cluster_subjects.iloc[0]['subject']
                expected_open_rate = cluster_subjects.iloc[0]['open_rate']
                recommended_subjects.append({
                    'subject': top_subject,
                    'expected_open_rate': expected_open_rate,
                    'cluster': cluster
                })
    
    # Sort recommendations by expected open rate
    recommended_subjects = sorted(recommended_subjects, key=lambda x: x['expected_open_rate'], reverse=True)
    
    # Extract subject line patterns that work well
    subject_patterns = []
    for cluster in cluster_performance.index[:3]:  # Top 3 clusters
        # Get all subjects in this cluster
        cluster_subjects = delivery_df.loc[subject_data['cluster'] == cluster, 'subject'].tolist()
        
        # Find common words/patterns
        if cluster_subjects:
            common_words = set(cluster_subjects[0].lower().split())
            for subject in cluster_subjects[1:]:
                common_words &= set(subject.lower().split())
            
            subject_patterns.append({
                'cluster': cluster,
                'avg_open_rate': cluster_performance[cluster],
                'common_words': list(common_words),
                'example_subjects': cluster_subjects[:3]
            })
    
    return recommended_subjects, subject_patterns

def recommend_subject(input_subject, delivery_df, recommended_subjects, subject_patterns):
    """Generate subject line recommendations based on patterns and top performers"""
    if not recommended_subjects:
        # Fallback to basic recommendations if no model available
        if len(input_subject) < 30:
            return "Your Exclusive Offer: Don't Miss Out!"
        elif "?" not in input_subject:
            return input_subject.strip() + "?"
        else:
            return input_subject
    
    # First try to match input with known patterns
    input_words = set(input_subject.lower().split())
    
    best_recommendation = None
    highest_open_rate = 0
    
    # Check if input matches any known good patterns
    for pattern in subject_patterns:
        if set(pattern['common_words']).issubset(input_words):
            # Input contains pattern words, find the best subject from this cluster
            for subj in recommended_subjects:
                if subj['cluster'] == pattern['cluster'] and subj['expected_open_rate'] > highest_open_rate:
                    best_recommendation = subj['subject']
                    highest_open_rate = subj['expected_open_rate']
    
    # If no pattern match, return the top performing subject
    if not best_recommendation and recommended_subjects:
        best_recommendation = recommended_subjects[0]['subject']
    
    # If still no recommendation (shouldn't happen), use the original
    return best_recommendation or input_subject