import json
import os
import datetime
import hashlib
import random
import pandas as pd

def get_model_version():
    """Generate a version identifier for models"""
    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    random_suffix = hashlib.md5(str(random.random()).encode()).hexdigest()[:6]
    return f"{timestamp}-{random_suffix}"

def save_model_metadata(model_results, save_path="saved_models/model_metadata.json"):
    """Save model metadata for tracking and version control"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    version = model_results.get('version', get_model_version())
    metadata = {
        "version": version,
        "created_at": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "data_stats": {
            "num_campaigns": model_results.get('num_campaigns', 0),
            "feature_count": len(model_results.get('feature_names', [])),
            "categorical_features": list(model_results.get('categorical_values', {}).keys())
        },
        "performance": model_results.get('performance', {}),
        "model_types": {
            metric: type(model).__name__
            for metric, model in model_results.get('models', {}).items()
        }
    }
    with open(save_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    return metadata

def load_model_metadata(path="saved_models/model_metadata.json"):
    """Load model metadata"""
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r') as f:
            metadata = json.load(f)
        return metadata
    except Exception as e:
        print(f"Error loading model metadata: {e}")
        return None

def track_model_performance(model_results, actual_metrics=None):
    """Track and log model performance for future analysis"""
    performance_log_path = "saved_models/performance_log.csv"
    log_data = {
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'version': model_results.get('version', 'unknown')
    }
    if 'performance' in model_results:
        for metric, results in model_results['performance'].items():
            log_data[f'{metric}_mae'] = results.get('mae', None)
            log_data[f'{metric}_std'] = results.get('std', None)
    if actual_metrics:
        for metric, value in actual_metrics.items():
            log_data[f'actual_{metric}'] = value
    log_df = pd.DataFrame([log_data])
    if os.path.exists(performance_log_path):
        existing_log = pd.read_csv(performance_log_path)
        updated_log = pd.concat([existing_log, log_df], ignore_index=True)
        updated_log.to_csv(performance_log_path, index=False)
    else:
        os.makedirs(os.path.dirname(performance_log_path), exist_ok=True)
        log_df.to_csv(performance_log_path, index=False)
    return log_data

def model_needs_retraining(performance_log_path="saved_models/performance_log.csv", threshold=0.05):
    """Check if model should be retrained based on performance degradation"""
    if not os.path.exists(performance_log_path):
        return False, "No performance history available"
    try:
        log_df = pd.read_csv(performance_log_path)
        if len(log_df) < 5:
            return False, "Not enough performance history to make a decision"
        metrics = [col.replace('_mae', '') for col in log_df.columns if col.endswith('_mae')]
        for metric in metrics:
            mae_col = f'{metric}_mae'
            if mae_col not in log_df.columns:
                continue
            first_entries = log_df.head(5)[mae_col].mean()
            recent_entries = log_df.tail(5)[mae_col].mean()
            if first_entries > 0:
                degradation = (recent_entries - first_entries) / first_entries
                if degradation > threshold:
                    return True, f"{metric} performance has degraded by {degradation*100:.1f}%"
        return False, "Model performance is stable"
    except Exception as e:
        return False, f"Error analyzing performance history: {str(e)}"