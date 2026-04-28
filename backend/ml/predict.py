import os
import warnings
import joblib
import pandas as pd
import numpy as np
from sklearn.exceptions import InconsistentVersionWarning

# Suppress warnings for loading older models
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

def predict_retention(student_id, data_dict):
    base_dir = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(base_dir, 'models', f'{student_id}_model.pkl')
    scaler_path = os.path.join(base_dir, 'models', f'{student_id}_scaler.pkl')
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    df = pd.DataFrame([data_dict])
    
    df['sleep_quality_proxy'] = df['sleep_hours'] * df['mood_score']
    df['cognitive_load'] = df['study_duration'] * df['subject_difficulty']
    df['is_morning'] = (df['study_hour'] < 12).astype(int)
    df['is_night'] = (df['study_hour'] >= 21).astype(int)
    
    features = [
        'sleep_hours', 'mood_score', 'energy_score', 'study_hour', 
        'study_duration', 'subject_difficulty', 'sleep_quality_proxy', 
        'cognitive_load', 'is_morning', 'is_night'
    ]
    df = df[features]
    
    # Use df.values to avoid feature names warning if scaler was fit without them
    X_scaled = scaler.transform(df.values)
    pred = model.predict(X_scaled)[0]
    
    return float(np.clip(pred, 0, 100))
