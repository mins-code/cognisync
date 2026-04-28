import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def retrain_model(student_id, df):
    if df.empty or len(df) < 5:
        print(f"Not enough data to retrain for {student_id}")
        return False
        
    df['sleep_quality_proxy'] = df['sleep_hours'] * df['mood_score']
    df['cognitive_load'] = df['study_duration'] * df['subject_difficulty']
    df['is_morning'] = (df['study_hour'] < 12).astype(int)
    df['is_night'] = (df['study_hour'] >= 21).astype(int)
    
    features = [
        'sleep_hours', 'mood_score', 'energy_score', 'study_hour', 
        'study_duration', 'subject_difficulty', 'sleep_quality_proxy', 
        'cognitive_load', 'is_morning', 'is_night'
    ]
    
    X = df[features]
    y = df['retention_score']
    
    scaler = StandardScaler()
    # Strip feature names
    X_scaled = scaler.fit_transform(X.values)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    base_dir = os.path.dirname(os.path.dirname(__file__))
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, f'{student_id}_model.pkl')
    scaler_path = os.path.join(models_dir, f'{student_id}_scaler.pkl')
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_scaled)
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_scaled, feature_names=features, plot_type="bar", show=False)
        
        static_dir = os.path.join(os.path.dirname(base_dir), 'frontend', 'static')
        os.makedirs(static_dir, exist_ok=True)
        shap_path = os.path.join(static_dir, f'{student_id}_shap.png')
        
        plt.tight_layout()
        plt.savefig(shap_path, bbox_inches='tight')
        plt.close()
        print(f"SHAP plot saved to {shap_path}")
    except Exception as e:
        print(f"Error generating SHAP plot: {e}")
    
    print(f"Retraining successful for {student_id}. Models saved to {models_dir}")
    return True
