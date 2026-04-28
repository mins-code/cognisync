import os
import glob
import joblib
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def generate_shap_charts():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, 'models')
    static_dir = os.path.join(os.path.dirname(base_dir), 'frontend', 'static')
    os.makedirs(static_dir, exist_ok=True)
    
    # Generate random data just to plot the SHAP values correctly
    # SHAP explainer requires background data for some models, but for TreeExplainer on RandomForest
    # we can use dummy data if needed, or better, we can just use the synthetic data generator.
    import sys
    sys.path.append(base_dir)
    from ml.synthetic import generate_synthetic_data
    from sklearn.preprocessing import StandardScaler
    
    # We will generate synthetic data, scale it, and use it for the SHAP summary plot
    dummy_df = generate_synthetic_data(100)
    features = [
        'sleep_hours', 'mood_score', 'energy_score', 'study_hour', 
        'study_duration', 'subject_difficulty', 'sleep_quality_proxy', 
        'cognitive_load', 'is_morning', 'is_night'
    ]
    # Add generated features
    dummy_df['sleep_quality_proxy'] = dummy_df['sleep_hours'] * dummy_df['mood_score']
    dummy_df['cognitive_load'] = dummy_df['study_duration'] * dummy_df['subject_difficulty']
    dummy_df['is_morning'] = (dummy_df['study_hour'] < 12).astype(int)
    dummy_df['is_night'] = (dummy_df['study_hour'] >= 21).astype(int)
    X = dummy_df[features]
    
    for model_file in glob.glob(os.path.join(models_dir, '*_model.pkl')):
        filename = os.path.basename(model_file)
        student_id = filename.replace('_model.pkl', '')
        
        print(f"Generating SHAP for {student_id}...")
        try:
            model = joblib.load(model_file)
            scaler_file = os.path.join(models_dir, f'{student_id}_scaler.pkl')
            scaler = joblib.load(scaler_file)
            
            X_scaled = scaler.transform(X.values)
            
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_scaled)
            
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_scaled, feature_names=features, plot_type="bar", show=False)
            
            shap_path = os.path.join(static_dir, f'{student_id}_shap.png')
            plt.tight_layout()
            plt.savefig(shap_path, bbox_inches='tight')
            plt.close()
            print(f"Saved {shap_path}")
        except Exception as e:
            print(f"Failed for {student_id}: {e}")

if __name__ == '__main__':
    generate_shap_charts()
