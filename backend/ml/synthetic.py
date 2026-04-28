import pandas as pd
import numpy as np

def generate_synthetic_data(num_samples=500):
    np.random.seed(42)
    
    sleep_hours = np.random.normal(7, 1.5, num_samples).clip(3, 10)
    mood_score = np.random.randint(1, 11, num_samples)
    energy_score = np.random.randint(1, 11, num_samples)
    study_hour = np.random.randint(0, 24, num_samples)
    study_duration = np.random.uniform(0.5, 4.0, num_samples)
    subject_difficulty = np.random.uniform(1.0, 5.0, num_samples)
    
    base_score = 40 + (sleep_hours - 3) * 5 + (energy_score * 2)
    
    morning_boost = np.where((study_hour >= 8) & (study_hour <= 12), 15, 0)
    night_penalty = np.where(study_hour >= 21, -10, 0)
    
    difficulty_penalty = subject_difficulty * 4
    duration_penalty = np.where(study_duration > 2.5, (study_duration - 2.5) * 5, 0)
    
    noise = np.random.normal(0, 5, num_samples)
    
    retention_score = (base_score + morning_boost + night_penalty - difficulty_penalty - duration_penalty + noise)
    retention_score = np.clip(retention_score, 0, 100)
    
    df = pd.DataFrame({
        'sleep_hours': sleep_hours,
        'mood_score': mood_score,
        'energy_score': energy_score,
        'study_hour': study_hour,
        'study_duration': study_duration,
        'subject_difficulty': subject_difficulty,
        'retention_score': retention_score
    })
    
    return df

if __name__ == '__main__':
    df = generate_synthetic_data(500)
    print("Generated 500 rows of synthetic data.")
    print(df.head())
