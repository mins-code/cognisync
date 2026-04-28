from .predict import predict_retention

def get_best_windows(student_id, current_conditions):
    predictions = []
    
    for hour in range(6, 24):
        test_conditions = current_conditions.copy()
        test_conditions['study_hour'] = hour
        
        score = predict_retention(student_id, test_conditions)
        
        if score is not None:
            predictions.append({
                'hour': hour,
                'predicted_score': score
            })
            
    predictions.sort(key=lambda x: x['predicted_score'], reverse=True)
    return predictions[:3]
