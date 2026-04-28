import os
from flask import Flask, jsonify, request, render_template, redirect, url_for, flash
import pandas as pd
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta

def get_ist_time():
    return datetime.utcnow() + timedelta(hours=5, minutes=30)

from ml.predict import predict_retention
from ml.train import retrain_model
from ml.synthetic import generate_synthetic_data
from ml.whatif_engine import get_best_windows

app = Flask(__name__, template_folder='../frontend/templates', static_folder='../frontend/static')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///cdt.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'super-secret-key'

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.String(50), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)

class StudyLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    sleep_hours = db.Column(db.Float)
    mood_score = db.Column(db.Float)
    energy_score = db.Column(db.Float)
    study_hour = db.Column(db.Integer)
    study_duration = db.Column(db.Float)
    subject_difficulty = db.Column(db.Float)
    retention_score = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=get_ist_time)

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

with app.app_context():
    db.create_all()

def count_logs(user_id):
    return StudyLog.query.filter_by(user_id=user_id).count()

def trigger_retrain(user_id, student_id):
    logs = StudyLog.query.filter_by(user_id=user_id).all()
    if not logs:
        return
    
    df = pd.DataFrame([{
        'sleep_hours': log.sleep_hours,
        'mood_score': log.mood_score,
        'energy_score': log.energy_score,
        'study_hour': log.study_hour,
        'study_duration': log.study_duration,
        'subject_difficulty': log.subject_difficulty,
        'retention_score': log.retention_score
    } for log in logs])
    
    retrain_model(student_id, df)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        student_id = request.form.get('student_id')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if not student_id or not password:
            flash("Missing student_id or password", "error")
            return redirect(url_for('register'))
            
        if password != confirm_password:
            flash("Passwords do not match", "error")
            return redirect(url_for('register'))
            
        if User.query.filter_by(student_id=student_id).first():
            flash("Student ID already exists", "error")
            return redirect(url_for('register'))
            
        new_user = User(
            student_id=student_id,
            password_hash=generate_password_hash(password)
        )
        db.session.add(new_user)
        db.session.commit()
        
        df = generate_synthetic_data(500)
        retrain_model(student_id, df)
        
        login_user(new_user)
        flash(f"Welcome to CogniSync, {student_id}!", "success")
        return redirect(url_for('dashboard'))
        
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        student_id = request.form.get('student_id')
        password = request.form.get('password')
        
        user = User.query.filter_by(student_id=student_id).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            flash(f"Welcome back, {user.student_id}!", "success")
            return redirect(url_for('dashboard'))
            
        flash("Invalid credentials", "error")
        return redirect(url_for('login'))
        
    return render_template('login.html')

@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/log', methods=['GET', 'POST'])
@login_required
def add_log():
    if request.method == 'POST':
        # Map Subject Complexity string to float values
        complexity_map = {'low': 1.0, 'medium': 3.0, 'high': 5.0}
        subject_difficulty = complexity_map.get(request.form.get('subject_complexity'), 3.0)
        
        new_log = StudyLog(
            user_id=current_user.id,
            sleep_hours=float(request.form.get('sleep_hours')),
            mood_score=float(request.form.get('mood_score')),
            energy_score=float(request.form.get('energy_score')),
            study_hour=int(request.form.get('study_hour')),
            study_duration=float(request.form.get('study_duration')),
            subject_difficulty=subject_difficulty,
            retention_score=float(request.form.get('retention_score'))
        )
        db.session.add(new_log)
        db.session.commit()
        
        count = count_logs(current_user.id)
        
        if count == 14 or (count > 14 and count % 7 == 0):
            print(f"TRIGGERING RETRAIN for {current_user.student_id} (Log count: {count})")
            trigger_retrain(current_user.id, current_user.student_id)
            
        flash("Neural Data Synchronized", "success")
        return redirect(url_for('dashboard'))
        
    logs = StudyLog.query.filter_by(user_id=current_user.id).order_by(StudyLog.id.desc()).limit(10).all()
    return render_template('log.html', logs=logs)

@app.route('/log/<int:log_id>/delete', methods=['POST'])
@login_required
def delete_log(log_id):
    log = db.session.get(StudyLog, log_id)
    if log and log.user_id == current_user.id:
        db.session.delete(log)
        db.session.commit()
        flash("Log deleted successfully", "success")
    return redirect(url_for('add_log'))

@app.route('/dashboard', methods=['GET'])
@login_required
def dashboard():
    last_log = StudyLog.query.filter_by(user_id=current_user.id).order_by(StudyLog.id.desc()).first()
    if last_log:
        current_conditions = {
            'sleep_hours': last_log.sleep_hours,
            'mood_score': last_log.mood_score,
            'energy_score': last_log.energy_score,
            'study_duration': last_log.study_duration,
            'subject_difficulty': last_log.subject_difficulty
        }
    else:
        current_conditions = {
            'sleep_hours': 7.0,
            'mood_score': 5.0,
            'energy_score': 5.0,
            'study_duration': 2.0,
            'subject_difficulty': 3.0
        }
    
    best_windows = get_best_windows(current_user.student_id, current_conditions)
    best_time = f"{best_windows[0]['hour']}:00" if best_windows else "10:00"
    
    log_count = count_logs(current_user.id)
    progress_percentage = min((log_count / 14) * 100, 100)
    
    shap_path = f"{current_user.student_id}_shap.png"
    
    return render_template('dashboard.html', 
                           student_id=current_user.student_id,
                           best_time=best_time,
                           log_count=log_count,
                           progress_percentage=progress_percentage,
                           shap_path=shap_path)

@app.route('/api/dashboard-stats', methods=['GET'])
@login_required
def dashboard_stats():
    logs = StudyLog.query.filter_by(user_id=current_user.id).order_by(StudyLog.id.desc()).limit(7).all()
    logs.reverse()
    
    data = {
        'labels': [f"Log {log.id}" for log in logs] if logs else ["Baseline"],
        'scores': [log.retention_score for log in logs] if logs else [50]
    }
    return jsonify(data)

@app.route('/test-predict')
def test_predict():
    dummy_data = {
        'sleep_hours': 8,
        'mood_score': 4,
        'energy_score': 4,
        'study_hour': 10,
        'study_duration': 2,
        'subject_difficulty': 1
    }
    result = predict_retention('student_001', dummy_data)
    return jsonify({'student_id': 'student_001', 'predicted_retention': result})

@app.route('/simulator', methods=['GET'])
@login_required
def simulator():
    return render_template('simulator.html')

@app.route('/api/predict-instant', methods=['POST'])
@login_required
def predict_instant():
    data = request.json
    
    # Run the main prediction
    main_pred = predict_retention(current_user.student_id, data)
    
    if main_pred is None:
        return jsonify({'error': 'Model not trained yet'}), 400
        
    # Generate sweep data for the Correlation Map (sweeping 'sleep_hours' from 3.0 to 12.0)
    sweep_labels = []
    sweep_scores = []
    
    sweep_data = data.copy()
    for s_int in range(6, 25):  # 3.0 to 12.0 with step 0.5
        s = s_int / 2.0
        sweep_data['sleep_hours'] = s
        score = predict_retention(current_user.student_id, sweep_data)
        if score is not None:
            sweep_labels.append(f"{s}h")
            sweep_scores.append(score)
            
    return jsonify({
        'predicted_retention': main_pred,
        'sweep_labels': sweep_labels,
        'sweep_scores': sweep_scores
    })

if __name__ == '__main__':
    app.run(debug=True)
