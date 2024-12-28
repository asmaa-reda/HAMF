import os

# -------------------------------
# DATABASE CONFIGURATION
# -------------------------------
DATABASE = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', 5432),
    'user': os.getenv('DB_USER', 'admin'),
    'password': os.getenv('DB_PASSWORD', 'password'),
    'dbname': os.getenv('DB_NAME', 'hamf_db')
}

# -------------------------------
# MLflow CONFIGURATION
# -------------------------------
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', "http://localhost:5000")
EXPERIMENT_NAME = os.getenv('MLFLOW_EXPERIMENT_NAME', "Phishing Detection Experiment")

# -------------------------------
# SLACK CONFIGURATION
# -------------------------------
SLACK_CONFIG = {
    "webhook_url": os.getenv('SLACK_WEBHOOK_URL', "https://hooks.slack.com/services/...")
}

# -------------------------------
# SMTP CONFIGURATION (EMAIL ALERTS)
# -------------------------------
SMTP_CONFIG = {
    "server": os.getenv('SMTP_SERVER', "smtp.example.com"),
    "port": os.getenv('SMTP_PORT', 587),
    "username": os.getenv('SMTP_USERNAME', "alert@example.com"),
    "password": os.getenv('SMTP_PASSWORD', "password123")
}

# -------------------------------
# PROMETHEUS CONFIGURATION
# -------------------------------
PROMETHEUS_EXPORT_PORT = os.getenv('PROMETHEUS_PORT', 8000)

# -------------------------------
# MODEL CONFIGURATION
# -------------------------------
MODEL_CONFIG = {
    "model_name": os.getenv('MODEL_NAME', "phishing_detection_model"),
    "n_estimators": int(os.getenv('MODEL_ESTIMATORS', 1000)),  # GradientBoosting iterations
    "random_state": int(os.getenv('MODEL_RANDOM_STATE', 42))
}

# -------------------------------
# DATA PATHS
# -------------------------------
DATA_PATHS = {
    "raw_data_path": os.getenv('RAW_DATA_PATH', "/data/raw_data.csv"),
    "processed_data_path": os.getenv('PROCESSED_DATA_PATH', "/data/processed_data.csv"),
    "features_csv_path": os.getenv('FEATURES_CSV_PATH', "/data/features.csv")
}

# -------------------------------
# ALERT THRESHOLDS
# -------------------------------
ALERT_THRESHOLDS = {
    "accuracy_threshold": float(os.getenv('ACCURACY_THRESHOLD', 98.0)),
    "f1_score_threshold": float(os.getenv('F1_THRESHOLD', 0.95))
}
