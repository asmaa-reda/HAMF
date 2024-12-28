
from src.data_collection.collect_data import collect_data
from src.data_preprocessing.preprocess_data import preprocess_data
from src.feature_optimization.optimize_features import optimize_features
from src.model_training.train_model import train_model
from src.performance_monitoring.monitor_performance import monitor_performance

# Step 1: Collect Data
sample_data = {
    "feature1": [1, 2, 3, 4, 5],
    "feature2": [5, 4, 3, 2, 1],
    "target": [0, 1, 0, 1, 0]
}

# Step 2: Preprocess Data
X_train, X_test, y_train, y_test = preprocess_data(sample_data, target_column="target")

# Step 3: Optimize Features
X_train_reduced = optimize_features(X_train)
X_test_reduced = optimize_features(X_test)

# Step 4: Train Model
model, report = train_model(X_train_reduced, y_train, X_test_reduced, y_test)
print("Classification Report:", report)

# Step 5: Monitor Performance
monitor_performance(model, X_test_reduced, y_test)
