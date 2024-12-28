## Scenario Workflow

This workflow outlines the steps to integrate a pre-trained model into the HAMF pipeline for real-time predictions and then monitor its performance to be retrained.

**1. Leverage the Existing Trained Model**

*   **Objective:** Load the trained model (stored as a `.pkl` file) and integrate it into the HAMF pipeline for real-time predictions.

*   **Steps:**
    *   **Store the .pkl model artifact in MLflow:**
        *   Log the trained model file (`.pkl`) as an artifact within an MLflow run. This allows for versioning, tracking, and retrieval of the model.
    *   **Deploy the model as a REST API for predictions:**
        *   Utilize MLflow's deployment capabilities to serve the model as a REST API endpoint. This enables other applications or services to send data and receive predictions in real-time.

---

**Loading the Existing Model**
```python
import pickle
import mlflow.pyfunc

# Load the pickle file
with open("path_to_model.pkl", "rb") as file:
    model = pickle.load(file)

# Save the model in MLflow
mlflow.pyfunc.log_model(
    artifact_path="phishing_model",
    python_model=model,
    registered_model_name="PhishingModel"
)

```

**2. Continuous Data Collection**
*   **Objective:** Collect new data from multiple sources, extract features, and save them in PostgreSQL.

*   **Tools:** Selenium (for web scraping), Custom API integration, and PostgreSQL.
---
**Data Collection Pipeline**
***Docker Services for Collectors:***
1. **Custom Collector:** Collects raw data from APIs or other structured sources.
2. **Selenium Scraper:** Scrapes websites to gather additional data.

***Implementation Example:***
```python
import requests
import psycopg2
from selenium import webdriver
from psycopg2.extras import execute_values

# Connect to PostgreSQL
conn = psycopg2.connect(
    host="localhost", dbname="hamf_db", user="postgres", password="postgres"
)

# Data Collector Function
def collect_data():
    # Example API data collection
    response = requests.get("https://api.example.com/data")
    raw_data = response.json()

    # Store raw data in PostgreSQL
    with conn.cursor() as cur:
        execute_values(
            cur,
            "INSERT INTO raw_data (source, collected_date, data) VALUES %s",
            [(source, collected_date, raw_data)],
        )
        conn.commit()

# Selenium Scraper Example
def scrape_website():
    driver = webdriver.Chrome()
    driver.get("https://example.com")
    scraped_data = driver.page_source
    # Process and store data in PostgreSQL
    with conn.cursor() as cur:
        cur.execute("INSERT INTO raw_data (source, collected_date, data) VALUES (%s, %s, %s)", ("website", "2024-01-01", scraped_data))
        conn.commit()
    driver.quit()

collect_data()
scrape_website()

```
---

## 3. Feature Extraction

### Objective
Process collected data to extract all features listed in PostgreSQL and generate a monthly CSV file for analysis.

### Implementation
1. **Read feature definitions** from the `features_master` table.
2. **Apply transformations** and generate datasets.
3. **Feature Extraction Script:**
    ```python
    import pandas as pd
    import psycopg2

    # Connect to PostgreSQL
    conn = psycopg2.connect(
        host="localhost", dbname="hamf_db", user="postgres", password="postgres"
    )

    def extract_features():
        # Read features from PostgreSQL
        query = "SELECT feature_name FROM features_master WHERE feature_status='active'"
        features = []
        with conn.cursor() as cur:
            cur.execute(query)
            features = [row[0] for row in cur.fetchall()]

        # Example transformation
        data = pd.read_sql("SELECT * FROM raw_data", conn)
        transformed_data = data[features]  # Extract relevant features
        transformed_data.to_csv("/app/monthly_dataset.csv", index=False)

    extract_features()

    ```

### Notes
- The extracted features are stored in the PostgreSQL database under the `extracted_features` table.
- Monitoring is conducted using Grafana to ensure performance and feature relevance.
- Feature retirement is tracked, and if invalid features are detected, the system triggers alerts and adjusts the feature set accordingly&#8203;:contentReference[oaicite:0]{index=0}.
---

## 4. Performance Evaluation

### Objective
Evaluate the model’s performance on new datasets monthly.

### Implementation
1. **Load Monthly Dataset**  
   - Retrieve the latest monthly dataset from PostgreSQL or CSV storage.  
   - Ensure the dataset is preprocessed and aligned with feature expectations.

2. **Run Predictions**  
   - Apply the phishing detection model to the new dataset.  
   - Use the model to generate predictions and output results.  

3. **Calculate Performance Metrics**  
   - Calculate key evaluation metrics, including:  
     - **Accuracy**  
     - **Precision**  
     - **Recall**  
     - **F1-Score**  
     - **AUC-ROC** 
**Evaluation Script:**
```python
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Load new dataset and trained model
data = pd.read_csv("/app/monthly_dataset.csv")
X = data.drop("phishing", axis=1)
y_true = data["phishing"]

# Load model from MLflow
model = mlflow.pyfunc.load_model("models:/PhishingModel/production")

# Predict and evaluate
y_pred = model.predict(X)
accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)

print(f"Accuracy: {accuracy}, F1: {f1}, Precision: {precision}, Recall: {recall}")

```
---

## 5. Notify Owners of Degraded Performance
### Objective: 
Use Slack and SMTP to alert model owners if performance decreases.
**Example Alert Logic:**
```python
import requests
import smtplib

# Thresholds
ACCURACY_THRESHOLD = 0.98

# Alert function
def send_alert(accuracy):
    if accuracy < ACCURACY_THRESHOLD:
        # Send Slack notification
        slack_webhook_url = "https://hooks.slack.com/services/your/webhook"
        slack_message = {"text": f"Model performance degraded! Current accuracy: {accuracy:.2%}"}
        requests.post(slack_webhook_url, json=slack_message)

        # Send SMTP email notification
        smtp_server = "smtp.example.com"
        sender_email = "your_email@example.com"
        receiver_email = "model_owner@example.com"
        message = f"Subject: Model Performance Alert\n\nModel performance degraded! Current accuracy: {accuracy:.2%}"

        with smtplib.SMTP(smtp_server, 587) as server:
            server.starttls()
            server.login("your_email@example.com", "your_password")
            server.sendmail(sender_email, receiver_email, message)

send_alert(accuracy)
```

## 6. Automate Retraining Suggestions
If performance degrades:

1. Suggest retraining.
2. Generate and log feature importance rankings.
3. Propose feature engineering.
**Example Feature Importance Script:**
```python
import numpy as np

# Feature importance
feature_importances = model.feature_importances_
important_features = sorted(zip(X.columns, feature_importances), key=lambda x: -x[1])

print("Feature Importance Rankings:")
for feature, importance in important_features:
    print(f"{feature}: {importance:.4f}")

```

