
# HAMF Framework – Comprehensive Implementation Plan

## Objective
Enhance the HAMF framework to perform **continuous phishing detection** using a pre-trained model. Automate data collection, retraining, performance monitoring, and alerting to ensure the model is regularly evaluated with new data.

---

## 1. Environment Setup

**Tools and Technologies:**
- **Data Collection:** Selenium, Requests, Tweepy (Twitter API)
- **Preprocessing:** Apache Spark, Pandas, NumPy
- **Feature Extraction:** Scikit-learn, SHAP, Python WHOIS
- **Model Training & Retraining:** TensorFlow, Scikit-learn
- **Monitoring:** Grafana, Prometheus, MLflow
- **Storage:** MinIO, PostgreSQL, Elasticsearch
- **Security & Backup:** OpenSSL, IAM/LDAP, Amanda Backup
- **Deployment:** Docker, GitLab CI/CD
- **Communication:** Slack, Trello, Email (SMTP)
- **Documentation:** BookStack, Swagger

---

## Phase 1: Environment Setup

### 1. Clone and Setup Repository
```bash
git clone https://github.com/asmaareda/HAMF.git
cd HAMF
```

### 2. Configure Environment Variables
Ensure the `.env` file is populated correctly:
```plaintext
DB_HOST=localhost
DB_PORT=5432
DB_USER=admin
DB_PASSWORD=secretpassword
DB_NAME=hamf_db
MLFLOW_TRACKING_URI=http://localhost:5000
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/xxx/xxx/xxx
SMTP_SERVER=smtp.example.com
SMTP_USERNAME=alert@example.com
SMTP_PASSWORD=password123
ACCURACY_THRESHOLD=98.0
F1_THRESHOLD=0.95
```

### 3. Build and Deploy Docker Containers
```bash
cd docker/
docker-compose up --build -d
```
- **MLflow** will be available at [http://localhost:5000](http://localhost:5000).
- **PostgreSQL** initialized for model tracking.

### 4. Create Database Schema
```bash
docker exec -it postgres psql -U postgres -d hamf_db -f /docker/services/data_management/database/init.sql
```
---

## Phase 2: Implementation Flow


### Step 1: Data Collection and Preprocessing

*Description:* Collect URLs (both phishing and non-phishing) and extract basic features like URL length, special character counts, and domain age using WHOIS.

```python
from selenium import webdriver
from whois import whois
import pandas as pd
import requests

# Collect Data from URLs
urls = ['https://example.com', 'https://phishing.com']
url_data = []

for url in urls:
    driver = webdriver.Chrome()
    driver.get(url)
    url_data.append({
        'url': url,
        'length': len(url),
        'special_chars': sum(1 for char in url if char in '!@#$%^&*')
    })
    driver.close()

# WHOIS Data Extraction
for item in url_data:
    domain_info = whois(item['url'])
    item['domain_age'] = (pd.Timestamp.now() - pd.to_datetime(domain_info.creation_date)).days

url_df = pd.DataFrame(url_data)
url_df.to_csv('processed_urls.csv', index=False)
```

---

### Step 2: Feature Extraction and Storage
*Description:* After preprocessing, extract features and store them in PostgreSQL. Use SHAP to evaluate feature importance dynamically.

```python
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import StandardScaler
import psycopg2

# Load Data
url_df = pd.read_csv('processed_urls.csv')
X = url_df[['length', 'special_chars', 'domain_age']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

selector = SelectKBest(chi2, k=2)
X_new = selector.fit_transform(X_scaled, [1, 0])

# Store Features in PostgreSQL
conn = psycopg2.connect(database='hamf', user='admin', password='pass')
cursor = conn.cursor()
for _, row in url_df.iterrows():
    cursor.execute(
        "INSERT INTO features (url, length, special_chars, domain_age) VALUES (%s, %s, %s, %s)",
        (row['url'], row['length'], row['special_chars'], row['domain_age'])
    )
conn.commit()
```

---

### Step 3: Model Training and Retraining
*Description:* Train the initial model using Random Forest. Implement automated retraining when accuracy falls below a defined threshold.

**Train and Register Model**

```python
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load

import pandas as pd
import mlflow
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from db_operations import connect_db
from config.settings import MLFLOW_TRACKING_URI


# Train Model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_new, [1, 0])
dump(model, 'phishing_model_v1.pkl')

# Automated Retraining
model = load('phishing_model_v1.pkl')
if model.score(X_new, [1, 0]) < 0.95:
    print("Retraining Model...")
    model.fit(X_new, [1, 0])
    dump(model, 'phishing_model_v2.pkl')



mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Phishing Detection Experiment")

data = pd.read_csv("/data/features.csv")
X = data.drop("phishing", axis=1)
y = data["phishing"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

with mlflow.start_run():
    model = GradientBoostingClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "phishing_model")
```

---

### Step 4: Monitoring and Alerts
*Description:* Use Prometheus to monitor model performance and trigger email alerts via SMTP when performance degrades.

```python
import smtplib
from prometheus_client import start_http_server, Summary

start_http_server(8000)
REQUEST_TIME = Summary('request_processing_seconds', 'Time spent processing request')

@REQUEST_TIME.time()
def process_request():
    if model.score(X_new, [1, 0]) < 0.95:
        with smtplib.SMTP('smtp.example.com') as server:
            server.sendmail(
                "alert@example.com",
                "admin@example.com",
                "Model Accuracy Dropped"
            )

```

---

### Step 5: Stakeholder Communication
*Description:* Communicate alerts and retraining triggers to stakeholders via Slack channels.

```python
import requests

# Notify stakeholders via Slack
slack_webhook = 'https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX'
message = {
    'text': "HAMF Alert: Model retraining triggered due to performance drop."
}
requests.post(slack_webhook, json=message)
```

---

### Step 6: Handling Feature Extraction with External APIs and Custom Code
*Description:* Support external APIs or custom code for feature extraction. Allow users to register API endpoints, including open-source APIs (e.g., WHOIS, VirusTotal) or custom scripts developed by the model builder.

*API Registration Process:*
```python
import requests

# Register New API Endpoint
api_registry = {
    'whois_api': 'https://api.whois.com/v1',
    'custom_feature_api': 'http://localhost:5000/custom_feature'
}

# Endpoint Testing and Validation
def test_api(endpoint):
    try:
        response = requests.get(endpoint)
        if response.status_code == 200:
            print(f"API {endpoint} is active.")
        else:
            print(f"API {endpoint} failed.")
    except Exception as e:
        print(f"Error reaching API: {str(e)}")

```
---

* Schedule API calls using cron jobs or event-driven triggers.
* Implement automatic retries or switch to backup APIs upon failure.
* Notify stakeholders of API failures and retraining progress.
* Containerize custom API calls using Docker for scalability.
* Document logs of successful/failed API calls in BookStack for future analysis.
---
### Step 7: Apply Data Privacy and Security
*Description:* Ensure sensitive data is anonymized and encrypted to comply with data privacy standards.
```java
# Data Anonymization using ARX
java -jar arx-tool.jar -i raw_data.csv -o anonymized_data.csv --privacy k-anonymity 5

# Encrypt sensitive data
openssl enc -aes-256-cbc -salt -in sensitive_data.csv -out encrypted_data.csv -pass pass:securepassword

```
---

### Step 8: Documentation and Knowledge Sharing
*Description:* Maintain comprehensive documentation and facilitate knowledge sharing among team members.
* Use BookStack for creating and managing documentation.
* Employ Swagger for API documentation.
* Encourage collaboration through Trello and Slack.
---
