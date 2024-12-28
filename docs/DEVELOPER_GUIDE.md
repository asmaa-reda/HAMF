# Developer Guide

## Overview
This guide provides technical details about the HAMF Framework, its components, and instructions for extending or customizing the framework.

---

## Components
### 1. **Data Collection**
- **Service Folder:** `docker/services/data-collection`
- **Technologies:** Selenium, MinIO
- **Purpose:** Collect phishing URLs and store them in MinIO.

### 2. **Data Preprocessing**
- **Service Folder:** `docker/services/preprocessing`
- **Technologies:** Apache Spark, K-Anonymity
- **Purpose:** Clean, validate, and anonymize data.

### 3. **Model Training**
- **Service Folder:** `docker/services/model-training`
- **Technologies:** MLFlow, AI Fairness 360
- **Purpose:** Train machine learning models and track experiments.

### 4. **Monitoring**
- **Service Folder:** `docker/services/monitoring`
- **Technologies:** Prometheus, Grafana, Elasticsearch
- **Purpose:** Monitor system and model performance.

### 5. **Governance and Privacy**
- **Service Folder:** `docker/services/governance_and_privacy`
- **Technologies:** PyCryptodome, L-Diversity
- **Purpose:** Ensure compliance with data privacy standards.

---

## Extending the Framework
1. **Add a New Service:**
   - Create a folder under `docker/services/`.
   - Add a `Dockerfile` and implementation files.

2. **Modify Existing Services:**
   - Edit the relevant `app.py` or configuration files.

3. **Update CI/CD Pipeline:**
   - Add new scripts or jobs in `.gitlab-ci.yml`.

---

## Integration
Refer to the [Integration Manual](INTEGRATION_MANUAL.md) for connecting HAMF with external systems.
