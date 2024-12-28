# HAMF Framework

## Overview
The Hybrid Approach Maintenance Framework (HAMF) is a scalable, modular, and secure system designed for phishing detection, governance, and monitoring. It integrates state-of-the-art open-source tools to provide a robust data pipeline for collection, preprocessing, training, and real-time monitoring.

---

## Features
- **Real-time Data Collection:** Using Selenium and MinIO.
- **Scalable Data Preprocessing:** Apache Spark for distributed processing.
- **Machine Learning Model Training:** MLFlow for experiment tracking.
- **Governance and Privacy:** Tools for compliance with K-Anonymity, L-Diversity, and GDPR-like policies.
- **Monitoring and Alerts:** Prometheus and Grafana dashboards with Slack/SMTP notifications.

---

## Quick Start
### Prerequisites
1. Install Docker and Docker Compose.
2. Clone the repository:
   ```bash
   git clone https://github.com/asmaareda/HAMF.git
   cd HAMF-Framework
