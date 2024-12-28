# **HAMF Framework Installation Guide**

## **Introduction**
This guide provides detailed steps to install and deploy the Hybrid Approach Maintenance Framework (HAMF) from scratch, leveraging Docker containerization and the centralized configuration file `config/settings.py`.

---

## **Prerequisites**
Before starting the installation, ensure the following are installed on your system:

1. **Docker**: [Install Docker](https://docs.docker.com/get-docker/)
2. **Docker Compose**: [Install Docker Compose](https://docs.docker.com/compose/install/)
3. **Python 3.9 or Later**: [Install Python](https://www.python.org/downloads/)
4. **Git**: [Install Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)

---

## **Repository Setup**

1. **Clone the Repository**
   ```bash
   git clone https://github.com/redaghareeb/HAMF.git
   cd HAMF
   ```

2. **Directory Overview**
   ```
   HAMF/
   ├── ci/
   ├── config/
   │   ├── settings.py
   ├── docker/
   │   ├── docker-compose.yml
   │   ├── services/
   │   │   ├── data_management/
   │   │   │   ├── data_collection/
   │   │   │   │   ├── custom_collector/
   │   │   │   │   ├── selenium_scapper/
   │   │   │   │   ├── data_cleaner/
   │   │   │   ├── data_preprocessing/
   │   │   │   │   ├── spark/
   │   │   │   │   ├── preprocess_data.py
   │   │   │   ├── database/
   │   │   │   │   ├── db_operations.py
   │   │   │   │   ├── init.sql
   │   │   │   ├── logstash/
   │   │   │   │   ├── logstash.conf
   │   │   ├── model_(re)training/
   │   │   ├── model_performance_monitoring/
   │   │   ├── communication_and_notification/
   │   │   │   ├── slack/
   │   │   │   ├── smtp/
   │   │   ├── governance_and_privacy/
   │   ├── shared/
   │   │   ├── config/
   │   │   ├── volumes/
   │   │   │   ├── minio/
   │   │   │   │   ├── raw/
   │   │   │   │   ├── processed/
   │   │   │   │   ├── models/  <-- Pickle stored here
   │   │   │   │   ├── amanda_backups/
   ├── docs/
   ├── examples/
   ├── tests/
   └── LICENSE
   ```

3. **Centralized Configuration (`config/settings.py`)**
   Open `config/settings.py` and adjust the following configurations:

   ```python
   # Core Settings for HAMF Framework

   # SMTP Settings
   SMTP_CONFIG = {
       "server": "smtp.example.com",
       "port": 587,
       "username": "your-email@example.com",
       "password": "your-password",
       "use_tls": True
   }

   # Slack Settings
   SLACK_CONFIG = {
       "api_token": "your-slack-api-token",
       "default_channel": "#notifications"
   }

   # PostgreSQL Database Settings
   DATABASE_CONFIG = {
       "host": "postgres",
       "port": 5432,
       "database": "hamf_db",
       "username": "postgres",
       "password": "postgres"
   }

   # Data Sources
   DATA_SOURCES = {
       "raw_data_bucket": "raw-data",
       "processed_data_bucket": "processed-data"
   }

   # Model Settings
   MODEL_SETTINGS = {
       "default_model": "RandomForestClassifier",
       "tracking_uri": "http://mlflow:5000"
   }

   # Logging
   LOGGING_CONFIG = {
       "level": "INFO",
       "log_file": "/app/logs/hamf.log"
   }
   ```

---

## **Docker Setup**

1. **Build and Start Containers**
   Navigate to the `docker/` directory and run:
   ```bash
   docker-compose up --build -d
   ```

2. **Verify Containers**
   Check the status of all containers:
   ```bash
   docker ps
   ```

3. **Access Services**
   After starting the containers, you can access the following services:

   ### **Infrastructure Services**
   - **PostgreSQL** (No Web Interface)
     - Host: `localhost:5432`
     - Username: `postgres`
     - Password: `postgres`
     - Database: `hamf_db`

   - **MinIO Object Storage**
     - URL: [http://localhost:9000](http://localhost:9000)
     - Console: [http://localhost:9001](http://localhost:9001)
     - Username: `minioadmin`
     - Password: `minioadmin`

   ### **Machine Learning & Storage**
   - **MLFlow**
     - URL: [http://localhost:5000](http://localhost:5000)
     - No authentication required by default.

   - **Apache Spark**
     - URL: [http://localhost:8080](http://localhost:8080)
     - Username/Password: Not Applicable

   ### **Monitoring & Visualization**
   - **Grafana**
     - URL: [http://localhost:3000](http://localhost:3000)
     - Username: `admin`
     - Password: `admin` (Default credentials; change after first login.)

   - **Prometheus**
     - URL: [http://localhost:9090](http://localhost:9090)
     - Username/Password: Not Applicable

   - **Elasticsearch**
     - URL: [http://localhost:9200](http://localhost:9200)
     - Username: `elastic`
     - Password: `changeme` (Default credentials; change in production.)

   ### **Documentation Services**
   - **BookStack**
     - URL: [http://localhost:6875](http://localhost:6875)
     - Username: `admin@bookstack`
     - Password: `password` (Default credentials; change after first login.)

   ### **Custom and Application Logic**
   - **Selenium (Scraping Tool)**
     - URL: [http://localhost:4444](http://localhost:4444)
     - Username/Password: Not Applicable

   - **Slack**
     - No direct URL interface. Uses the API token defined in `SLACK_CONFIG`.

   - **SMTP**
     - No direct URL interface. Uses the SMTP settings defined in `SMTP_CONFIG`.

   ### **Backup Services**
   - **Amanda Backup**
     - URL: [http://localhost:10080](http://localhost:10080)
     - Username/Password: Not Applicable

   ---

## **Customizing Docker Compose**

### **Update Configuration File**
If any settings in `config/settings.py` change, restart the containers to apply updates:
```bash
docker-compose down
docker-compose up --build -d
```

### **Adding New Services**
To add a custom service, update `docker-compose.yml` with the following format:
```yaml
new-service:
  build:
    context: ./services/new-service
  container_name: new-service
  volumes:
    - ./shared/volumes/new-service:/data
  networks:
    - hamf-net
```

---

## **Running Custom Scripts**

### **Data Collection**
Run the data collection service manually:
```bash
docker exec -it data-collection python app.py
```

### **Model Training**
Trigger model training with:
```bash
docker exec -it model-training python train.py
```

### **Governance and Privacy**
Validate data governance compliance:
```bash
docker exec -it governance_and_privacy python privacy_manager.py
```

---

## **Logs and Monitoring**

1. **Access Logs**
   Logs for each service are available in the `/app/logs/` directory inside their respective containers. Access logs via Docker:
   ```bash
   docker logs <container-name>
   ```

2. **Real-Time Monitoring**
   - Open **Grafana** at `http://localhost:3000`.
   - Configure Prometheus as a data source to visualize metrics.

---

## **Testing and Validation**

1. **Run Tests**
   Execute unit and integration tests:
   ```bash
   pytest tests/
   ```

2. **Integration Validation**
   Validate service integrations using the provided scripts in `ci/scripts/`.

---

## **Troubleshooting**

1. **Container Issues**
   Restart a container if it stops unexpectedly:
   ```bash
   docker restart <container-name>
   ```

2. **Configuration Errors**
   Ensure all configurations in `config/settings.py` are correct and match the expected environment.

3. **Network Conflicts**
   Check if the default Docker network (`hamf-net`) is already in use and change it in `docker-compose.yml` if necessary.

---

## **Advanced Configuration**

### **Scaling Services**
For scalable deployments, increase replicas for specific services:
```yaml
services:
  model-training:
    deploy:
      replicas: 3
```

### **Secure Deployment**
- Use a secrets manager (e.g., AWS Secrets Manager or HashiCorp Vault) for sensitive configurations like API keys and passwords.

---

## **Contributing**

Contributions are welcome! Fork the repository, make your changes, and submit a pull request. Follow the coding guidelines provided in the `DEVELOPER_GUIDE.md`.

---
