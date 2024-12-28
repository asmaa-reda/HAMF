# Integration Manual

## Overview
This manual provides instructions to integrate the HAMF Framework with external systems, including APIs, databases, and third-party tools.

---

## API Integration
### Data Collection
- **Endpoint:** `/collect`
- **Method:** POST
- **Payload:**
  ```json
  {
    "source": "https://example.com",
    "frequency": "hourly"
  }
  ```
### Model Predictions
- **Endpoint:** `/predict`
- **Method:** POST
- **Payload:**
  ```json
  {
    "url": "http://malicious-site.com"
  }
  ```


## External Tool Integration
*1. Slack*
- Set up a Slack App and generate an API token.
- Update the slack_notification.py configuration:
```python
  {
    "source": "https://example.com",
    "frequency": "hourly"
  }
  ```

*2. SMTP*
- Configure the SMTP settings in smtp_service.py
```python
  SMTP_SERVER = "smtp.example.com"
  SMTP_PORT = 587
  USERNAME = "user@example.com"
  PASSWORD = "password"
```



