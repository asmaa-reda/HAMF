import psycopg2
from psycopg2.extras import execute_values
import importlib.util
from datetime import datetime

# Load settings dynamically from config/settings.py
def load_settings():
    spec = importlib.util.spec_from_file_location("settings", "./config/settings.py")
    settings = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(settings)
    return settings

settings = load_settings()
DATABASE = settings.DATABASE


# Establish database connection dynamically
def connect_db():
    try:
        conn = psycopg2.connect(
            host=DATABASE['host'],
            port=DATABASE['port'],
            user=DATABASE['user'],
            password=DATABASE['password'],
            dbname=DATABASE['dbname']
        )
        return conn
    except Exception as e:
        print(f"[ERROR] Database connection failed: {e}")
        return None


# Insert new feature into features_master table
def insert_feature(feature_name, feature_type, feature_status="active"):
    query = """
    INSERT INTO features_master (feature_name, feature_type, feature_status, last_updated)
    VALUES (%s, %s, %s, %s) RETURNING feature_id;
    """
    values = (feature_name, feature_type, feature_status, datetime.now())
    with connect_db() as conn:
        with conn.cursor() as cur:
            cur.execute(query, values)
            feature_id = cur.fetchone()[0]
            conn.commit()
            print(f"[INFO] Feature '{feature_name}' inserted with ID: {feature_id}")


# Insert model into models_master table
def insert_model(model_name, algorithm, model_version):
    query = """
    INSERT INTO models_master (model_name, algorithm, model_version, creation_date)
    VALUES (%s, %s, %s, %s) RETURNING model_id;
    """
    values = (model_name, algorithm, model_version, datetime.now())
    with connect_db() as conn:
        with conn.cursor() as cur:
            cur.execute(query, values)
            model_id = cur.fetchone()[0]
            conn.commit()
            print(f"[INFO] Model '{model_name}' inserted with ID: {model_id}")


# Insert training result into training_results table
def insert_training_result(model_id, accuracy, f1_score, precision, recall, training_status="completed"):
    query = """
    INSERT INTO training_results (model_id, accuracy, f1_score, precision, recall, training_status, timestamp)
    VALUES (%s, %s, %s, %s, %s, %s, %s);
    """
    values = (model_id, accuracy, f1_score, precision, recall, training_status, datetime.now())
    with connect_db() as conn:
        with conn.cursor() as cur:
            cur.execute(query, values)
            conn.commit()
            print(f"[INFO] Training result logged for model ID {model_id}.")


# Audit Logging for any table changes
def log_audit(table_name, operation_type, old_data=None, new_data=None):
    query = """
    INSERT INTO audit_log (table_name, operation_type, old_data, new_data)
    VALUES (%s, %s, %s, %s);
    """
    values = (table_name, operation_type, old_data, new_data)
    with connect_db() as conn:
        with conn.cursor() as cur:
            cur.execute(query, values)
            conn.commit()
            print(f"[INFO] Audit log created for {operation_type} on {table_name}.")


# Bulk insert data (e.g., for raw_data)
def bulk_insert_data(table_name, data_list):
    query = f"""
    INSERT INTO {table_name} (source, collected_date, data)
    VALUES %s;
    """
    with connect_db() as conn:
        with conn.cursor() as cur:
            execute_values(cur, query, data_list)
            conn.commit()
            print(f"[INFO] Bulk data inserted into {table_name}.")


# Fetch latest training result for monitoring
def get_latest_training_result(model_id):
    query = """
    SELECT accuracy, f1_score FROM training_results
    WHERE model_id = %s
    ORDER BY timestamp DESC LIMIT 1;
    """
    with connect_db() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (model_id,))
            result = cur.fetchone()
            if result:
                return {"accuracy": result[0], "f1_score": result[1]}
            return None
