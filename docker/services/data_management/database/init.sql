-- ==========================================
-- Existing Tables from HAMF (init.sql)
-- ==========================================

-- Users Table (Existing)
CREATE TABLE IF NOT EXISTS users (
    user_id SERIAL PRIMARY KEY,
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Roles Table (Existing)
CREATE TABLE IF NOT EXISTS roles (
    role_id SERIAL PRIMARY KEY,
    role_name VARCHAR(100) UNIQUE NOT NULL
);

-- User Roles Mapping (Existing)
CREATE TABLE IF NOT EXISTS user_roles (
    user_id INT REFERENCES users(user_id),
    role_id INT REFERENCES roles(role_id),
    PRIMARY KEY (user_id, role_id)
);

-- ==========================================
-- Enhanced Tables (Model Management)
-- ==========================================

-- Feature Master Table (Enhanced)
CREATE TABLE IF NOT EXISTS features_master (
    feature_id SERIAL PRIMARY KEY,
    feature_name VARCHAR(255) NOT NULL UNIQUE,
    feature_type VARCHAR(50) NOT NULL,  -- e.g., numerical, categorical
    feature_status VARCHAR(50) DEFAULT 'active',
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Models Master Table (Enhanced)
CREATE TABLE IF NOT EXISTS models_master (
    model_id SERIAL PRIMARY KEY,
    model_name VARCHAR(255) NOT NULL UNIQUE,
    algorithm VARCHAR(100) NOT NULL,  -- e.g., GradientBoosting, RandomForest
    model_version VARCHAR(50) NOT NULL,
    creation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    description TEXT
);

-- Feature-Model Mapping (Enhanced)
CREATE TABLE IF NOT EXISTS features_models_map (
    map_id SERIAL PRIMARY KEY,
    model_id INT REFERENCES models_master(model_id) ON DELETE CASCADE,
    feature_id INT REFERENCES features_master(feature_id) ON DELETE CASCADE,
    accuracy DECIMAL(5, 2) DEFAULT 0,
    last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Model Training Results (Enhanced)
CREATE TABLE IF NOT EXISTS training_results (
    result_id SERIAL PRIMARY KEY,
    model_id INT REFERENCES models_master(model_id) ON DELETE CASCADE,
    accuracy DECIMAL(5, 2) NOT NULL,
    f1_score DECIMAL(5, 2) NOT NULL,
    precision DECIMAL(5, 2) NOT NULL,
    recall DECIMAL(5, 2) NOT NULL,
    training_status VARCHAR(50) DEFAULT 'completed',
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ==========================================
-- Data Management Tables (Existing)
-- ==========================================

-- Data Inventory (Merged)
CREATE TABLE IF NOT EXISTS data_inventory (
    data_id SERIAL PRIMARY KEY,
    data_name VARCHAR(255) NOT NULL,
    data_category VARCHAR(100) NOT NULL,  -- e.g., raw, processed, features
    access_roles VARCHAR(255),
    description TEXT,
    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Raw Data Table (Existing)
CREATE TABLE IF NOT EXISTS raw_data (
    data_id SERIAL PRIMARY KEY,
    source VARCHAR(255),
    collected_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    data JSONB
);

-- Processed Data Table (Existing)
CREATE TABLE IF NOT EXISTS processed_data (
    data_id SERIAL PRIMARY KEY,
    processing_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    feature_data JSONB
);

-- ==========================================
-- Monitoring and Audit Tables (Enhanced)
-- ==========================================

-- Audit Log Table (Enhanced)
CREATE TABLE IF NOT EXISTS audit_log (
    log_id SERIAL PRIMARY KEY,
    table_name VARCHAR(255) NOT NULL,
    operation_type VARCHAR(50) NOT NULL,  -- e.g., INSERT, UPDATE, DELETE
    old_data JSONB,
    new_data JSONB,
    operation_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Prometheus Metrics (New - Optional)
CREATE TABLE IF NOT EXISTS prometheus_metrics (
    metric_id SERIAL PRIMARY KEY,
    model_id INT REFERENCES models_master(model_id),
    accuracy DECIMAL(5, 2),
    f1_score DECIMAL(5, 2),
    collection_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- OPTIONAL 
-- Index for feature lookups
CREATE INDEX idx_feature_name ON features_master (feature_name);

-- Index for model performance lookups
CREATE INDEX idx_accuracy ON training_results (accuracy);

-- Triggers for Automatic Auditing
CREATE OR REPLACE FUNCTION log_audit()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO audit_log (table_name, operation_type, old_data, new_data)
    VALUES (
        TG_TABLE_NAME,
        TG_OP,
        row_to_json(OLD),
        row_to_json(NEW)
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Attach to tables for automatic auditing
CREATE TRIGGER audit_features
AFTER INSERT OR UPDATE OR DELETE
ON features_master
FOR EACH ROW
EXECUTE FUNCTION log_audit();

--Partitioning for Large Datasets (Optional)
-- Partitioning for training results by year
CREATE TABLE training_results_2024
PARTITION OF training_results
FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');
