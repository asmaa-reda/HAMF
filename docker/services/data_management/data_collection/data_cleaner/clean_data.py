import pandas as pd
from pyspark.sql import SparkSession

def clean_csv(file_path, output_path):
    df = pd.read_csv(file_path)
    # Example cleaning: Drop duplicates and fill missing values
    df = df.drop_duplicates().fillna("Unknown")
    df.to_csv(output_path, index=False)

def clean_with_spark(input_path, output_path):
    spark = SparkSession.builder.appName("DataCleaning").getOrCreate()
    df = spark.read.csv(input_path, header=True, inferSchema=True)
    df = df.dropDuplicates()
    df = df.na.fill("Unknown")
    df.write.csv(output_path, header=True, mode="overwrite")

if __name__ == "__main__":
    # Clean CSV data using Pandas
    clean_csv("./data/raw_csv.csv", "./data/cleaned_csv.csv")

    # Clean data using Spark
    clean_with_spark("./data/raw_api.csv", "./data/cleaned_api")
