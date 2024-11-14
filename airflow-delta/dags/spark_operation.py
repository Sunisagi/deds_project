from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator

# Default arguments for the DAG
default_args = {
    'owner': 'user',
    'depends_on_past': False,
    'start_date': datetime(2022, 1, 2),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

base_config = {
        "executor_memory": "16g",
        "driver_memory": "1g",
        "total_executor_cores": 4,
        "executor_cores": 4,
        "conf": {
            "spark.delta.logStore.class": "org.apache.spark.sql.delta.storage.HDFSLogStore", 
            "spark.jars.packages": "io.delta:delta-spark_2.12:3.2.0,io.delta:delta-storage:3.2.0",
            "spark.sql.catalog.spark_catalog": "org.apache.spark.sql.delta.catalog.DeltaCatalog",
            "spark.sql.extensions": "io.delta.sql.DeltaSparkSessionExtension",
        },
}

dag = DAG(
    'test_spark_connection',
    default_args=default_args,
    schedule_interval=timedelta(days=1),
    catchup=False,
)

create_mock_table = SparkSubmitOperator(
    **base_config,
    task_id='create_mock_table',
    application='./dags/spark_job/create_delta_table.py',  # Path to your Spark job script
    name='create_mock_table',
    conn_id='spark_default',  # Use the connection ID you created
    verbose=True,
    application_args=['{{ macros.ds_add("2023-06-01", -1) }}', '{{ "2023-06-01" }}'],
    dag=dag,
)

create_mock_table