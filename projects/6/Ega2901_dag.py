from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.sensors.filesystem import FileSensor
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator

base_dir = '/opt/airflow/airflow_home/dags/example'
pyspark_python = "/opt/conda/envs/dsenv/bin/python"

with DAG(
        dag_id="Ega2901_dag",
        start_date=datetime(2023, 4, 26),
        schedule=None,
        catchup=False,
        description="Это наш первый DAG",
        doc_md = """
        Это учебный DAG. Не надо его переносить в продакшен!
        """,
        tags=["example"],
) as dag:
    bash_task = BashOperator(
        task_id='bash_task',
        bash_command=f'pwd; echo $USER; hdfs dfs -ls /datasets'
    )

    sensor_task = FileSensor(
        task_id=f'sensor_task',
        filepath=f"{base_dir}/some_file",
        poke_interval=30,
        timeout=60 * 5,
    )

    count_lines_task = BashOperator(
        task_id='count_lines_task',
        bash_command=f'hdfs dfs -cat /datasets/movielens/ratings.csv | wc -l',
    )

    sensor_task >> bash_task >> count_lines_task >> spark_task

