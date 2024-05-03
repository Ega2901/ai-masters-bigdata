from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.sensors.filesystem import FileSensor
from airflow.contrib.operators.spark_submit_operator import SparkSubmitOperator

base_dir = '{{ dag_run.conf["base_dir"] if dag_run else "" }}'
pyspark_python = "/opt/conda/envs/dsenv/bin/python"

with DAG(
        dag_id="Ega2901_dag",
        start_date=datetime(2023, 4, 26),
        schedule=None,
        catchup=False,
        description="HW6",
        doc_md = """
        HW6 Big Data !!!
        """,
        tags=["example"],
) as dag:
    feature_eng_task = SparkSubmitOperator(
        task_id='feature_eng_task',
        application=f'{base_dir}/filter.py',
        spark_binary='/usr/bin/spark3-submit',
        application_args=[f'{base_dir}/datasets/amazon/amazon_extrasmall_train.json', f'{base_dir}/Ega2901_train_out'],
        conf={'PYSPARK_PYTHON':pyspark_python}
     ) 

    download_train_task = BashOperator(
        task_id='download_train_task',
        bash_command=f'hdfs dfs -get {base_dir}/Ega2901_train_out {base_dir}/Ega2901_train_out_local',
    )

    train_task = BashOperator(
        task_id='train_task',
        bash_command=f'{pyspark_python} {base_dir}/model.py {base_dir}/Ega2901_train_out_local {base_dir}/6.joblib',
    )

    model_sensor = FileSensor(
        task_id='model_sensor',
        filepath=f'{base_dir}/6.joblib',
        poke_interval=60,
        timeout=600,
    )
    feature_eng_test_task = SparkSubmitOperator(
        task_id='feature_eng_test_task',
        application=f'{base_dir}/filter.py',
        spark_binary='/usr/bin/spark3-submit',
        application_args=[f'{base_dir}/datasets/amazon/amazon_extrasmall_test.json', f'{base_dir}/Ega2901_test_out'],
        conf={'PYSPARK_PYTHON':pyspark_python}
     )

    download_test_task = BashOperator(
        task_id='download_test_task',
        bash_command=f'hdfs dfs -get {base_dir}/Ega2901_test_out {base_dir}/Ega2901_test_out_local',
    )


    predict_task = BashOperator(
        task_id='predict_task',
        bash_command= f'{pyspark_python} {base_dir}/Ega2901_test_out_local {base_dir}/Ega2901_hw6_prediction {base_dir}/6.joblib',
    )

    feature_eng_task >> download_train_task >> train_task >> model_sensor >> feature_eng_test_task >> download_test_task >> predict_task
