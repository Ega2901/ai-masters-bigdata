from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.sensors.filesystem import FileSensor

base_dir = '{{ dag_run.conf["base_dir"] if dag_run else "" }}'
pyspark_python = "/opt/conda/envs/dsenv/bin/python"

with DAG(
        dag_id="Ega2901_dag",
        start_date=datetime(2023, 4, 26),
        schedule_interval=None,
        catchup=False,
        description="HW6",
        doc_md="""
        HW6 Big Data !!!
        """,
        tags=["example"],
) as dag:
    feature_eng_task = BashOperator(
        task_id='feature_eng_task',
        bash_command=f'{pyspark_python} {base_dir}/filter.py /datasets/amazon/amazon_extrasmall_train.json Ega2901_train_out',       
    )
    feature_echo = BashOperator(
        task_id='feature_echo',
        bash_command=f'ls -laht {base_dir}/Ega2901_train_out_local',       
    )   

    download_train_task = BashOperator(
        task_id='download_train_task',
        bash_command=f'pwd; echo \$USER hdfs dfs -getmerge Ega2901_train_out {base_dir}/Ega2901_train_out_local',
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

    predict_task = BashOperator(
        task_id='predict_task',
        bash_command=f'{pyspark_python} {base_dir}/predict.py /datasets/amazon/amazon_extrasmall_test.json {base_dir}/Ega2901_hw6_prediction {base_dir}/6.joblib',
    )

    feature_eng_task >> download_train_task >> feature_echo >> train_task >> model_sensor >> predict_task

