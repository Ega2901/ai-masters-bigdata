from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.sensors.filesystem import FileSensor
from pendulum import today

dag = DAG(
    'Ega2901_dag',
    start_date=today('UTC').subtract(days=2),
    description='A simple tutorial DAG',
    schedule=None,
    catchup=False,
)

pyton_path='/opt/conda/envs/dsenv/bin/python'
base_dir = '{{ dag_run.conf["base_dir"] if dag_run else "" }}'

feature_eng_train_task = SparkSubmitOperator(
    task_id='feature_eng_train_task',
    application=f"{base_dir}/filter.py",
    application_args=["/datasets/amazon/amazon_extrasmall_train.json", "Ega2901_train_out"],
    spark_binary='/usr/bin/spark3-submit',
    num_executors=7,
    executor_cores=1,
    executor_memory="2G",
    env_vars={'PYSPARK_PYTHON': pyton_path},
    dag=dag
)

download_train_task = BashOperator(
    task_id='download_train_task',
    bash_command=f'hadoop fs -get Ega2901_train_out {base_dir}/Ega2901_train_out_local',
    dag=dag
)

train_task = BashOperator(
    task_id='train_task',
    bash_command=f'{pyton_path} {base_dir}/model.py {base_dir}/Ega2901_train_out_local {base_dir}/6.joblib',
    dag=dag
)

model_sensor = FileSensor(
    task_id='model_sensor',
    filepath=f'{base_dir}/6.joblib',
    poke_interval=20,
    timeout=600,
    dag=dag
)

feature_eng_test_task = SparkSubmitOperator(
    task_id='feature_eng_test_task',
    application=f"{base_dir}/filter.py",
    application_args=["/datasets/amazon/amazon_extrasmall_test.json", "Ega2901_test_out"],
    spark_binary='/usr/bin/spark3-submit',
    num_executors=7,
    executor_cores=1,
    executor_memory="2G",
    env_vars={'PYSPARK_PYTHON': pyton_path},
    dag=dag
)

predict_task = SparkSubmitOperator(
    task_id='predict_task',
    application=f"{base_dir}/predict.py",
    application_args=["Ega2901_test_out", "Ega2901_hw6_prediction", f"{base_dir}/6.joblib"],
    spark_binary='/usr/bin/spark3-submit',
    num_executors=7,
    executor_cores=1,
    executor_memory="2G",
    env_vars={'PYSPARK_PYTHON': pyton_path},
    dag=dag
)
feature_eng_train_task >> download_train_task >> train_task >> model_sensor >> feature_eng_test_task >> predict_task

