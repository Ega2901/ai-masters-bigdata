#!/bin/bash

# Параметры
project_files=$1
input_file_path=$2
output_file_path=$3
predict_script=$4

# Запуск скрипта предсказаний
/opt/conda/envs/dsenv/bin/python $predict_script $project_files $input_file_path > $output_file_path
