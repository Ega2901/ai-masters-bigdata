#!/bin/bash

# Параметры
project_number=$1
training_file_path=$2

# Запуск скрипта обучения
python projects/1a/train.py $project_number $training_file_path
