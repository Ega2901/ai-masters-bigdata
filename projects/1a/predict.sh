#!/bin/bash
cd ai-masters-bigdata
#remove output dataset if exists
hdfs dfs -rm -r -f -skipTrash predicted.csv
projects/1a/predict.sh projects/1a/predict.py,1a.joblib /datasets/criteo/train-with-id.txt predicted.csv predict.py
