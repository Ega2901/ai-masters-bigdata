#!/bin/bash

# 1st arg - files to send with the job
# 2nd arg - input path
# 3rd arg - output path
# 4th arg - mapper file
cd ai-masters-bigdata
#copy input dataset to HDFS
hdfs dfs -copyFromLocal /datasets/criteo/train-with-id.txt train-with-id.txt
#remove output dataset if exists
hdfs dfs -rm -r -f predicted.csv
projects/tut1/predict.sh projects/tut1/predict.py,projects/tut1/model.py,tut1.joblib hotels.csv predicted.csv predict.py
