#!/opt/conda/envs/dsenv/bin/python
import os
import sys
SPARK_HOME = "/usr/lib/spark3"
PYSPARK_PYTHON = "/opt/conda/envs/dsenv/bin/python"
os.environ["PYSPARK_PYTHON"]= PYSPARK_PYTHON
os.environ["PYSPARK_DRIVER_PYTHON"]= PYSPARK_PYTHON
os.environ["SPARK_HOME"] = SPARK_HOME
PYSPARK_HOME = os.path.join(SPARK_HOME, "python/lib")
sys.path.insert(0, os.path.join(PYSPARK_HOME, "py4j-0.10.9.5-src.zip"))
sys.path.insert(0, os.path.join(PYSPARK_HOME, "pyspark.zip"))
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
conf = SparkConf()
conf.set("spark.ui.port", "4099")
spark = SparkSession.builder.config(conf=conf).appName("Spark SQL").getOrCreate()                
from pyspark.sql.types import *
import pyspark.sql.functions as f
from sklearn.preprocessing import StandardScaler
from pyspark.ml.feature import VectorAssembler
import numpy as np
import pandas as pd 
from sklearn.linear_model import LogisticRegression
from joblib import dump, load


if __name__ == "__main__":
    input_path = str(sys.argv[1])
    out_path = str(sys.argv[2])
    df = spark.read.json(input_path)
    
    assembler = VectorAssembler(inputCols=["rawFeatures"], outputCol="features")
    df = assembler.transform(df)
    
    scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=True)
    scaler_model = scaler.fit(df)
    df = scaler_model.transform(df)
    
    X = df.select("scaledFeatures").rdd.map(lambda x: x[0]).collect()
    y = df.select("label").rdd.map(lambda x: x[0]).collect()
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    dump(model, out_path)
    spark.stop()

