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
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler

def filter_df(df):
    
    dataset = df.select('id', 'label', 'reviewText')
    dataset = dataset.na.fill({'label': 0, 'reviewText': 'unknown'})

    tokenizer = Tokenizer(inputCol = 'reviewText', outputCol = 'tokens')
    df2 = tokenizer.transform(dataset)
    
    hashingTF = HashingTF(numFeatures=100, binary=True, inputCol = 'tokens', outputCol = 'rawFeatures')
    df2 = hashingTF.transform(df2)
    
    df2 = df2.select('id', 'label','rawFeatures')
    return df2

if __name__ == "__main__":
    input_path = str(sys.argv[1])
    out_path = str(sys.argv[2])
    df = spark.read.json(input_path)
    df.cache()
    df2 = filter_df(df)
    df2.write_json(out_path)
    spark.catalog.clearCache()
    spark.stop()

