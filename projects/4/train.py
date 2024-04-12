import sys
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate() 
spark.sparkContext.setLogLevel('WARN')
from model import pipeline
from pyspark.sql.types import *
import pyspark.sql.functions as f



if __name__ == "__main__":
    conf = SparkConf()
    train_path = str(sys.argv[1])
    model_path = str(sys.argv[2])
    spark.catalog.clearCache()
    df = spark.read.json(train_path)
    df.cache()
    pipeline_model = pipeline.fit(df)
    pipeline_model.write().overwrite().save(model_path)
    spark.catalog.clearCache()
    spark.stop()
