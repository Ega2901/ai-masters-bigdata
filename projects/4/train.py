import sys
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate() 
spark.sparkContext.setLogLevel('WARN')
from model import pipeline
from pyspark.sql.types import *
import pyspark.sql.functions as f
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, BooleanType
# для создания пользовательских функций
from pyspark.sql.functions import udf 
# для использования оконных функций
from pyspark.sql.window import Window
# для работы с PySpark DataFrame
from pyspark.sql import DataFrame
# для задания типа возвращаемого udf функцией
from pyspark.sql.types import StringType
# для создания регулярных выражений
import re


if __name__ == "__main__":
    conf = SparkConf()
    train_path = str(sys.argv[1])
    model_path = str(sys.argv[2])
    df = spark.read.json(train_path)
    df.cache()
    pipeline_model = pipeline.fit(df)
    pipeline_model.write().overwrite().save(model_path)
    spark.catalog.clearCache()
    spark.stop()
