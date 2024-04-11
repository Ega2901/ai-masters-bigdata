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

def text_prep(text):
    text = str(text).lower()
    text = re.sub('\s+',' ',text)
    text = text.strip()
    return text

if __name__ == "__main__":
    conf = SparkConf()
    train_path = str(sys.argv[1])
    model_path = str(sys.argv[2])
    df = spark.read.json(train_path)
    df.cache()
    df2 = df.select("overall", "reviewText")
    prep_text_udf = udf(text_prep, StringType())
    t = df2.withColumn('prep_text', prep_text_udf("reviewText"))\
        .filter('prep_text <> ""')
    pipeline_model = pipeline.fit(t)
    pipeline_model.write().overwrite().save(model_path)
    spark.catalog.clearCache()
    spark.stop()
