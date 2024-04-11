import sys
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline, PipelineModel
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
from pyspark.sql.functions import monotonically_increasing_id

def text_prep(text):
    text = str(text).lower()
    text = re.sub('\s+',' ',text)
    text = text.strip()
    return text

if __name__ == "__main__":
    conf = SparkConf()
    spark = SparkSession.builder.getOrCreate() 
    spark.sparkContext.setLogLevel('WARN')
    model_path = str(sys.argv[1])
    test_path = str(sys.argv[2])
    pred_path = str(sys.argv[3])
    model = PipelineModel.load(model_path)
    df = spark.read.json(test_path)
    df.cache()
    df2 = df.select("reviewText")
    prep_text_udf = udf(text_prep, StringType())
    t = df2.withColumn('prep_text', prep_text_udf("reviewText"))\
        .filter('prep_text <> ""')
    predictions = model.transform(t)
    predictions = predictions.withColumn("id", monotonically_increasing_id())
    predictions_to_save = predictions.select("id", "prediction")
    predictions_to_save.coalesce(1).write.csv(pred_path, header=False)
    spark.catalog.clearCache()
    spark.stop()