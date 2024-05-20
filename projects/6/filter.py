from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
import sys

spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel('WARN')

path_in = sys.argv[1]
path_out = sys.argv[2]

df = spark.read.json(path_in)
drop_list=['reviewTime', 'reviewerName', 'reviewText', 'summary', 'reviewerID', 'asin']
df_local = df.drop(*drop_list)
df_local = df_local.withColumn('vote', col('vote').cast('float'))
df_local = df_local.withColumn('vote', when(col('vote').isNull(), 0).otherwise(col('vote')))
df_local.write.json(path_out, mode='overwrite')

#amazon_extrasmall_train.json train_local.json
#amazon_extrasmall_test.json test_local.json
