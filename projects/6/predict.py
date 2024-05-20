import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, pandas_udf
import sys
from joblib import load
from pyspark.sql import DataFrame
from pandas import DataFrame

spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel('WARN')

@pandas_udf("double")
def process_partition(id, unixReviewTime, verified, vote):
    print("******* ======== Start process_partition")
    df = DataFrame({'id': id, 'unixReviewTime': unixReviewTime, 'verified': verified, 'vote': vote})
    print(df.columns)
    df['prediction'] = model.predict(df).astype('double')
    print("******* ======== Model predicted  batch ===  ", df['prediction'].shape)

    return df['prediction']

print("******* ======== Start predict")
print("******* Python version  ", sys.version)
print("******* Spark version  ", pyspark.__version__)


test_data_path = sys.argv[1]
predictions_path = sys.argv[2]
model_path = sys.argv[3]

print ('*******  ', test_data_path, predictions_path, model_path)
model = load(model_path)
print("******* ======== Loaded model")

df = spark.read.json(test_data_path)
df = df.withColumn("verified", col("verified").cast("integer"))
num_rows = df.count()
num_cols = len(df.columns)
print("******* ======== Loaded TRAIN  ", (num_rows, num_cols) )

df_pred = df.withColumn("prediction", process_partition(df['id'], df['unixReviewTime'], df['verified'], df['vote'])).select("id", "prediction")
num_rows = df_pred.count()
num_cols = len(df_pred.columns)
print("******* ======== Model predicted", (num_rows, num_cols))

df_pred.write.mode('overwrite').csv(predictions_path, header=False)
print("******* ======== Save predict")

