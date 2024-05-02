import os
import sys
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate() 
spark.sparkContext.setLogLevel('WARN')
from pyspark.sql.types import *
import pyspark.sql.functions as f
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler

def filter_df(df):
    
    dataset = df.select('id', 'label', 'reviewText')
    dataset = dataset.na.fill({'overall': 0, 'reviewText': 'unknown'})

    tokenizer = Tokenizer(inputCol = 'reviewText', outputCol = 'tokens')
    df2 = tokenizer.transform(dataset)
    
    hashingTF = HashingTF(numFeatures=100, binary=True, inputCol = 'tokens', outputCol = 'rawFeatures')
    df2 = hashingTF.transform(df2)
    
    df2 = df2.select('id', 'label','rawFeatures')
    return df2

if __name__ == "__main__":
    conf = SparkConf()
    input_path = str(sys.argv[1])
    out_path = str(sys.argv[2])
    df = spark.read.json(input_path)
    df.cache()
    df2 = filter_df(df)
    df2.write.csv(out_path)
    spark.catalog.clearCache()
    spark.stop()

