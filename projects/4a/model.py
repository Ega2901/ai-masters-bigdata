from pyspark.sql.types import *
from pyspark.sql.functions import col, lit, when
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.ml import Pipeline, Transformer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
import pyspark.sql.functions as F

class CustomImputer(Transformer, DefaultParamsReadable, DefaultParamsWritable):
    def __init__(self, inputCols=None, outputCols=None, missingValue="missing"):
        super(CustomImputer, self).__init__()
        self.inputCols = inputCols
        self.outputCols = outputCols
        self.missingValue = missingValue

    def _transform(self, df: DataFrame) -> DataFrame:
        for inputCol, outputCol in zip(self.inputCols, self.outputCols):
            df = df.withColumn(outputCol, when(col(inputCol).isNull(), lit(self.missingValue)).otherwise(col(inputCol)))
        return df


customImputer = CustomImputer(inputCols='reviewText', outputCols='reviewText_clear', missingValue="missing")
tokenizer = Tokenizer(inputCol = 'reviewText_clear', outputCol = 'tokens')
hashingTF = HashingTF(numFeatures=100, binary=True, inputCol = 'tokens', outputCol = 'rawFeatures')
lr = LogisticRegression(featuresCol='rawFeatures',
                        labelCol='overall')
pipeline = Pipeline(stages = [customImputer, tokenizer, hashingTF, lr])
