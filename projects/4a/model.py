from pyspark.sql.types import *
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, lit, when
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.ml import Pipeline, Transformer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
import pyspark.sql.functions as F

class CustomImputer(Transformer, DefaultParamsReadable, DefaultParamsWritable):
    def __init__(self, inputCol=None, outputCol=None, missingValue="missing"):
        super(CustomImputer, self).__init__()
        self.inputCol = inputCol
        self.outputCol = outputCol
        self.missingValue = missingValue

    def _transform(self, df: DataFrame) -> DataFrame:
        if self.inputCol in df.columns:
            df = df.withColumn(self.outputCol, when(col(self.inputCol).isNull(), lit(self.missingValue)).otherwise(col(self.inputCol)))
        else:
            raise ValueError(f"Column '{self.inputCol}' does not exist in DataFrame.")
        return df


customImputer = CustomImputer(inputCol='reviewText', outputCol='reviewText_clear', missingValue="missing")
tokenizer = Tokenizer(inputCol = 'reviewText_clear', outputCol = 'tokens')
hashingTF = HashingTF(numFeatures=100, binary=True, inputCol = 'tokens', outputCol = 'rawFeatures')
lr = LogisticRegression(featuresCol='rawFeatures',
                        labelCol='overall')
pipeline = Pipeline(stages = [customImputer, tokenizer, hashingTF, lr])
