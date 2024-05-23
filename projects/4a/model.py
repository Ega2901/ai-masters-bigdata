from pyspark.sql.types import *
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, lit, when
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.ml import Pipeline, Transformer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.param.shared import Param, Params, TypeConverters
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable


class CustomImputer(Transformer, DefaultParamsReadable, DefaultParamsWritable):
    
    inputCol = Param(Params._dummy(), "inputCol", "The input column", typeConverter=TypeConverters.toString)
    outputCol = Param(Params._dummy(), "outputCol", "The output column", typeConverter=TypeConverters.toString)
    fillValue = Param(Params._dummy(), "fillValue", "The value to replace missing values with", typeConverter=TypeConverters.toString)

    def __init__(self, inputCol=None, outputCol=None, fillValue=None):
        super(CustomImputer, self).__init__()
        self._setDefault(inputCol=None, outputCol=None, fillValue=None)
        if inputCol is not None:
            self.setInputCol(inputCol)
        if outputCol is not None:
            self.setOutputCol(outputCol)
        if fillValue is not None:
            self.setFillValue(fillValue)

    def setInputCol(self, value):
        self._set(inputCol=value)
        return self

    def setOutputCol(self, value):
        self._set(outputCol=value)
        return self

    def setFillValue(self, value):
        self._set(fillValue=value)
        return self

    def getInputCol(self):
        return self.getOrDefault(self.inputCol)

    def getOutputCol(self):
        return self.getOrDefault(self.outputCol)

    def getFillValue(self):
        return self.getOrDefault(self.fillValue)

    def _transform(self, dataset: DataFrame) -> DataFrame:
        input_col = self.getInputCol()
        output_col = self.getOutputCol()
        fill_value = self.getFillValue()
        return dataset.withColumn(output_col, col(input_col).na.fill(fill_value))



customImputer = CustomImputer(inputCol='reviewText', outputCol='reviewText_clear', fillValue="missing")
tokenizer = Tokenizer(inputCol = 'reviewText_clear', outputCol = 'tokens')
hashingTF = HashingTF(numFeatures=100, binary=True, inputCol = 'tokens', outputCol = 'rawFeatures')
lr = LogisticRegression(featuresCol='rawFeatures',
                        labelCol='overall')
pipeline = Pipeline(stages = [customImputer, tokenizer, hashingTF, lr])
