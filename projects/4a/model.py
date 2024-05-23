from pyspark.sql.types import *
from pyspark import keyword_only
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, lit, when
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.ml import Pipeline, Transformer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.param import Param, Params
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable

class CustomImputer(
    DefaultParamsReadable,
    DefaultParamsWritable,
    Transformer
):

    tokens = Param(Params()._dummy(), "tokens", "dict: column_name -> value")

    @keyword_only
    def __init__(self, tokens=None):
        super(CustomImputer, self).__init__()
        kwargs = self._input_kwargs
        self._set(**kwargs)

    def getTokens(self):
        return self.getOrDefault(self.tokens)

    def setTokens(self, tokens):
        return self.setParams(tokens=tokens)

    def _transform(self, df):
        df = df.fillna(self.getTokens())
        return df

customImputer = CustomImputer(tokens={"reviewText": "missing"})
tokenizer = Tokenizer(inputCol = 'reviewText', outputCol = 'tokens')
hashingTF = HashingTF(numFeatures=100, binary=True, inputCol = 'tokens', outputCol = 'rawFeatures')
lr = LogisticRegression(featuresCol='rawFeatures',
                        labelCol='overall')
pipeline = Pipeline(stages = [customImputer, tokenizer, hashingTF, lr])
