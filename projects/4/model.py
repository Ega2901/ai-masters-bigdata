from pyspark.sql.types import *
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression


Tokenizer = Tokenizer(inputCol = 'reviewText', outputCol = 'tokens')
hashingTF = HashingTF(numFeatures=100, binary=True, inputCol = Tokenizer.getOutputCol(), outputCol = 'rawFeatures')
lr = LogisticRegression(featuresCol='rawFeatures',
                        labelCol='overall')
pipeline = Pipeline(stages = [Tokenizer, hashingTF, assembler, lr])
