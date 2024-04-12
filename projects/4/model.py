from pyspark.sql.types import *
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.sql.types import *
import pyspark.sql.functions as f

Tokenizer = Tokenizer(inputCol = 'prep_text', outputCol = 'tokens')
hashingTF = HashingTF(inputCol = 'tokens', outputCol = 'rawFeatures')
assembler = VectorAssembler(inputCols=['rawFeatures'],
                            outputCol='features')
lr = LogisticRegression(featuresCol='features',
                        labelCol='overall')
pipeline = Pipeline(stages = [Textprep, Tokenizer, hashingTF, assembler, lr])
