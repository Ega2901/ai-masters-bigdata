from pyspark.sql.types import *
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression


class Textprep:
  
  def __init__(self):
    self.df = df
    
  def text_prep(text):
    text = str(text).lower()
    text = re.sub('\s+',' ',text)
    text = text.strip()
    return text
    
  def preptxt(self):
    prep_text_udf = udf(text_prep, StringType())
    df = (self.df).withColumn('prep_text', prep_text_udf("reviewText"))\
        .filter('prep_text <> ""')
    return df

Textprep = Textprep().preptxt
Tokenizer = Tokenizer(inputCol = 'prep_text', outputCol = 'tokens')
hashingTF = HashingTF(inputCol = 'tokens', outputCol = 'rawFeatures')
assembler = VectorAssembler(inputCols=['rawFeatures'],
                            outputCol='features')
lr = LogisticRegression(featuresCol='features',
                        labelCol='overall')
pipeline = Pipeline(stages = [Textprep, Tokenizer, hashingTF, assembler, lr])
