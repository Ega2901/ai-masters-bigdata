#!/opt/conda/envs/dsenv/bin/python
import os
import sys
from pyspark.ml.feature import VectorAssembler
import numpy as np
import pandas as pd 
from sklearn.linear_model import LogisticRegression
from joblib import dump, load

if __name__ == "__main__":
    input_path = str(sys.argv[1])
    out_path = str(sys.argv[2])
    df = pd.read_csv(input_path)
    X = df["rawFeatures"].tolist()
    y = df["label"]
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    dump(model, out_path)
    spark.stop()

