#!/opt/conda/envs/dsenv/bin/python

import sys, os
import logging
from joblib import load
import pandas as pd
import numpy as np

sys.path.append('.')
# from model import fields_no_label

#
# Init the logger
#
logging.basicConfig(level=logging.DEBUG)
logging.info("CURRENT_DIR {}".format(os.getcwd()))
logging.info("SCRIPT CALLED AS {}".format(sys.argv[0]))
logging.info("ARGS {}".format(sys.argv[1:]))

#load the model
model = load("2a.joblib")

numeric_features = ["if"+str(i) for i in range(1,14)]
categorical_features = ["cf"+str(i) for i in range(1,27)]
fields = ["id", "label"] + numeric_features + categorical_features
fields_no_label = ["id"] + numeric_features + categorical_features

read_opts=dict(sep='\t', names=fields_no_label, index_col=False, header=None, iterator=True, chunksize=100)

for df in pd.read_csv(sys.stdin, **read_opts):
    df.iloc[:, :14] = df.iloc[:, :14].replace('\\N', 0)
    df.iloc[:, :14] = df.iloc[:, :14].replace('', 0)
    df.iloc[:, 14:] = df.iloc[:, 14:].replace('\\N', '')
    X=df.drop(df.columns[[0]], axis=1)
    pred = model.predict_proba(X)
    out = zip(df.id, pred[:,0])
    print("\n".join(["{0}\t{1}".format(*i) for i in out]))
