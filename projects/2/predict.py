#!/opt/conda/envs/dsenv/bin/python

import sys, os
import logging
from joblib import load
import pandas as pd

numeric_features = ["if"+str(i) for i in range(1,14)]
categorical_features = ["cf"+str(i) for i in range(1,27)]

# fields = ["id", "label"] + numeric_features + categorical_features
fields_without_category = ["id"] + numeric_features

sys.path.append('.')

#
# Init the logger
#
logging.basicConfig(level=logging.DEBUG)
logging.info("CURRENT_DIR {}".format(os.getcwd()))
logging.info("SCRIPT CALLED AS {}".format(sys.argv[0]))
logging.info("ARGS {}".format(sys.argv[1:]))

#load the model
model = load("2a.joblib")

read_opts=dict(
        sep='\t', names=fields, index_col=0, header=None,
        iterator=True, chunksize=100
)

for df in pd.read_csv(sys.stdin, **read_opts):
    pred = model.predict_proba(df.iloc[:,:13])
    out = zip(df.index, pred[:, 1])
    print("\n".join(["{0}\t{1}".format(*i) for i in out]))
