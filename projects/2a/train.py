#!/opt/conda/envs/dsenv/bin/python

import os, sys
import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from joblib import dump

#
# Import model definition
#
from model import model, fields


#
# Logging initialization
#
logging.basicConfig(level=logging.DEBUG)
logging.info("CURRENT_DIR {}".format(os.getcwd()))
logging.info("SCRIPT CALLED AS {}".format(sys.argv[0]))
logging.info("ARGS {}".format(sys.argv[1:]))

#
# Read script arguments
#
try:
  proj_id = sys.argv[1] 
  train_path = sys.argv[2]
except:
  logging.critical("Need to pass both project_id and train dataset path")
  sys.exit(1)


logging.info(f"TRAIN_ID {proj_id}")
logging.info(f"TRAIN_PATH {train_path}")

read_table_opts = dict(sep="\t", names=fields, index_col=0)
df = pd.read_table(train_path, **read_table_opts)

#split train/test
X_train, X_test, y_train, y_test = train_test_split(
    df.iloc[:,1:14], df.iloc[:,0], test_size=0.33, random_state=42
)

#
# Train the model
#
model.fit(X_train, y_train)

y_pred = model.predict_proba(X_test)[:,1]
model_score = log_loss(y_test, y_pred)

logging.info(f"model score: {model_score:.3f}")

# save the model
dump(model, "{}.joblib".format(proj_id))

