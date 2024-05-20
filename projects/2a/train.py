#!/opt/conda/envs/dsenv/bin/python

import os, sys
import logging
import pandas as pd
from sklearn.metrics import log_loss
from joblib import dump

#
# Import model definition
#
from model import model, fields, fields_no_label


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

#
# Read dataset
#
#fields = """doc_id,hotel_name,hotel_url,street,city,state,country,zip,class,price,
#num_reviews,CLEANLINESS,ROOM,SERVICE,LOCATION,VALUE,COMFORT,overall_ratingsource""".replace("\n",'').split(",")


train_data = pd.read_csv(train_path, sep='\t', names=fields)
X = train_data.drop(train_data.columns[[0, 1]], axis=1)
y=train_data['label']
model.fit(X, y)

model_score = log_loss(y, model.predict_proba(X))

logging.info(f"model score: {model_score:.3f}")

# save the model
dump(model, "{}.joblib".format(proj_id))
