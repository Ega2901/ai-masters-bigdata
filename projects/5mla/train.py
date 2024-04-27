#!/opt/conda/envs/dsenv/bin/python
import os
import sys
import logging
import mlflow
import mlflow.sklearn

import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import dump

# Import model definition
from model import model

# Logging initialization
logging.basicConfig(level=logging.DEBUG)
logging.info("CURRENT_DIR {}".format(os.getcwd()))
logging.info("SCRIPT CALLED AS {}".format(sys.argv[0]))
logging.info("ARGS {}".format(sys.argv[1:]))

# Read script arguments
if len(sys.argv) < 3:
    logging.critical("Need to pass train dataset path and model_param1")
    sys.exit(1)

train_path = sys.argv[1]
print("TRAINPATH==", train_path)
model_param1 = float(sys.argv[2])
logging.info(f"TRAIN_PATH {train_path}")
logging.info(f"MODEL_PARAM1 {model_param1}")

# Read dataset
numeric_features = ["if"+str(i) for i in range(1, 14)]
categorical_features = ["cf"+str(i) for i in range(1, 27)]
fields = ["id", "label"] + numeric_features + categorical_features

# Определение опций чтения данных
read_table_opts = dict(sep="\t", names=fields, index_col=False)
df = pd.read_table(train_path, **read_table_opts)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    df.iloc[:,:-1], df.iloc[:,-1], test_size=0.2, random_state=42
)

# Log parameters
mlflow.log_param("model_param1", model_param1)

# End current run, if exists
if mlflow.active_run():
    mlflow.end_run()

# Start MLflow run
with mlflow.start_run() as run:
    # Train the model
    model.fit(X_train, y_train)
    
    # Evaluate model
    model_score = model.score(X_test, y_test)
    
    # Log metrics
    mlflow.log_metric("model_score", model_score)
    
    logging.info(f"model score: {model_score:.3f}")

    # Save the model
    model_output_path = "model.joblib"
    dump(model, model_output_path)

    # Log model output
    mlflow.log_artifact(model_output_path)
