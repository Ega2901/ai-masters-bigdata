import os, sys
import logging
import mlflow
import mlflow.sklearn

import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import dump

# Import model definition
from model import model, fields

# Logging initialization
logging.basicConfig(level=logging.DEBUG)
logging.info("CURRENT_DIR {}".format(os.getcwd()))
logging.info("SCRIPT CALLED AS {}".format(sys.argv[0]))
logging.info("ARGS {}".format(sys.argv[1:]))

# Read script arguments
try:
    proj_id = sys.argv[1]
    train_path = sys.argv[2]
    model_param1 = float(sys.argv[3]) if len(sys.argv) > 3 else None
except:
    logging.critical("Need to pass project_id, train dataset path, and model_param1")
    sys.exit(1)

logging.info(f"TRAIN_ID {proj_id}")
logging.info(f"TRAIN_PATH {train_path}")
logging.info(f"MODEL_PARAM1 {model_param1}")

# Read dataset
read_table_opts = dict(sep="\t", names=fields, index_col=False)
df = pd.read_table(train_path, **read_table_opts)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    df.iloc[:,:-1], df.iloc[:,-1], test_size=0.2, random_state=42
)

# Log parameters
mlflow.log_param("model_param1", model_param1)

# Start MLflow run
with mlflow.start_run() as run:
    # Train the model
    model.fit(X_train, y_train)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
    
    # Evaluate model
    model_score = model.score(X_test, y_test)
    
    # Log metrics
    mlflow.log_metric("model_score", model_score)
    
    logging.info(f"model score: {model_score:.3f}")

    # Save the model
    model_output_path = "{}.joblib".format(proj_id)
    dump(model, model_output_path)

    # Log model output
    mlflow.log_artifact(model_output_path)
