#!/opt/conda/envs/dsenv/bin/python

import os
import sys
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from joblib import dump
import mlflow
import mlflow.sklearn

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
    model_param1 = int(sys.argv[3])  # Новый параметр для модели
except:
    logging.critical("Need to pass project_id, train dataset path, and model_param1")
    sys.exit(1)

logging.info(f"TRAIN_ID {proj_id}")
logging.info(f"TRAIN_PATH {train_path}")
logging.info(f"MODEL_PARAM1 {model_param1}")

#
# Read dataset
#
read_table_opts = dict(sep="\t", names=fields, index_col=0)
df = pd.read_table(train_path, **read_table_opts)

# split train/test
X_train, X_test, y_train, y_test = train_test_split(
    df.iloc[:, 1:14], df.iloc[:, 0], test_size=0.33, random_state=42
)

#
# Train the model
#
# Пример: использование model_param1 в модели (при необходимости)
# model.set_params(param1=model_param1)

with mlflow.start_run():
    # Логирование параметров
    mlflow.log_param("train_path", train_path)
    mlflow.log_param("model_param1", model_param1)

    model.fit(X_train, y_train)

    y_pred = model.predict_proba(X_test)[:, 1]
    model_score = log_loss(y_test, y_pred)

    logging.info(f"model score: {model_score:.3f}")

    # Логирование метрики
    mlflow.log_metric("log_loss", model_score)

    # Сохранение модели с помощью mlflow
    mlflow.sklearn.log_model(model, "model")

    # Также сохранение модели локально, если необходимо
    dump(model, "{}.joblib".format(proj_id))
