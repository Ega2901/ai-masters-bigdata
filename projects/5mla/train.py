import os
import sys
import logging
import mlflow
import mlflow.sklearn

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import log_loss
from joblib import dump

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
model_param1 = float(sys.argv[2])
logging.info(f"TRAIN_PATH {train_path}")
logging.info(f"MODEL_PARAM1 {model_param1}")

# Read dataset


# Define features and target
numeric_features = ["if"+str(i) for i in range(1, 14)]
categorical_features = ["cf"+str(i) for i in range(1, 27)]
fields = ["id", "label"] + numeric_features + categorical_features

# Определение опций чтения данных
read_table_opts = dict(sep="\t", names=fields, index_col=False)
df = pd.read_csv(train_path, **read_table_opts) # Assuming CSV format, you may need to adjust this based on your actual dataset format
X = df.drop(columns=['label'])
y = df['label']

# Define preprocessing pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(missing_values=pd.NA, strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(missing_values=pd.NA, strategy='constant', fill_value='miss')),
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Log parameters
mlflow.log_param("model_param1", model_param1)

# End current run, if exists
if mlflow.active_run():
    mlflow.end_run()

# Start MLflow run
with mlflow.start_run() as run:
    # Define the full pipeline including preprocessing and model
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression())  # You can replace this with your custom model
    ])

    # Train the model
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict_proba(X_test)
    loss = log_loss(y_test, y_pred)
    
    # Log metrics
    mlflow.log_metric("log_loss", loss)
    
    logging.info(f"log loss: {loss:.3f}")

    # Save the model
    model_output_path = "MLmodel"
    dump(model, model_output_path)

    # Log model output
    mlflow.log_artifact(model_output_path)
