import os
import sys
import numpy as np
import pandas as pd 
from sklearn.linear_model import LogisticRegression
from joblib import load


if __name__ == "__main__":
    input_path = str(sys.argv[1])
    out_path = str(sys.argv[2])
    model_path = str(sys.argv[3])
    df = pd.read_csv(input_path)
    X_test = df['rawFeatures']
    
    model = load(model_path)
    y_pred = model.predict(X_test)
    result_df = pd.DataFrame({'id': df['id'], 'prediction': y_pred})
    
    result_df.to_csv(out_path, index=False, header=False)
