#!/opt/conda/envs/dsenv/bin/python
import sys
import pandas as pd
from joblib import load
from model import fields

def main(model_path, input_file_path):
    model = load(model_path)
    
    data = pd.read_csv(input_file_path, sep='\t', names=fields)
    
    X = data.drop(columns=["id", "label"])
    
    predictions = model.predict_proba(X)[:, 1]
    
    for pred in predictions:
        print(pred)

if __name__ == "__main__":
    model_path = sys.argv[1]
    input_file_path = sys.argv[2]
    main(model_path, input_file_path)
