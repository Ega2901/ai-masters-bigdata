#!/opt/conda/envs/dsenv/bin/python
import sys
import pandas as pd
from joblib import dump
from model import model, fields

def main(project_number, training_file_path):
    data = pd.read_csv(training_file_path, sep='\t', names=fields)
    
    X = data.drop(columns=["id", "label"])
    y = data["label"]
    
    model.fit(X, y)
    
    dump(model, f"{project_number}.joblib")

if __name__ == "__main__":
    project_number = sys.argv[1]
    training_file_path = sys.argv[2]
    main(project_number, training_file_path)


