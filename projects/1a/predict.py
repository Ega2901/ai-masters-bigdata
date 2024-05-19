#!/opt/conda/envs/dsenv/bin/python
import sys
import pandas as pd
from joblib import load
from model import fields

def main(model_path, input_file_path):
    # Загрузка модели
    model = load(model_path)
    
    # Загрузка данных
    data = pd.read_csv(input_file_path, sep='\t', names=fields)
    
    # Разделение данных на признаки
    X = data.drop(columns=["id", "label"])
    
    # Получение предсказаний
    predictions = model.predict_proba(X)[:, 1]
    
    # Вывод предсказаний
    for pred in predictions:
        print(pred)

if __name__ == "__main__":
    model_path = sys.argv[1]
    input_file_path = sys.argv[2]
    main(model_path, input_file_path)
