import sys
import glob
import pandas as pd
import json
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from joblib import dump


path_out= sys.argv[1] #"train.json"
model_path=sys.argv[2] #"model_sp"

files = glob.glob(path_out + '/*.json')
data = []

for file in files:
    with open(file, 'r') as f:
        data.extend([json.loads(line) for line in f])

df = pd.DataFrame(data)

X = df.drop('label', axis=1)
y = df['label']

print (df.columns)

clf = LogisticRegression(solver='liblinear')
clf.fit(X, y)

dump(clf, model_path)
