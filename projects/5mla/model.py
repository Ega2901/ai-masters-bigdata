from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import pandas as pd

# Определение списка признаков
numeric_features = ["if"+str(i) for i in range(1, 14)]
categorical_features = ["cf"+str(i) for i in range(1, 25)]
fields = ["id", "label"] + numeric_features + categorical_features

# Определение опций чтения данных
read_table_opts = dict(sep="\t", names=fields, index_col=False)

# Определение преобразований для числовых и категориальных признаков
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(missing_values=pd.NA, strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(missing_values=pd.NA, strategy='constant', fill_value='miss')),
    ('onehot', OneHotEncoder(drop='first'))
])

# Создание объекта ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Определение модели
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])
