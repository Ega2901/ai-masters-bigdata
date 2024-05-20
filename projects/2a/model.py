from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
import pandas as pd

#
# Dataset fields
#
numeric_features = ["if"+str(i) for i in range(1,14)]
categorical_features = ["cf"+str(i) for i in range(1,27)]
fields = ["id", "label"] + numeric_features + categorical_features
fields_no_label = ["id"] + numeric_features + categorical_features

#
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_features = ['cf1', 'cf3', 'cf4', 'cf6',  'cf10', 'cf11', 'cf13', 'cf14', 'cf15', 'cf17', 'cf20', 'cf22', 'cf23', 'cf25']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('LogisticRegression', LogisticRegression(
                                              max_iter=10000,
                                            #   C=0.1,
                                            #   fit_intercept=True,
                                            #   penalty='l1',
                                            #   solver='saga',
                                            ))
])
