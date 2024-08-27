import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Sample dataset
data = pd.DataFrame({
    'Size': [1500, 1800, 2400, 2000, 1300],
    'Location': ['Urban', 'Suburban', 'Urban', 'Urban', 'Suburban'],
    'Rooms': [3, 4, 4, 3, 2],
    'Age': [10, 5, 15, 7, 20],
    'Price': [300000, 350000, 450000, 400000, 250000]
})

# Features and target
X = data.drop('Price', axis=1)
y = data['Price']

# Define feature columns
numerical_features = ['Size', 'Rooms', 'Age']
categorical_features = ['Location']

# Pipelines
numerical_pipeline = Pipeline(steps=[
    ('scaler', StandardScaler())
])
categorical_pipeline = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ])

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])