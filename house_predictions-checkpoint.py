from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
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

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fit the model
model_pipeline.fit(X_train, y_train)

# Predict and evaluate
y_pred = model_pipeline.predict(X_test)
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))


# Predict new house price
new_house = pd.DataFrame({
    'Size': [1600],
    'Location': ['Urban'],
    'Rooms': [3],
    'Age': [8]
})
predicted_price = model_pipeline.predict(new_house)
print(f"Predicted Price for the new house: ${predicted_price[0]:,.2f}")
