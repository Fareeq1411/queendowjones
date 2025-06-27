#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 20:51:27 2024

@author: fareeq1411
"""
import pandas as pd
import numpy as np
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import os

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Load the dataset
csv_file_url = '/Users/fareeq1411/Desktop/ITProject/QueenDowJones/US3060_Train.csv'
test_data_url = '/Users/fareeq1411/Desktop/ITProject/QueenDowJones/US3060_test.csv'

dataset_train = pd.read_csv(csv_file_url, skipinitialspace=True)
dataset_test = pd.read_csv(test_data_url, skipinitialspace=True)

# Strip extra spaces from column names
dataset_train.columns = [col.strip() for col in dataset_train.columns]
dataset_test.columns = [col.strip() for col in dataset_test.columns]

# Check if 'open' column exists and select relevant features
if 'open' in dataset_train.columns and 'open' in dataset_test.columns:
    training_set1 = dataset_train[['open', 'close', 'high' , 'low', 'volume']].values
    test_set1 = dataset_test['open'].values
else:
    raise KeyError("The 'open' column is not present in one of the datasets")

# Prepare the dataset for Random Forest
timesteps = 30
future_steps = 2
X_train, y_train = [], []

# Create data structure with 30 timesteps and future steps
for i in range(timesteps, len(training_set1) - future_steps + 1):
    X_train.append(training_set1[i - timesteps:i].flatten())  # Flatten the features
    y_train.append(training_set1[i:i + future_steps, 0])  # Predict 'open' price (index 0)

X_train = np.array(X_train)
y_train = np.array(y_train).reshape(-1, future_steps)

# Check for and remove NaN values from X_train and y_train
nan_mask = ~np.isnan(y_train[:, 0])  # We care about the first future step for now
X_train = X_train[nan_mask]
y_train = y_train[nan_mask]

# Split the data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Set up the Random Forest Regressor
rf_model = RandomForestRegressor(random_state=42)

# Define hyperparameters to tune
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Perform GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

# Train the model
grid_search.fit(X_train, y_train[:, 0])  # Train for the first future step prediction

# Best hyperparameters
best_params = grid_search.best_params_
print(f"Best Hyperparameters: {best_params}")

# Train the Random Forest Regressor with the best hyperparameters
best_rf_model = RandomForestRegressor(**best_params, random_state=42)
best_rf_model.fit(X_train, y_train[:, 0])

# Predict on validation data
y_pred = best_rf_model.predict(X_val)

# Evaluate the model
mae = mean_absolute_error(y_val[:, 0], y_pred)
print(f'Mean Absolute Error: {mae}')

# Prepare the test data for prediction (similar process as above)
test_set1_flattened = []

for i in range(timesteps, len(test_set1)):
    test_set1_flattened.append(test_set1[i - timesteps:i].flatten())

X_test = np.array(test_set1_flattened)

# Make predictions on the test set
predictions = best_rf_model.predict(X_test)

# Plot the results
real_values_flat = test_set1[:len(predictions)]
time_steps = np.arange(len(real_values_flat))

plt.figure(figsize=(14, 7))
plt.plot(time_steps, real_values_flat, color='blue', label='Real Values', marker='o', markersize=5)
plt.plot(time_steps, predictions, color='red', linestyle='--', label='Predicted Values', marker='o', markersize=5)

plt.title('Real vs Predicted Values - Random Forest')
plt.xlabel('Time Steps')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

os.system('say "Prediction completed!"')
