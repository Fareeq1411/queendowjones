
import numpy as np
import pandas as pd
import random
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.utils import class_weight  # Import for class weights

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# File paths
csv_file_url = '/Users/fareeq1411/Desktop/ITProject/QueenDowJones.nosync/US3060_Train.csv'
test_data_url = '/Users/fareeq1411/Desktop/ITProject/QueenDowJones.nosync/US3060_test.csv'

# Hyperparameters
n_estimators = 100
max_depth = 40
min_samples_split = 10
min_samples_leaf = 1
bootstrap = False
timesteps = 4
price_change_threshold = 0.03
folds = 10

# Load and preprocess data
dataset_train = pd.read_csv(csv_file_url, skipinitialspace=True)
dataset_test = pd.read_csv(test_data_url, skipinitialspace=True)

# Strip extra spaces from column names
dataset_train.columns = [col.strip() for col in dataset_train.columns]
dataset_test.columns = [col.strip() for col in dataset_test.columns]

# Select relevant features if 'open' column exists
if 'open' in dataset_train.columns and 'open' in dataset_test.columns:
    training_set1 = dataset_train[['open', 'close', 'high', 'low', 'volume', 'RSI_6', 'MACD', 'MACD_Signal', 'Upper_BB', 'Lower_BB']].values
    test_set1 = dataset_test[['open', 'close', 'high', 'low', 'volume', 'RSI_6', 'MACD', 'MACD_Signal', 'Upper_BB', 'Lower_BB']].dropna().values
else:
    raise KeyError("The 'open' column is not present in one of the datasets")

# Store unscaled features
unscaled_features = training_set1[~np.isnan(training_set1).any(axis=1)]  # Use the original data without NaNs

# Remove rows with NaNs from scaled features
training_set = training_set1[~np.isnan(training_set1).any(axis=1)]

# Scale features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(training_set)

# Prepare input and output
X_train, y_train = [], []

# Adjusted training data loop with percentage-based price change threshold using unscaled data
for i in range(timesteps, len(scaled_features)):
    # Append timesteps of scaled features to X_train
    current_input = list(scaled_features[i - timesteps:i].flatten())  # Flatten the timesteps

    X_train.append(current_input)

    # Get the open prices for current and next step using unscaled data
    next_close_unscaled = unscaled_features[i, 1]  # Retrieve next closing price (assuming it's in the second column)
    current_open_unscaled = unscaled_features[i - 1, 0]  # Current open price
    
    # Calculate the percentage change in price using unscaled open prices
    if current_open_unscaled != 0:  # Avoid division by zero
        price_change = ((next_close_unscaled - current_open_unscaled) / current_open_unscaled) * 100
    
        # Determine the class label based on percentage change
        if price_change > price_change_threshold:
            y_train.append(0)  # Buy
        elif price_change < -price_change_threshold:
            y_train.append(2)  # Sell
        else:
            y_train.append(1)  # Hold
    else:
        y_train.append(1)  # Default to Hold if current_open is zero

# Convert lists to arrays
X_train = np.array(X_train)
y_train = np.array(y_train)
X_train_flat = X_train.reshape(X_train.shape[0], -1)

# Calculate class weights
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

# K-Fold Cross-Validation
kf = KFold(n_splits=folds, shuffle=True, random_state=42)
accuracies = []

for fold, (train_index, val_index) in enumerate(kf.split(X_train_flat)):
    X_fold_train, X_fold_val = X_train_flat[train_index], X_train_flat[val_index]
    y_fold_train, y_fold_val = y_train[train_index], y_train[val_index]

    rf_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split,
                                      min_samples_leaf=min_samples_leaf, bootstrap=bootstrap, random_state=42,
                                      class_weight=class_weight_dict)  # Add class_weight here
    rf_model.fit(X_fold_train, y_fold_train)

    y_val_pred = rf_model.predict(X_fold_val)
    accuracy = accuracy_score(y_fold_val, y_val_pred)
    accuracies.append(accuracy)
    print(f"Fold {fold + 1}, Accuracy: {accuracy:.4f}")

# Save model and scaler
joblib.dump(rf_model, 'random_forest_modelv2.pkl')
joblib.dump(scaler, 'scalerv2.pkl')
print(f"Average K-Fold Accuracy: {np.mean(accuracies):.4f}")
print("Training completed!")
# Alert when complete
os.system('say "Training completed!"')






"""model_path = 'random_forest_model.pkl'
# Load the saved model for predictions
rf_model = joblib.load(model_path)

# Test data setup
dataset_total = pd.DataFrame(training_set1).dropna()
test_set1_scaled = scaler.transform(test_set1)
inputs_scaled = scaler.transform(dataset_total.iloc[-timesteps:, :].values)

predictions, real_decisions, real_prices_unscaled = [], [], []
previous_real_price_unscaled = inputs_scaled[-1, 0]  # Initialize with the last real price before scaling

# Print header for results
print(f"{'Time Step':<10}{'Real Price':<15}{'Model Prediction':<20}{'Real Decision':<15}")
print("=" * 90)

# Start prediction loop from the point after the initial timesteps
start_idx = timesteps  # Start predicting after the initial timesteps
for idx in range(start_idx, len(test_set1_scaled)):
    # Prepare the test input for prediction from the test set directly
    X_test = inputs_scaled[-timesteps:].flatten().tolist()  # Flatten the last 'timesteps' inputs
    X_test = np.array(X_test).reshape(1, -1)

    # Get numerical prediction
    action = rf_model.predict(X_test)[0]

    # Map numerical action to string
    action_name = {0: 'Buy', 1: 'Hold', 2: 'Sell'}[action]
    predictions.append(action_name)

    # Get the unscaled real price for the current step
    current_real_price_unscaled = test_set1[idx, 0]
    real_prices_unscaled.append(current_real_price_unscaled)

    # Log the prediction for the current price point
    if idx < len(test_set1_scaled) - 1:  # Check if there's a next price to compare
        next_real_price_unscaled = test_set1[idx + 1, 0]
        percentage_change = ((next_real_price_unscaled - current_real_price_unscaled) /
                             current_real_price_unscaled) * 100

        if percentage_change < -price_change_threshold:
            real_decision = 'Sell'
        elif percentage_change > price_change_threshold:
            real_decision = 'Buy'
        else:
            real_decision = 'Hold'

        real_decisions.append(real_decision)
        print(f"{idx:<10}{current_real_price_unscaled:<15.2f}{action_name:<20}{real_decision:<15}")

    else:
        real_decisions.append('N/A')

    # Update the scaled inputs array for the next test input
    inputs_scaled = np.append(inputs_scaled, test_set1_scaled[idx].reshape(1, -1), axis=0)

# Calculate accuracy
correct_predictions = sum(1 for real, pred in zip(real_decisions[1:], predictions[1:]) if real == pred)
average_accuracy = (correct_predictions / (len(predictions) - 1) * 100) if len(predictions) > 1 else 0
print(f"\nAverage Accuracy: {average_accuracy:.2f}%")

# Plotting Real Prices using the original unscaled prices
plt.figure(figsize=(14, 7))
plt.plot(real_prices_unscaled, color='blue', label='Real Prices', linewidth=1.5)

# Annotate each prediction (Buy, Hold, Sell) on the data points
for idx, action in enumerate(predictions):
    plt.text(idx, real_prices_unscaled[idx], action,
             fontsize=9, ha='center',
             color='green' if action == 'Buy' else 'red' if action == 'Sell' else 'black')

plt.title('Real Prices with Model Predictions')
plt.xlabel('Time Steps')
plt.ylabel('Price')
plt.legend(['Real Prices'], loc="upper left")
plt.grid(True)
plt.tight_layout()
plt.show()

# Alert when complete
os.system('say "Prediction completed!"') """
