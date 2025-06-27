import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.utils import class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Set random seed for reproducibility
np.random.seed(42)

# File path
csv_file_url = '/Users/fareeq1411/Desktop/ITProject/QueenDowJones.nosync/5M_LABELED.csv'

# Hyperparameters
timesteps = 6  # Number of timesteps (previous 6 candles)
n_estimators = 100
max_depth = 40
min_samples_split = 10
min_samples_leaf = 1
bootstrap = False
n_splits = 5  # Number of splits for KFold Cross-Validation

# Load and preprocess data
dataset = pd.read_csv(csv_file_url, skipinitialspace=True)

# Strip extra spaces from column names
dataset.columns = [col.strip() for col in dataset.columns]

# Ensure necessary columns are present
if 'label' not in dataset.columns:
    raise KeyError("The 'label' column is not present in the dataset")

# Prepare feature columns (adjust according to your feature set)
feature_columns = ['Volume', 'RSI_6', 'MACD', 'MACD_Signal', 'Upper_BB', 'Lower_BB']
if not all(col in dataset.columns for col in feature_columns):
    raise KeyError("One or more required feature columns are not present in the dataset")

# Do not drop NaN values yet, keep all rows with NaN for now
# Prepare feature set
feature_set = dataset[feature_columns].values
labels = dataset['label'].values

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(feature_set, labels, test_size=0.2, random_state=42, shuffle=False)

# Scale features using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Function to create sequences with timesteps and corresponding labels (don't remove NaNs at first)
def create_sequences_with_labels(X, y, timesteps):
    X_seq, y_seq = [], []
    for i in range(timesteps, len(X)):
        if not np.isnan(y[i]):  # Only include data where the label is valid (not NaN)
            X_seq.append(X[i - timesteps:i].flatten())  # Include timesteps
            y_seq.append(y[i])  # The corresponding label for the next candle
    return np.array(X_seq), np.array(y_seq)

# Prepare training and testing sequences (with valid labels)
X_train_seq, y_train_seq = create_sequences_with_labels(X_train_scaled, y_train, timesteps)
X_test_seq, y_test_seq = create_sequences_with_labels(X_test_scaled, y_test, timesteps)

# Compute class weights using valid labels in y_train_seq
unique_y_train_labels = np.unique(y_train_seq)

# Check and correct if necessary
print("Labels in y_train_seq:", unique_y_train_labels)
print("Labels in y_test_seq:", np.unique(y_test_seq))

# Compute class weights
class_weights = class_weight.compute_class_weight('balanced', classes=unique_y_train_labels, y=y_train_seq)

# Create a dictionary for class weights
class_weight_dict = {cls: weight for cls, weight in zip(unique_y_train_labels, class_weights)}

# Print for verification
print("Class weights:", class_weight_dict)

# Initialize the model
rf_model = RandomForestClassifier(
    n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf, bootstrap=bootstrap, random_state=42,
    class_weight=class_weight_dict
)

# K-Fold Cross-Validation
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

fold = 1
for train_index, val_index in kf.split(X_train_seq):
    print(f"Training fold {fold}/{n_splits}...")

    # Split data into training and validation for this fold
    X_train_fold, X_val_fold = X_train_seq[train_index], X_train_seq[val_index]
    y_train_fold, y_val_fold = y_train_seq[train_index], y_train_seq[val_index]
    
    # Train the model
    rf_model.fit(X_train_fold, y_train_fold)
    
    # Predict on the validation set
    y_val_pred = rf_model.predict(X_val_fold)
    
    # Calculate accuracy for this fold
    fold_accuracy = accuracy_score(y_val_fold, y_val_pred)
    print(f"Accuracy for fold {fold}: {fold_accuracy:.4f}")
    
    fold += 1

# After cross-validation, evaluate the model on the test set
y_test_pred = rf_model.predict(X_test_seq)

# Evaluate the model on the test set
test_accuracy = accuracy_score(y_test_seq, y_test_pred)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Save the final model
joblib.dump(rf_model, 'random_forest_model_with_cross_validation3.0.pkl')
