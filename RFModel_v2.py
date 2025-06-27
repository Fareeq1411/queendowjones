import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.utils import class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from imblearn.over_sampling import SMOTE

# Set random seed for reproducibility
np.random.seed(42)

# Load the pattern recognition dataset
file_path = "/Users/fareeq1411/Desktop/ITProject/QueenDowJones.nosync/5M_PatternRec_v2.csv"
df = pd.read_csv(file_path)

# Ensure necessary columns exist
if 'label' not in df.columns:
    raise KeyError("The 'label' column is missing in the dataset.")

# Extract features and labels
X = df.drop(columns=['label']).values
y = df['label'].values

# Ensure no NaN values in features
X = np.nan_to_num(X)

# Train/test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)

# Balance dataset using SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Scale features using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler
joblib.dump(scaler, "/Users/fareeq1411/Desktop/ITProject/QueenDowJones.nosync/scaler_pattern.pkl")
print("Scaler saved as 'scaler_pattern.pkl'")

# Compute class weights
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {cls: weight for cls, weight in zip(np.unique(y_train), class_weights)}

# Initialize the Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=100, max_depth=10, min_samples_split=20, min_samples_leaf=5, 
    bootstrap=True, random_state=42, class_weight=class_weight_dict
)

# Time-Series Split Cross-Validation
tscv = TimeSeriesSplit(n_splits=5)
fold = 1
for train_index, val_index in tscv.split(X_train):
    print(f"Training fold {fold}/5...")
    
    # Split data into training and validation for this fold
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

    # Train the model
    rf_model.fit(X_train_fold, y_train_fold)

    # Predict on the validation set
    y_val_pred = rf_model.predict(X_val_fold)

    # Calculate accuracy for this fold
    fold_accuracy = accuracy_score(y_val_fold, y_val_pred)
    print(f"Accuracy for fold {fold}: {fold_accuracy:.4f}")
    fold += 1

# Evaluate the model on the test set
y_test_pred = rf_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Save the trained model
joblib.dump(rf_model, "/Users/fareeq1411/Desktop/ITProject/QueenDowJones.nosync/random_forest_pattern_model.pkl")
print("Model saved as 'random_forest_pattern_model.pkl'")
