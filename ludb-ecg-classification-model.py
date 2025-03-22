import wfdb
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import os

# Define paths
DATA_DIR = "/Users/vaideesh/Documents/Jupyter notebooks/ECG_Classification_LUDB/data"
CSV_DIR = "/Users/vaideesh/Documents/Jupyter notebooks/ECG_Classification_LUDB"
CSV_FILE = os.path.join(CSV_DIR, "ludb.csv")

# Target parameters
TARGET_DURATION = 10  # seconds
TARGET_FS = 500  # Hz
TARGET_SAMPLES = TARGET_DURATION * TARGET_FS  # 5000 samples per lead

def standardize_signal(signals, original_fs):
    """Resample and standardize ECG signal to 10 seconds at 500 Hz."""
    current_samples = signals.shape[0]
    current_duration = current_samples / original_fs

    # Resample to 500 Hz
    if original_fs != TARGET_FS:
        num_samples_new = int(current_duration * TARGET_FS)
        signals = resample(signals, num_samples_new, axis=0)

    # Adjust to exactly 10 seconds
    current_samples = signals.shape[0]
    if current_samples > TARGET_SAMPLES:
        signals = signals[:TARGET_SAMPLES, :]
    elif current_samples < TARGET_SAMPLES:
        pad_length = TARGET_SAMPLES - current_samples
        signals = np.pad(signals, ((0, pad_length), (0, 0)), mode='constant', constant_values=0)

    return signals

# Load the CSV data (only need Rhythms now)
def load_csv_data(csv_file):
    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip()
    return df[['ID', 'Rhythms']].set_index('ID')

# Load and preprocess ECG data without Sex and Age
def load_ecg_data(data_dir, record_ids, csv_data):
    ecg_data = []
    labels = []
    
    for record_id in record_ids:
        record_path = os.path.join(data_dir, str(record_id))
        try:
            record = wfdb.rdrecord(record_path)
        except Exception as e:
            print(f"Error loading record {record_id}: {e}")
            continue
        
        # Standardize the signal
        signals = standardize_signal(record.p_signal, record.fs)
        flattened_signal = signals.flatten()  # Shape: (60000,)
        
        ecg_data.append(flattened_signal)
        
        # Get the rhythm diagnosis
        rhythm = csv_data.loc[int(record_id), 'Rhythms'].strip()
        labels.append(rhythm)
    
    return np.array(ecg_data), np.array(labels)

# Get all 200 record IDs
record_ids = [str(i) for i in range(1, 201)]

# Load CSV data
print("Loading CSV data...")
csv_data = load_csv_data(CSV_FILE)

# Load ECG data and labels
print("Loading ECG data...")
X, y = load_ecg_data(DATA_DIR, record_ids, csv_data)

# Encode the labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Filter out classes with fewer than 2 samples
unique, counts = np.unique(y, return_counts=True)
valid_classes = unique[counts >= 2]
mask = np.isin(y, valid_classes)

X_filtered = X[mask]
y_filtered = y[mask]
y_encoded_filtered = y_encoded[mask]

print(f"Filtered dataset size: {len(X_filtered)} (removed {len(X) - len(X_filtered)} singleton classes)")

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X_filtered, y_encoded_filtered, test_size=0.2, random_state=42, stratify=y_encoded_filtered
)

# Train Random Forest model
print("Training Random Forest model...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
rf_model.fit(X_train, y_train)

# Evaluate the model
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Get unique labels in y_test and y_pred
unique_test_labels = np.unique(np.concatenate([y_test, y_pred]))
target_names = [label_encoder.classes_[i] for i in unique_test_labels]

print("Classification Report:")
print(classification_report(y_test, y_pred, labels=unique_test_labels, target_names=target_names, zero_division=0))

# Save the model and label encoder
import joblib
joblib.dump(rf_model, "ludb_rf_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
print("Model and label encoder saved.")