from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import os
import shutil
import wfdb
import numpy as np
import joblib
import tempfile
import uvicorn
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
import io
import base64
from scipy.signal import resample
import tensorflow as tf
import gcsfs
import sys

# Debug: Log Python and TensorFlow versions
print(f"Python version: {sys.version}")
print(f"TensorFlow version: {tf.__version__}")

app = FastAPI()

# Mount static directory for serving HTML
app.mount("/static", StaticFiles(directory="static"), name="static")

# Google Cloud Storage setup
GCS_BUCKET = "ludb-ecg-models2"  # Updated bucket name
MODEL_FILES = {
    "ludb_rf_model.pkl": f"gs://{GCS_BUCKET}/ludb_rf_model.pkl",
    "ludb_lr_model.pkl": f"gs://{GCS_BUCKET}/ludb_lr_model.pkl",
    "ludb_lstm_model.h5": f"gs://{GCS_BUCKET}/ludb_lstm_model.h5",
    "ludb_cnn_model.h5": f"gs://{GCS_BUCKET}/ludb_cnn_model.h5",
    "label_encoder.pkl": f"gs://{GCS_BUCKET}/label_encoder.pkl"
}

# Download models from GCS
fs = gcsfs.GCSFileSystem()
for local_file, gcs_path in MODEL_FILES.items():
    if not os.path.exists(local_file):
        try:
            print(f"Downloading {local_file} from {gcs_path}...")
            fs.get(gcs_path, local_file)
            print(f"Successfully downloaded {local_file}")
        except Exception as e:
            print(f"Error downloading {local_file} from {gcs_path}: {str(e)}")
            raise  # Re-raise the exception to fail startup if downloads fail

# Load the trained models and label encoder
try:
    print("Loading Random Forest model...")
    rf_model = joblib.load("ludb_rf_model.pkl")
    print("Random Forest model loaded successfully")
except Exception as e:
    print(f"Error loading Random Forest model: {str(e)}")
    raise

try:
    print("Loading Logistic Regression model...")
    lr_model = joblib.load("ludb_lr_model.pkl")
    print("Logistic Regression model loaded successfully")
except Exception as e:
    print(f"Error loading Logistic Regression model: {str(e)}")
    raise

try:
    print("Loading LSTM model...")
    lstm_model = tf.keras.models.load_model("ludb_lstm_model.h5")
    print("LSTM model loaded successfully")
except Exception as e:
    print(f"Error loading LSTM model: {str(e)}")
    raise

try:
    print("Loading CNN model...")
    cnn_model = tf.keras.models.load_model("ludb_cnn_model.h5")
    print("CNN model loaded successfully")
except Exception as e:
    print(f"Error loading CNN model: {str(e)}")
    raise

try:
    print("Loading label encoder...")
    label_encoder = joblib.load("label_encoder.pkl")
    print("Label encoder loaded successfully")
except Exception as e:
    print(f"Error loading label encoder: {str(e)}")
    raise

# Target parameters
TARGET_DURATION = 10  # seconds
TARGET_FS = 500  # Hz
TARGET_SAMPLES = TARGET_DURATION * TARGET_FS  # 5000 samples per lead

def standardize_signal(signals, original_fs):
    """Resample and standardize ECG signal to 10 seconds at 500 Hz."""
    current_samples = signals.shape[0]
    current_duration = current_samples / original_fs

    if original_fs != TARGET_FS:
        num_samples_new = int(current_duration * TARGET_FS)
        signals = resample(signals, num_samples_new, axis=0)

    current_samples = signals.shape[0]
    if current_samples > TARGET_SAMPLES:
        signals = signals[:TARGET_SAMPLES, :]
    elif current_samples < TARGET_SAMPLES:
        pad_length = TARGET_SAMPLES - current_samples
        signals = np.pad(signals, ((0, pad_length), (0, 0)), mode='constant', constant_values=0)

    return signals

def plot_ecg_signal(signals, lead_names):
    """Generate a plot of the standardized 12-lead ECG signal."""
    num_samples = signals.shape[0]
    time = np.arange(num_samples) / TARGET_FS

    fig, axes = plt.subplots(12, 1, figsize=(10, 12), sharex=True)
    for i in range(12):
        axes[i].plot(time, signals[:, i], label=lead_names[i])
        axes[i].set_ylabel(lead_names[i])
        axes[i].grid(True)
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close(fig)
    return img_base64

def get_top_3_rhythms(probabilities, model_name):
    """Extract the top 3 rhythms and their confidences from a model's probabilities."""
    top_3_indices = np.argsort(probabilities)[-3:][::-1]
    top_3_confidences = probabilities[top_3_indices]
    top_3_rhythms = label_encoder.inverse_transform(top_3_indices)
    return list(zip(top_3_rhythms, top_3_confidences))

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html", "r") as f:
        return f.read()

@app.post("/classify-ecg/")
async def classify_ecg(
    dat_file: UploadFile = File(...),
    hea_file: UploadFile = File(...)
):
    with tempfile.TemporaryDirectory() as temp_dir:
        dat_path = os.path.join(temp_dir, dat_file.filename)
        hea_path = os.path.join(temp_dir, hea_file.filename)
        with open(dat_path, "wb") as f:
            shutil.copyfileobj(dat_file.file, f)
        with open(hea_path, "wb") as f:
            shutil.copyfileobj(hea_file.file, f)

        record_name = os.path.splitext(dat_file.filename)[0]
        try:
            record = wfdb.rdrecord(os.path.join(temp_dir, record_name))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing ECG files: {str(e)}")

        signals = standardize_signal(record.p_signal, record.fs)
        flattened_signal = signals.flatten()
        reshaped_signal = signals.reshape(1, TARGET_SAMPLES, 12)

        # Predict with all models and get top 3 rhythms
        # Random Forest
        rf_prob = rf_model.predict_proba(flattened_signal.reshape(1, -1))[0]
        rf_top_3 = get_top_3_rhythms(rf_prob, "Random Forest")
        rf_results = [(rhythm, confidence, "Random Forest") for rhythm, confidence in rf_top_3]

        # Logistic Regression
        lr_prob = lr_model.predict_proba(flattened_signal.reshape(1, -1))[0]
        lr_top_3 = get_top_3_rhythms(lr_prob, "Logistic Regression")
        lr_results = [(rhythm, confidence, "Logistic Regression") for rhythm, confidence in lr_top_3]

        # LSTM
        lstm_prob = lstm_model.predict(reshaped_signal, verbose=0)[0]
        lstm_top_3 = get_top_3_rhythms(lstm_prob, "LSTM")
        lstm_results = [(rhythm, confidence, "LSTM") for rhythm, confidence in lstm_top_3]

        # CNN
        cnn_prob = cnn_model.predict(reshaped_signal, verbose=0)[0]
        cnn_top_3 = get_top_3_rhythms(cnn_prob, "CNN")
        cnn_results = [(rhythm, confidence, "CNN") for rhythm, confidence in cnn_top_3]

        # Combine all results and get the overall top 3
        all_results = rf_results + lr_results + lstm_results + cnn_results
        all_results.sort(key=lambda x: x[1], reverse=True)  # Sort by confidence
        top_3_overall = all_results[:3]  # Take the top 3

        # Format the top 3 rhythms
        top_3_rhythms = []
        for rhythm, confidence, model in top_3_overall:
            top_3_rhythms.append({
                "rhythm": rhythm,
                "confidence": f"{confidence:.2f}",
                "model": model
            })

        # Generate ECG plot
        ecg_plot_base64 = plot_ecg_signal(signals, record.sig_name)

        return {
            "top_3_rhythms": top_3_rhythms,
            "ecg_plot": ecg_plot_base64
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)