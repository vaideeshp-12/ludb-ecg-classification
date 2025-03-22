from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import os
import shutil
import wfdb
import numpy as np
import pandas as pd
import joblib
import tempfile
import uvicorn
import matplotlib.pyplot as plt
import io
import base64
from scipy.signal import resample  # For resampling signals

app = FastAPI()

# Mount static directory for serving HTML
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load the trained model and label encoder
rf_model = joblib.load("ludb_rf_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Target parameters for LUDB compatibility
TARGET_DURATION = 10  # seconds
TARGET_FS = 500  # Hz
TARGET_SAMPLES = TARGET_DURATION * TARGET_FS  # 5000 samples per lead

def standardize_signal(signals, original_fs):
    """Resample and standardize ECG signal to 10 seconds at 500 Hz."""
    current_samples = signals.shape[0]
    current_duration = current_samples / original_fs

    # Resample to 500 Hz
    if original_fs != TARGET_FS:
        num_samples_new = int(current_duration * TARGET_FS)  # New number of samples at 500 Hz
        signals = resample(signals, num_samples_new, axis=0)

    # Adjust to exactly 10 seconds (5000 samples at 500 Hz)
    current_samples = signals.shape[0]
    if current_samples > TARGET_SAMPLES:
        # Truncate to first 10 seconds
        signals = signals[:TARGET_SAMPLES, :]
    elif current_samples < TARGET_SAMPLES:
        # Pad with zeros to reach 10 seconds
        pad_length = TARGET_SAMPLES - current_samples
        signals = np.pad(signals, ((0, pad_length), (0, 0)), mode='constant', constant_values=0)

    return signals

def plot_ecg_signal(record):
    """Generate a plot of the 12-lead ECG signal and return it as a base64 string."""
    signals = record.p_signal  # Original signal for plotting
    lead_names = record.sig_name
    num_samples = signals.shape[0]
    time = np.arange(num_samples) / record.fs

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

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html", "r") as f:
        return f.read()

@app.post("/classify-ecg/")
async def classify_ecg(
    dat_file: UploadFile = File(...),
    hea_file: UploadFile = File(...),
    sex: str = Form(...),
    age: float = Form(...)
):
    # Validate sex input
    sex = sex.strip().upper()
    if sex not in ["M", "F"]:
        raise HTTPException(status_code=400, detail="Sex must be 'M' or 'F'.")

    # Validate age input
    if age < 0 or age > 150:
        raise HTTPException(status_code=400, detail="Age must be between 0 and 150.")

    # Create temporary directory to store uploaded files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save uploaded files
        dat_path = os.path.join(temp_dir, dat_file.filename)
        hea_path = os.path.join(temp_dir, hea_file.filename)
        with open(dat_path, "wb") as f:
            shutil.copyfileobj(dat_file.file, f)
        with open(hea_path, "wb") as f:
            shutil.copyfileobj(hea_file.file, f)

        # Convert to WFDB record
        record_name = os.path.splitext(dat_file.filename)[0]
        try:
            record = wfdb.rdrecord(os.path.join(temp_dir, record_name))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing ECG files: {str(e)}")

        # Standardize the signal length and frequency
        signals = standardize_signal(record.p_signal, record.fs)

        # Extract 12-lead signals for prediction
        flattened_signal = signals.flatten()  # Always 60,000 values (5000 * 12)

        # Encode sex (M=1, F=0)
        sex_encoded = 1 if sex == "M" else 0

        # Combine features with user-provided sex and age
        combined_features = np.concatenate([flattened_signal, [sex_encoded, age]]).reshape(1, -1)

        # Predict rhythm
        prediction = rf_model.predict(combined_features)
        rhythm = label_encoder.inverse_transform(prediction)[0]

        # Generate ECG plot (use original record for plotting)
        ecg_plot_base64 = plot_ecg_signal(record)

        # Return both the rhythm and the plot
        return {"rhythm": rhythm, "ecg_plot": ecg_plot_base64}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)