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

app = FastAPI()

# Mount static directory for serving HTML
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load the trained model and label encoder
rf_model = joblib.load("ludb_rf_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html", "r") as f:
        return f.read()

def plot_ecg_signal(record):
    """Generate a plot of the 12-lead ECG signal and return it as a base64 string."""
    signals = record.p_signal  # Shape: (samples, 12)
    lead_names = record.sig_name  # List of lead names (e.g., ['I', 'II', ...])
    num_samples = signals.shape[0]
    time = np.arange(num_samples) / record.fs  # Time axis in seconds

    # Create a 12-subplot figure (one for each lead)
    fig, axes = plt.subplots(12, 1, figsize=(10, 12), sharex=True)
    for i in range(12):
        axes[i].plot(time, signals[:, i], label=lead_names[i])
        axes[i].set_ylabel(lead_names[i])
        axes[i].grid(True)
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()

    # Save plot to a BytesIO buffer and encode as base64
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close(fig)  # Close the figure to free memory
    return img_base64

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

        # Extract 12-lead signals for prediction
        signals = record.p_signal  # Shape: (samples, 12)
        flattened_signal = signals.flatten()

        # Encode sex (M=1, F=0)
        sex_encoded = 1 if sex == "M" else 0

        # Combine features with user-provided sex and age
        combined_features = np.concatenate([flattened_signal, [sex_encoded, age]]).reshape(1, -1)

        # Predict rhythm
        prediction = rf_model.predict(combined_features)
        rhythm = label_encoder.inverse_transform(prediction)[0]

        # Generate ECG plot
        ecg_plot_base64 = plot_ecg_signal(record)

        # Return both the rhythm and the plot
        return {"rhythm": rhythm, "ecg_plot": ecg_plot_base64}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)