LUDB ECG Classification
This project is a web-based application for classifying 12-lead ECG signals using a Random Forest model trained on the Lobachevsky University Electrocardiography Database (LUDB). 
Upload .dat and .hea files to predict the best 3 diagnoses possibilities of cardiac rhythm and view the ECG plot.

Features
Classifies ECG rhythms (e.g., Sinus rhythm, Atrial fibrillation).
Standardizes signals to 10 seconds at 500 Hz (resamples and truncates/pads as needed).
Displays a 12-lead ECG plot alongside the predicted rhythm.
Deployed on Google Cloud Run for online access.
Project Structure

ludb-ecg-classification/
├── main.py                  # FastAPI app for the web interface and predictions
├── ludb-ecg-classification-model.py  # Script to train the model
├── ludb_rf_model.pkl        # Trained Random Forest model
├── label_encoder.pkl        # Label encoder for rhythm classes
├── static/
│   └── index.html           # HTML frontend
├── requirements.txt         # Python dependencies
└── Dockerfile               # For Cloud Run deployment

Prerequisites
Python 3.11+
LUDB Dataset (for training): Download from PhysioNet and place .dat, .hea, and ludb.csv in a data/ directory.

Setup
1. Clone the Repository
git clone https://github.com/vaideeshp-12/ludb-ecg-classification.git
cd ludb-ecg-classification

2. Set Up a Virtual Environment
python -m venv ludb_env
source ludb_env/bin/activate  # macOS/Linux
ludb_env\Scripts\activate     # Windows

3. Install Dependencies
pip install -r requirements.txt

4. (Optional) Retrain the Model
To retrain the model with the LUDB dataset:

Place the LUDB dataset files in data/ (e.g., data/1.dat, data/1.hea, ..., data/200.dat, data/200.hea).
Place ludb.csv in the project root.
Update paths in ludb-ecg-classification-model.py:
DATA_DIR = "path/to/your/data"
CSV_DIR = "path/to/your/csv"

Run:

python ludb-ecg-classification-model.py
Usage
Running Locally
Start the server:
python main.py
Open http://localhost:8080 in your browser.
Upload files:
Select a .hea file (header).
Select a .dat file (signal data).
Click "Convert and Classify ECG".
View the results:
Predicted rhythm (e.g., "Sinus rhythm").
A 10-second, 12-lead ECG plot.

Testing with Other Data
You can test with any 12-lead ECG dataset (e.g., MIT-BIH, PTB-XL) in .dat/.hea format. The app will automatically standardize the signal.

Deployment
The app is deployed on Google Cloud Run. To redeploy or deploy a new instance:

Install the Google Cloud SDK: Instructions.
Deploy:
gcloud run deploy ecg-classification \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --source .
Access the app via the provided URL
https://ludb-ecg-classification-713000400110.us-central1.run.app/static/index.html

Model Details
Training Data: LUDB dataset (200 records, 12-lead ECGs, 500 Hz, 10 seconds).
Models: Random Forest Classifier, Logistic regression, LSTM and CNN.
Features: Flattened 12-lead ECG signal (60,000 values).
Labels: 11 rhythm diagnoses (e.g., Sinus rhythm, Atrial fibrillation).
Limitations
Classifies rhythms only, not structural conditions (e.g., hypertension).
Requires 12-lead ECGs in .dat/.hea format.

Contributing
Open an issue or submit a pull request on GitHub.

License
MIT License. See the  file for details.

Acknowledgments
LUDB for the dataset.
PhysioNet for ECG data.
FastAPI and Google Cloud Run.
