# 🔥 Forest Fire Area Prediction Using Machine Learning

This project focuses on predicting the **burned area of forest fires** in the northeast region of Portugal using various machine learning regression models. The objective is to support early wildfire risk management and mitigation efforts by providing predictive insights.

---

## 📁 Project Structure

ML_PROJECT/
│
├── artifacts/ # Saved artifacts like models, scalers, etc.
├── data/ # Raw and cleaned data
│ ├── raw.csv
│ └── data.csv
├── notebooks/ # Jupyter notebooks for EDA and experimentation
├── src/ # Source code package
│ ├── components/ # Data transformation and model training modules
│ ├── pipeline/ # Prediction and training pipeline
│ ├── utils.py # Utility functions
│ └── logger.py # Custom logging
│
├── app.py # Streamlit app for UI
├── Dockerfile # Docker configuration for deployment
├── requirements.txt # Project dependencies
├── setup.py # Python package setup
└── README.md # Project documentation


---

## 📊 Dataset

- Source: UCI Machine Learning Repository — [Forest Fires Dataset](https://archive.ics.uci.edu/ml/datasets/Forest+Fires)
- Features:
  - Weather conditions (e.g., temperature, wind, RH)
  - FWI System Codes (e.g., FFMC, DMC, DC, ISI)
  - Temporal features (month, day)
- Target:
  - `area` burned (in hectares)

---

## 🚀 Features

- 📌 Data ingestion and preprocessing pipeline
- 📈 Multiple regression models trained (Linear, Decision Tree, etc.)
- 📊 Model evaluation using metrics like MAE, RMSE
- 🌐 Streamlit web app for real-time predictions
- 🐳 Dockerized for easy deployment

---

## 🧪 How to Run the Project

### 🔧 Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Dinesh-jangir/ML_PROJECT.git
   cd ML_PROJECT
2 .Create and activate a virtual environment:

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

