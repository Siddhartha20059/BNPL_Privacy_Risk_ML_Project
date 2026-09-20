# BNPL Privacy Risk Detection using Machine Learning

## Project Overview

BNPL (Buy Now, Pay Later) services collect and process different types of user data. This project develops an academic machine-learning prototype that predicts the privacy risk level of a BNPL service based on privacy-related factors.

The system uses a Random Forest Classifier and provides an interactive Streamlit web application for prediction.

## Objective

The main objectives of this project are:

- Analyze privacy-related factors of BNPL services
- Predict privacy risk as High, Medium, or Low
- Generate a privacy score out of 100
- Show model confidence
- Explain model predictions using SHAP
- Identify important features affecting the prediction

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Random Forest
- Streamlit
- SHAP
- Matplotlib
- Machine Learning

## Dataset

The project uses a synthetic academic dataset containing 1200 records.

The main features are:

- Number of permissions
- Data collection level
- Third-party data sharing
- Advertising/tracking
- Encryption
- Multi-factor authentication (MFA)
- Secure server usage
- Data deletion option

The target variable is:

- High
- Medium
- Low

## Machine Learning Model

A Random Forest Classifier is used for privacy risk prediction.

Configuration:

- Number of trees: 250
- Random state: 42
- Class weighting: Balanced

The model achieved approximately 80% accuracy on the test data.

## Application Features

### 1. Privacy Risk Prediction

Users can enter privacy-related factors through the Streamlit interface.

The application predicts:

- Privacy Score
- Risk Level
- Privacy Category

### 2. Model Confidence

The application displays the confidence of the Random Forest prediction.

### 3. Feature Importance

The application displays the importance of different features used by the machine-learning model.

### 4. Explainable AI using SHAP

SHAP is used to explain how individual features contribute to the model prediction.

This improves the interpretability of the machine-learning system.

## Project Structure

```text
BNPL_Privacy_Risk_ML_Project/
│
├── data/
│   └── bnpl_privacy_dataset.csv
│
├── app.py
├── train.py
├── explain.py
├── requirements.txt
└── README.md