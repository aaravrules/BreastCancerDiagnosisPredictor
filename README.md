# BreastCancerDiagnosisPredictor

This repository contains the code for a machine learning-based tool designed to predict breast cancer diagnosis. The project utilizes a two-layer Random Forest Classifier model to analyze clinical data and predict whether a patient has breast cancer.

## Description

The `BreastCancerDiagnosisPredictor` uses clinical measurements related to breast cancer characteristics to predict the likelihood of a breast cancer diagnosis. The model has been trained on a comprehensive dataset and provides an interface for inputting patient data for prediction.

## Getting Started

### Dependencies

- Python 3.x
- NumPy
- scikit-learn
- joblib

### Installing

- Clone this repository to your local machine.
- Ensure Python 3.x is installed on your system.
- Install the required Python packages:
  ```bash
  pip install numpy scikit-learn joblib
  ```

### Executing the Program

- Run the main script from the command line:
  ```bash
  python BreastCancerDiagnosisPredictor.py
  ```
- Follow the prompts to input patient data.
- The program will output a prediction for breast cancer diagnosis.

## How to Use

1. Start the script.
2. You will be prompted to enter values for various clinical features, such as radius mean, texture mean, etc.
3. For each feature, you will also be asked whether the entered value is null.
4. After all inputs are provided, the script will display the prediction result.

## Models

The repository includes two Random Forest Classifier models saved in `.sav` format:
- `finalized_modelLayer1.sav`
- `finalized_modelLayer2.sav`

These models are loaded by the script to make predictions based on the input data.

## Authors

Aarav Sharma

