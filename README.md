# Symptom-Based Disease Predictor

This project aims to provide a simple app that will visualize and predict diseases using a machine learning model trained on previous symptom/disease data.

## Features
- **SQLite Database**: Data is stored in a sql database, `disease_symptoms.db`.
- **Interactive User Input**: Users are able to enter any combination of available symptoms to get predictions.
- **Model Output**: The model outputs into a table that shows the top-10 most likely diseases and their respective probabilities.
- **Diagnostics**: Using symptom_classifier.py, users are able to evaluate model performance on the different diseases.

## Model Overview

In symptom_classifier.py, two machine learning models are trained, a Logistic Regression model and a Random Forest model. The script then splits data into training and testing splits (80/20), and picks the model that performs best on the test data based on prediction accuracy.

## Installation
1. Clone the repository.
2. Ensure that the python environment contains `streamlit`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, and `sqlalchemy`. If uninstalled, they can be installed using `pip install (enter library)`.
3. To run the app, navigate to the repo directory and use `streamlit run app.py`.
4. To view model diagnostics, go to the same directory and use `streamlit run symptom_classifier.py`.

## Other Files
`artifacts`: A folder that stores the confusion matrix visualization from the model.
`data`: A folder containing the training and testing datasets.
`etl_to_sqlite`: A python script that loads, cleans, and exports the original dataset to sqlite.
`label_encoder.joblib`: File that does one-hot encoding in the symptom-disease dataset.
`pyvenv.cfg`: A config file detailing how virtual environment is setup.
`symptom_model.joblib`: A file that stores the selected model after running the symptom_classifier.py script.

## Requirements
- Python 3.7+
- Streamlit
- Scikit-learn
- Pandas
- Numpy
- SQLite
- SQLalchemy

## Notes
- This tool is for educational/demo purposes only and **not intended for medical diagnosis**.
- The quality of predictions is dependent on the selected dataset and model training, with larger, more diverse datasets improving prediction accuracy.
