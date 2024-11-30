import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.datasets import load_breast_cancer

# Load the model and preprocessing objects
with open('ann_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('selected_features.pkl', 'rb') as features_file:
    selected_features = pickle.load(features_file)
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)
with open('selector.pkl', 'rb') as selector_file:
    selector = pickle.load(selector_file)

# Load dataset to get min and max values for sliders
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
min_values = X.min()
max_values = X.max()

# Get the feature names used during model training
feature_names = scaler.feature_names_in_

# Function to preprocess user input
def preprocess_input(input_data, scaler, selector, feature_names):
    # Ensure the input_data is in the same order as the original feature set
    input_df = pd.DataFrame([input_data], columns=feature_names)
    
    # Scale the input data
    input_data_scaled = scaler.transform(input_df)
    
    # Select the top k features
    input_data_selected = selector.transform(input_data_scaled)
    
    return input_data_selected

# Streamlit app
st.title('Breast Cancer Prediction App')

st.write("""
This app allows you to make predictions on the Breast Cancer dataset using a trained Artificial Neural Network (ANN) model.
""")

# User input for making predictions using sliders
st.write("### Adjust the values for the following features:")

input_data = []
for feature in feature_names:
    min_value = float(min_values[feature])
    max_value = float(max_values[feature])
    value = st.slider(f"Adjust value for {feature}", min_value, max_value, (min_value + max_value) / 2)
    input_data.append(value)

# Make prediction
if st.button('Predict'):
    input_data_preprocessed = preprocess_input(input_data, scaler, selector, feature_names)
    prediction = model.predict(input_data_preprocessed)
    prediction_proba = model.predict_proba(input_data_preprocessed)

    st.write(f"### Prediction: {'Malignant' if prediction[0] else 'Benign'}")
    st.write(f"### Prediction Probability: {prediction_proba[0][1]:.4f} (Malignant), {prediction_proba[0][0]:.4f} (Benign)")
