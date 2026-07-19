import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Load saved model, scaler, and columns (relative to repo root's /models folder)
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

model = pickle.load(open(os.path.join(MODEL_DIR, "stroke_model.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb"))
model_columns = pickle.load(open(os.path.join(MODEL_DIR, "model_columns.pkl"), "rb"))

st.title("🧠 Stroke Risk Prediction App")
st.markdown("Provide the patient's details below:")

# Inputs from user
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
age = st.slider("Age", 0, 100, 30)
hypertension = st.selectbox("Hypertension", [0, 1])
heart_disease = st.selectbox("Heart Disease", [0, 1])
ever_married = st.selectbox("Ever Married", ["Yes", "No"])
work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
avg_glucose_level = st.number_input("Average Glucose Level", 50.0, 300.0, 100.0)
bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
smoking_status = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])

# Submit
if st.button("Predict Stroke Risk"):
    # Build a single-row DataFrame
    input_df = pd.DataFrame({
        'gender': [gender],
        'age': [age],
        'hypertension': [hypertension],
        'heart_disease': [heart_disease],
        'ever_married': [ever_married],
        'work_type': [work_type],
        'Residence_type': [residence_type],
        'avg_glucose_level': [avg_glucose_level],
        'bmi': [bmi],
        'smoking_status': [smoking_status]
    })

    # One-hot encode
    input_encoded = pd.get_dummies(input_df)

    # Reindex to match training features
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

    # Scale
    input_scaled = scaler.transform(input_encoded)

    # Predict
    proba = model.predict_proba(input_scaled)[0][1]
    st.write(f"🔍 **Predicted Stroke Probability:** `{proba:.2f}`")

    # Threshold decision
    if proba > 0.3:
        st.error("⚠️ High risk of stroke. Immediate attention recommended.")
    else:
        st.success("✅ Low risk of stroke.")
