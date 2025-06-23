import numpy as np
import streamlit as st
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'final_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, '..', 'models', 'scaler.pkl')

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


selected_features = [
    'concave points_mean', 'radius_se', 'symmetry_se', 'radius_worst',
    'texture_worst', 'concave points_worst', 'area_ratio',
    'symmetry_change', 'area_to_radius', 'avg_radius'
]

st.set_page_config(page_title=" Breast Cancer Prediction", layout="centered")
st.title("🩺 Breast Cancer Predictor")
st.markdown("""
Enter diagnostic data below to predict whether the tumor is **Benign (Non-Cancerous)** or **Malignant (Cancerous)**.
""")

user_input = []
for feature in selected_features:
    val = st.number_input(f"🔹 {feature}", min_value=0.0, step=0.01, format="%.5f")
    user_input.append(val)

if st.button(" Predict"):
    try:
        input_array = np.array([user_input])
        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)[0]
        confidence = model.predict_proba(input_scaled)[0][prediction]

        result_label = " Malignant (Cancerous)" if prediction == 1 else " Benign (Non-Cancerous)"
        st.success(f" Prediction: {result_label}")
        st.info(f"Model Confidence: {confidence * 100:.2f}%")

    except Exception as e:
        st.error(f" Error during prediction: {e}")
