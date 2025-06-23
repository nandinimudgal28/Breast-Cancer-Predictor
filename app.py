import streamlit as st
import numpy as np
import joblib

#  Load trained model and scaler
model = joblib.load("models/final_model (2).pkl")
scaler = joblib.load("models/scaler (3).pkl")

#  Features used in the 10-feature model
features = [
    "concave points_mean", "radius_se", "symmetry_se",
    "radius_worst", "texture_worst", "concave points_worst",
    "area_mean", "area_worst", "compactness_mean", "smoothness_mean"
]

#  UI
st.title("🔬 Breast Cancer Prediction Tool")
st.markdown("Enter the values for the following medical features:")

#  Input section
input_values = []
for feature in features:
    val = st.number_input(f"{feature}", format="%.4f", step=0.01)
    input_values.append(val)

#  Prediction
if st.button(" Predict"):
    input_array = np.array(input_values).reshape(1, -1)

    # Scale input data
    input_scaled = scaler.transform(input_array)

    # Predict label and probability
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0]

    # Output
    label = " Benign (Non-Cancerous)" if prediction == 0 else " Malignant (Cancerous)"
    st.success(f"Prediction Result: **{label}**")

    # Show probabilities
    st.markdown("###  Prediction Probabilities")
    st.write(f" Benign: **{proba[0]*100:.2f}%**")
    st.write(f" Malignant: **{proba[1]*100:.2f}%**")
