import joblib
import numpy as np

def predict_cancer(input_data, selected_features):
    model = joblib.load('models/final_model.pkl')
    scaler = joblib.load('models/scaler.pkl')

    input_scaled = scaler.transform([input_data])
    prediction = model.predict(input_scaled)[0]
    result = "🩺 Malignant (Cancerous)" if prediction == 1 else "✅ Benign (Non-Cancerous)"
    print("\n🎯 Prediction:", result)
    return result
