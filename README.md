#  Breast Cancer Tumor Classification - ML & Deep Learning Based Predictor

This project implements a robust Breast Cancer Prediction system using both **Machine Learning (ML)** and **Deep Learning (DL)** techniques to classify tumors as **Benign (non-cancerous)** or **Malignant (cancerous)** using the **Breast Cancer Wisconsin dataset**.

---

## 📌 Project Highlights

-  **10-feature selection** based on Recursive Feature Elimination (RFE) and SHAP importance
-  Compared multiple ML models: SVM, Logistic Regression, Random Forest, XGBoost
-  Achieved high accuracy and AUC using deep learning (Keras + TensorFlow)
-  Visualized model interpretability using SHAP, heatmaps, and confusion matrices
-  Built a real-time **prediction interface using Streamlit**
-  Integrated into a local **VS Code deployment environment**
-  Final model and scaler saved via `joblib` for production use

---

## 🔧 Tech Stack

- **Python**
- **Pandas, NumPy, Matplotlib, Seaborn** for EDA & visualization
- **Scikit-learn** for ML models, preprocessing, and evaluation
- **TensorFlow / Keras** for deep learning model
- **SHAP** for model explainability
- **Streamlit** for web deployment
- **Joblib** for model serialization
- **VS Code** and **Google Colab** for development

---

## 📈 Models Compared

| Model               | Accuracy | AUC Score |
|--------------------|----------|-----------|
| Logistic Regression| ✅ High  | ✅ Good   |
| SVM (RBF)          | ✅ High  | ✅ High   |
| Random Forest      | ✅ Stable| ✅ Good   |
| XGBoost            | ✅ Best  | ✅ Best   |
| Keras Neural Net   | ✅ 95.6% | ✅ Very Good |

---

## 🧪 SHAP & Feature Engineering

- Used SHAP to interpret predictions
- Created new features like:
  - `area_to_radius = area_mean / radius_mean`
  - `symmetry_change = symmetry_worst - symmetry_mean`
- Performed correlation analysis and removed redundant features

---

## 🚀 Streamlit Web App

- Collects user input for 10 top features
- Predicts class with **probability breakdown**
- Displays intuitive outputs:  
  - 🟢 Benign or 🔴 Malignant  
  - Confidence score in %
- Run locally via:

Breast cancer predictor/
│
├── models/
│   ├── final_model.pkl
│   └── scaler.pkl
│
├── app.py
├── README.md
└── breast-cancer.csv

## Future Enhancements

-CSV batch upload and result download

-Auto PDF report generation

-Deployment on Hugging Face or Streamlit Cloud




![image](https://github.com/user-attachments/assets/4002d218-05a9-410e-bfd7-da2476aa1b19)

![image](https://github.com/user-attachments/assets/ad8a48b7-4904-480a-bec2-105b2edd381b)


```bash
streamlit run app.py


