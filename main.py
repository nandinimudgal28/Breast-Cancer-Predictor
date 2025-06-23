from src.preprocessing import load_data, preprocess
from src.training import train_model, save_model
from src.evaluate import evaluate_model
from src.predictor import predict_cancer

# 1. Load data
df = load_data('data/breast-cancer.csv')

# 2. Feature list (from RFE or selected manually)
selected_features = [
    'concave points_mean', 'radius_se', 'symmetry_se', 'radius_worst',
    'texture_worst', 'concave points_worst', 'area_ratio', 
    'symmetry_change', 'area_to_radius', 'avg_radius'
]

# 3. Preprocess
X_train, X_test, y_train, y_test, scaler = preprocess(df, selected_features)

# 4. Train and Save
model = train_model(X_train, y_train)
save_model(model, scaler)

# 5. Evaluate
evaluate_model(model, X_test, y_test)

# 6. Predict example
sample_input = [0.102, 0.87, 0.02, 25.38, 17.33, 0.27, 1.05, 0.03, 15.8, 13.2]
predict_cancer(sample_input, selected_features)
