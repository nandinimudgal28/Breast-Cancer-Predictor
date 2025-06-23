from xgboost import XGBClassifier
import joblib

def train_model(X_train, y_train):
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    return model

def save_model(model, scaler):
    joblib.dump(model, 'models/final_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
