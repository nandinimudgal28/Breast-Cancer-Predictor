import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def load_data(path):
    df = pd.read_csv(path)
    df.drop(columns=['id'], errors='ignore', inplace=True)

    le = LabelEncoder()
    df['diagnosis'] = le.fit_transform(df['diagnosis'])  # B=0, M=1

    return df

def preprocess(df, selected_features=None):
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']

    if selected_features:
        X = X[selected_features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, scaler
