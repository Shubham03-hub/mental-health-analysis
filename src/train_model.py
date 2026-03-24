import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from data_preprocessing import load_data, preprocess_data

df = load_data()
df = preprocess_data(df)

X = df.drop('mental_health_risk', axis=1)
y = df['mental_health_risk']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier()
model.fit(X_train, y_train)
joblib.dump(model, '../models/model_v1.pkl')
print("Model saved!")