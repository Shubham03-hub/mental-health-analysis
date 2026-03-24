import streamlit as st
import joblib

st.title("Mental Health Risk Predictor")

age = st.slider("Age", 18, 60)
sleep = st.slider("Sleep Hours", 0, 10)

if st.button("Predict"):
    model = joblib.load("models/model_v1.pkl")
    prediction = model.predict([[age, sleep]])
    st.write("Predicted Risk:", prediction)