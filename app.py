import streamlit as st
import joblib
import numpy as np

# Load the saved model and encoder
model = joblib.load('wealth_index_xgb_model.pkl')
encoder = joblib.load('wealth_index_encoder.pkl')

st.title("DHS Household Wealth Index Predictor")
st.write("Enter household asset features to predict the wealth quintile (`hv270`).")

user_input = st.text_input("Enter feature values separated by commas (e.g., 1, 0, 3, ...):")

if st.button("Predict Wealth Index"):
    try:
        features = np.array([float(x.strip()) for x in user_input.split(",")]).reshape(1, -1)
        encoded_pred = model.predict(features)
        predicted_wealth = encoder.inverse_transform(encoded_pred)
        st.success(f"Predicted Wealth Index: **{predicted_wealth[0]}**")
    except Exception as e:
        st.error(f"Error in prediction: {e}")
