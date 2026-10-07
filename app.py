import streamlit as st
import joblib
import numpy as np

# Load the saved model and encoder
model = joblib.load('wealth_index_xgb_model.pkl')
encoder = joblib.load('wealth_index_encoder.pkl')

st.title("DHS Household Wealth Index Predictor")
st.write("Fill in the household details below to predict the wealth quintile (`hv270`):")

# Interactive form inputs
tv = st.selectbox("Does the household have a Television?", ["No", "Yes"])
car = st.selectbox("Does the household own a Car?", ["No", "Yes"])
electricity = st.selectbox("Does the household have Electricity?", ["No", "Yes"])

# Convert selections to numeric values
tv_val = 1 if tv == "Yes" else 0
car_val = 1 if car == "Yes" else 0
elec_val = 1 if electricity == "Yes" else 0

if st.button("Predict Wealth Index"):
    try:
        # Create an input array matching the exact 6,332 features expected by the model
        input_data = np.zeros((1, 6332))
        
        # Map user inputs to their corresponding feature indices
        input_data[0, 0] = tv_val
        input_data[0, 1] = car_val
        input_data[0, 2] = elec_val

        # Make prediction
        encoded_pred = model.predict(input_data)
        predicted_wealth = encoder.inverse_transform(encoded_pred)
        
        st.success(f"Predicted Wealth Index: **{predicted_wealth[0]}**")
    except Exception as e:
        st.error(f"Error in prediction: {e}")
