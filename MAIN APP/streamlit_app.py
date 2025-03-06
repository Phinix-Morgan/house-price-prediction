import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load trained model
# final_model = pickle.load(open("house_price_model_kaggle.pkl", "rb"))



import os
import pickle

# Get the directory where the script (streamlit_app.py) is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the model file relative to the script
model_path = os.path.join(script_dir, "house_price_model_kaggle.pkl")

# Check if the file exists before loading
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")
else:
    final_model = pickle.load(open(model_path, "rb"))





st.title("🏡 House Price Prediction App Kaggle ")
	
# User Input
square_feet = st.number_input("Enter Square Feet:", min_value=500, max_value=5000)
bedrooms = st.slider("Number of Bedrooms:", 1, 5)
year_built = st.number_input("Enter Year Built:", min_value=1900, max_value=2023)
Lot_Size = st.number_input("Enter Lot_Size:", min_value=0.0, max_value=5.0, value=0.0, step=0.1)
Garage_Size = st.number_input("Enter Garage_Size:", min_value=0, max_value=3)

if st.button("Predict Price"):
    # Create a NumPy array with the input values (should have shape (1, 3))
    new_house = np.array([[square_feet, bedrooms, year_built,Lot_Size, Garage_Size]])

    
    # Define the same feature names used during model training
    feature_names = ["Square_Footage", "Num_Bedrooms", "Year_Built",'Lot_Size', 'Garage_Size']
    
    # Debug: check that the number of features matches the length of feature_names
    # if new_house.shape[1] != len(feature_names):
        #st.error("Mismatch between input features and feature names!")
    # else:
        # Convert the NumPy array to a DataFrame with the correct column names
    new_house_df = pd.DataFrame(new_house, columns=feature_names)
        #st.write("New house DataFrame:", new_house_df)
    
        # Use the DataFrame for prediction
    final_prediction = final_model.predict(new_house_df)
    st.success(f"Predicted Price: ${final_prediction[0]:,.2f}")
    