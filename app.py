import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# --- Path setup ---
# Get the absolute path to the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Define the path to the model file
MODEL_PATH = os.path.join(BASE_DIR, 'linreg_house_price.pkl')

print("Looking for model at:", MODEL_PATH)
print("Model exists:", os.path.exists(MODEL_PATH))
print("Current working dir:", os.getcwd())

# Load the model pipeline
@st.cache_resource
def load_pipeline():
    """Loads the pre-trained pipeline object."""
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

pipeline = load_pipeline()

# --- Page Setup ---
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered",
    initial_sidebar_state="expanded"
)

if pipeline is None:
    st.error(
        f"**Model not found.** The app requires the `linreg_house_price.pkl` file. "
        f"Please make sure it's in the same directory as the app script.\n\n"
        f"Looking for model at: `{MODEL_PATH}`"
    )
    st.stop()

# --- Sidebar for Project Intro --- 
st.sidebar.title("🏠 House Price Prediction Project")
st.sidebar.markdown(
    """
    Welcome to the House Price Predictor app! This tool uses a Linear Regression model 
    trained on various housing features to estimate property prices. 
    
    Adjust the parameters in the main section to see how different house 
    characteristics influence the predicted price. 
    
    **Features considered:** Area, number of bedrooms, bathrooms, stories, 
    road access, guest room, basement, water heating, air conditioning, parking, 
    preferred area, and furnishing status.

    Created by: Dhiraj Mandal
    """
)
st.sidebar.info("Navigate to the main section to input house features.")

# --- Main Section for Prediction Parameters --- 
st.title("🏡 Predict Your House Price")
st.markdown("### Enter the details of the house below:")

# Group inputs for better layout
col1, col2 = st.columns(2)

with col1:
    area = st.number_input("Area (sq ft)", min_value=1000, max_value=20000, value=7420, step=100)
    bedrooms = st.slider("Number of Bedrooms", min_value=1, max_value=6, value=4)
    bathrooms = st.slider("Number of Bathrooms", min_value=1, max_value=4, value=2)
    stories = st.slider("Number of Stories", min_value=1, max_value=4, value=2)
    parking = st.slider("Parking Spaces", min_value=0, max_value=3, value=2)

with col2:
    st.markdown("##### Binary Features")
    mainroad = st.checkbox("Main Road Access", value=True)
    guestroom = st.checkbox("Guest Room", value=False)
    basement = st.checkbox("Basement", value=True)
    hotwaterheating = st.checkbox("Hot Water Heating", value=False)
    airconditioning = st.checkbox("Air Conditioning", value=True)
    prefarea = st.checkbox("Preferred Area", value=True)
    
    st.markdown("##### Furnishing Status")
    furnishingstatus = st.selectbox(
        "Select Furnishing Status", 
        ('furnished', 'semi-furnished', 'unfurnished'),
        index=0 # Default to 'furnished'
    )

# --- Create Input DataFrame ---
# The pipeline is expected to handle all preprocessing, including one-hot encoding.
# We can create a DataFrame with the raw inputs.
input_data = pd.DataFrame({
    'area': area,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'stories': stories,
    'mainroad': 'yes' if mainroad else 'no',
    'guestroom': 'yes' if guestroom else 'no',
    'basement': 'yes' if basement else 'no',
    'hotwaterheating': 'yes' if hotwaterheating else 'no',
    'airconditioning': 'yes' if airconditioning else 'no',
    'parking': parking,
    'prefarea': 'yes' if prefarea else 'no',
    'furnishingstatus': furnishingstatus
}, index=[0])

st.markdown("--- ")

# Prediction button in the main section
if st.button("Predict Price"):
    try:
        predicted_price = pipeline.predict(input_data)[0]
        st.success(f"### Predicted House Price: **₹ {predicted_price:,.2f}**")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

st.markdown("--- ")
st.info("Adjust the features above and click 'Predict Price' to get an estimate.")

# Optional: Display input data (for debugging/transparency)
# with st.expander("View Input Data"): 
#     st.dataframe(input_data)

# Optional: Add some styling or additional information
st.markdown(
    """
    <style>
    .st-emotion-cache-z5fcl4 {
        padding-top: 1rem;
    }
    </style>
    """, unsafe_allow_html=True
)
