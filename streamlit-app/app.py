import boto3
import streamlit as st
import pandas as pd
import logging.config
from src.aws_utils import fetch_model_from_s3
from src.load_config import get_config

# Set up logging
logging.config.fileConfig('config/logging.conf')

config = get_config('config/config.yaml')

# AWS S3 Setup
S3_BUCKET_NAME = config['aws']['s3_bucket']
PREFIX = config['aws']['bucket_prefix']
MODEL_VERSIONS_LIST = config['aws']['model_versions']

# Initialize S3 client
s3_client = boto3.client('s3')

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        color: white;
        background-color: #0073e6;
    }
    .stTextInput>div>div>input {
        color: black;
        background-color: #dfe3ee;
    }
    </style>
    """, unsafe_allow_html=True)

# Streamlit UI
st.title("Let's Predict!!!")
st.markdown("### Which model do you want to use??", unsafe_allow_html=True)

# Model selection
chosen_model_version = st.selectbox('Choose Model Version', MODEL_VERSIONS_LIST)
logging.info(f'Selected model version: {chosen_model_version}')

# Load model
model = fetch_model_from_s3(S3_BUCKET_NAME, PREFIX, chosen_model_version)
logging.info(f'Model loaded successfully: {chosen_model_version}')

# Feature input section with columns
st.markdown("### Adjust the features as needed", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    log_entropy = st.number_input('log visible entropy', value=0, step=1)

with col2:
    IR_norm_range = st.number_input('IR norm range', value=0, step=1)

with col3:
    entropy_x_contrast = st.number_input('visible contrast x visible entropy', value=0, step=1)

# Generate predictions
if st.button('Predict'):
    features = pd.DataFrame([[log_entropy, IR_norm_range, entropy_x_contrast]], 
                            columns=['log_visible_entropy', 'IR_norm_range', 'visible_contrast_x_visible_entropy'])
    
    try:
        prediction = model.predict(features)
        st.markdown(f'### For these Features the Prediction is: {prediction[0]}', unsafe_allow_html=True)
        logging.info(f'Successful prediction with features: {features.values.tolist()} - Prediction: {prediction[0]}')
    except Exception as err:
        st.error(f"Error occurred during prediction: {err}")
        logging.error(f"Prediction error: {err}", exc_info=True)
