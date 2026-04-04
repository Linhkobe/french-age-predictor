import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import unicodedata
import numpy as np
import logging
import time
import json
from datetime import date
import pandas as pd

## FUNCTION TO LOAD MODEL ASSETS (MODEL ITSELF, TOKENIZER, SCALER)
@st.cache_resource
def load_assets():
    # The model itself
    model = tf.keras.models.load_model("french_age_predictor.h5", compile=False)
    
    # Tokenizer
    with open("age_tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
        
    # Scaler
    import json
    with open("age_scaler.json", "r") as f:
        metadata = json.load(f)
        
    return model, tokenizer, metadata

## FUNCTION TO LOAD THE TREND OF NAMES (GROUPED DATA)
@st.cache_data
def load_trend_data():
    return pd.read_csv('name_trends.csv')

## FUNCTION OF UNCERTAINTY
def predict_with_uncertainty(model, X_input, iterations= 50):
    X_tensor = tf.convert_to_tensor(X_input, dtype = tf.float32)
    predictions = [model(X_tensor, training=True) for _ in range(iterations)]
    predictions = np.array(predictions)
    mean_prediction = np.mean(predictions)
    std_dev = np.std(predictions)
    
    return mean_prediction, std_dev

model, tokenizer, metadata = load_assets()

df_trends = load_trend_data()

min_year = metadata["min_"]
max_year = metadata["max_"]

# STREAMLIT UI DESIGN ---
st.title("French Name Age Predictor")

name_input = st.text_input("First name:", placeholder="e.g., Jean-Kevin")

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)

if name_input:
    start_time = time.time()
    # --- STEP 1: PREPROCESS ---
    name_clean = unicodedata.normalize("NFC", name_input.lower().strip())
    seq = tokenizer.texts_to_sequences([name_clean])
    padded = pad_sequences(seq, maxlen=18, padding="post")
    
    # --- STEP 2: PREDICT ---
    # The model outputs a value between 0 and 1
    pred_scaled = model.predict(padded, verbose=0)[0][0]
    
    mean_scaled, std_dev_scaled = predict_with_uncertainty(model, padded, iterations=50)
    
    # --- STEP 3: REVERSE SCALING ---
    # Formula: Year = (Scaled_Value * (Max - Min)) + Min
    final_year = int((pred_scaled * (max_year - min_year)) + min_year)
    
    # Current date
    current_date = date.today()
    
    # Current year
    current_year = current_date.year
    
    # Predicted age = current_year - predicted year
    final_age = current_year - final_year
    duration_time = time.time() - start_time
    
    # CALCULATE CONFIDENCE INTERVAL
    z_score = 1.96
    
    lower_scaled = mean_scaled - (z_score * std_dev_scaled)
    upper_scaled = mean_scaled + (z_score * std_dev_scaled)
    
    year_low = int((lower_scaled * (max_year - min_year)) + min_year)
    year_high = int((upper_scaled * (max_year - min_year)) + min_year)
    
    # Convert to age range
    age_low = current_year - year_high
    age_high = current_year - year_low
    
    # logger.info(f"PREDICTION_LOG: name= {name_input}, result= {final_year}, latency = {duration_time: .4f}s")
    log_data = {
        "name": name_input,
        "predicted_year": final_year,
        "predicted_age": final_age,
        "latency": round(duration_time,4),
        "model_version": "v1.0"
    }
    logger.info(json.dumps(log_data, ensure_ascii=False))
    
    # --- STEP 4: OUTPUT ---
    st.success(f"Prediction: **{final_year}**")
    st.metric(
        label="Estimated age", 
        value = f"{final_age} years old"
    )
    
    st.subheader("Statistical age analysis")
    st.write(f"95% Confidence interval: between {age_low} and {age_high} years old.")
    
    st.subheader(f"Historical popularity of {name_input}")

    name_history = df_trends[df_trends['first_name'].str.upper() == name_input.upper()]

    if not name_history.empty:
        st.line_chart(name_history.set_index('birth_year')['count'])
    else:
        st.info("No historical data found for this psecific name to show trends.")
        
        
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(name_history['birth_year'], name_history['count'], 
            label='Actual Historical Data', 
            color='#1f77b4', linewidth=2)

    # AI Prediction vertical line
    ax.axvline(x=final_year, color='red', linestyle='--', 
            label=f'AI Prediction ({int(final_year)})')

    ax.set_title(f"Historical Popularity vs. AI Prediction for '{name_input.capitalize()}'", fontsize=14)
    ax.set_xlabel("Year of Birth", fontsize=12)
    ax.set_ylabel("Number of Births (INSEE)", fontsize=12)

    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend()
    st.pyplot(fig)